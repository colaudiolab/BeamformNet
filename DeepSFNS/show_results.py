import argparse
from pathlib import Path

import numpy as np
import scipy
import torch
import yaml
from matplotlib import pyplot as plt
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.gen_data import get_arrangement
from models.daMUSIC import DeepAugmentMusic
from models.deepSFNS import DeepSFNS
from models.lowSNRNet import DOALowSNRNet
from utils.dataset import DOADataset
from utils.util import create_worker_seed_val

from utils.util import get_steer_vector


plt.rcParams.update({
    'font.size': 14,               # 默认字体大小（影响 legend、tick labels 等）
    'axes.labelsize': 16,          # x 和 y 轴标签字体大小
    'axes.titlesize': 18,          # 图标题字体大小
    'legend.fontsize': 12,         # 图例字体大小
    'xtick.labelsize': 14,         # x 轴刻度数字字体大小
    'ytick.labelsize': 14,       # y 轴刻度数字字体大小
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab20.colors)  # 添加颜色循环

})


def normalize_minmax(array):
    """最小-最大归一化到0-1范围"""
    array_min = array.min()
    array_max = array.max()

    # 避免除零错误
    if array_max - array_min == 0:
        return np.zeros_like(array)

    return (array - array_min) / (array_max - array_min)


def classicMUSIC(incident, arrangment, thetas, f, c, sources, phi = 0):
    # # calculate EVD of covariance matrix
    # covariance = np.cov(incident)
    covariance = incident @ incident.conj().T


    eigenvalues, eigenvectors = linalg.eig(covariance)

    # number of sources known
    d = sources


    #排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # 对 eigenvalues 进行排序

    sorted_eigenvalues = eigenvalues[sorted_indices]

    # 对 eigenvectors 进行排序

    sorted_eigenvectors = eigenvectors[:,sorted_indices]


    # the noise matrix
    En = sorted_eigenvectors[:, d:]

    EN = En @ En.conj().transpose()
    # calculate spatial spectrum
    numSamples = len(thetas)
    spectrum = np.zeros(numSamples)
    for i in range(numSamples):
        # establish array steering vector
        a = get_steer_vector(arrangment, thetas[i], phi, f, c)
        spectrum[i] = 1. / (a.conj().transpose() @ EN @ a)

    # plt.figure(figsize=(12, 6))
    # plt.plot(spectrum, label='result_MUSIC_energy', color='blue')
    # # plt.scatter((thetas / np.pi) * 180 + 180, np.ones(thetas.shape[0]) * max(spectrum), color='green')
    # plt.title('MUSIC_energy')
    # plt.xlabel('degree')
    # plt.ylabel('energy')
    # plt.grid(True)
    # plt.show()

    return spectrum

parser = argparse.ArgumentParser(description='Run Script')
parser.add_argument('--gen_data_config_path', type=str, help='Path to the configuration file')
parser.add_argument('--train_config_path', type=str, help='Path to the configuration file')
parser.add_argument('--model_config_path', type=str, help='Path to the configuration file')
parser.add_argument('--validate_config_path', type=str, help='Path to the configuration file')
args = parser.parse_args()

gen_data_config_path = Path(args.gen_data_config_path)
train_config_path = Path(args.train_config_path)
model_config_path = Path(args.model_config_path)
validate_config_path = Path(args.validate_config_path)

with open(gen_data_config_path, 'r') as file:
    gen_data_config = yaml.safe_load(file)

with open(train_config_path, 'r') as file:
    train_config = yaml.safe_load(file)

with open(model_config_path, 'r') as file:
    model_config = yaml.safe_load(file)

with open(validate_config_path, 'r') as file:
    validate_config = yaml.safe_load(file)


arrangement = get_arrangement(gen_data_config)

val_dataset = DOADataset(arrangement = arrangement, num_samples = 10000, gen_data_config=gen_data_config, is_cov=True, is_Rx_tou=False,is_eigenvalue=False)
batch_size = train_config['bs']
test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,
                         worker_init_fn=create_worker_seed_val(5201314))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

deepSFNS_model = DeepSFNS(gen_data_config, model_config, train_config).to(device)
# #加载预训练权重
deepSFNS_model.load_state_dict(torch.load('./best_model.pth', map_location=device))
deepSFNS_model.eval()

lowSNRNet_model = DOALowSNRNet(
    len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p'])))).to(device)
lowSNRNet_model.load_state_dict(torch.load('./best_model.pth', map_location=device))
lowSNRNet_model.eval()

daMUSIC_model = DeepAugmentMusic(gen_data_config['m'], gen_data_config['c'] / (2 * gen_data_config['f']),
                         gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']).to(device)
daMUSIC_model.load_state_dict(torch.load('./best_model.pth', map_location=device))
daMUSIC_model.eval()


with torch.no_grad():
    for batch in tqdm(test_loader):
        signal = batch['signal'].to(device)
        labels = batch['label'].numpy()
        cov = batch['cov'].to(device)
        steeringMatrix_A = batch['steeringMatrix_A'].to(device)

        deepSFNS_outputs = deepSFNS_model(steeringMatrix_A, signal)
        lowSNRNet_outputs = lowSNRNet_model(cov).detach().cpu().numpy()
        daMUSIC_outputs = daMUSIC_model(signal)


        signal = batch['signal'].numpy()


        spectrums = torch.atanh(torch.clamp(deepSFNS_outputs, -1 + 1e-6, 1 - 1e-6)).detach().cpu().numpy()  # 反atanh得到空间能量谱
        deepSFNS_outputs = deepSFNS_outputs.detach().cpu().numpy()
        for index in range(batch['label'].shape[0]):
            d = len(np.where(labels[index] == 1)[0])
            spectrum = spectrums[index]
            spectrum_copy = np.copy(spectrum)
            lowSNRNet_probability = lowSNRNet_outputs[index]
            deepSFNS_probability = deepSFNS_outputs[index]

            daMUSIC_doa = daMUSIC_outputs[index][:d].detach().cpu().numpy()
            daMUSIC_doa = daMUSIC_doa * 180 / np.pi + 90

            # 过滤空间噪声 即低于阈值0.5认为不存在目标
            spectrum[np.where(deepSFNS_outputs[index] <= validate_config['td'])[0]] = 0

            classicMUSIC_outputs = classicMUSIC(signal[index], arrangement, list(
                np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']) * np.pi / 180),
                                                gen_data_config['f'], gen_data_config['c'], d, 0)

            plt.figure(figsize=(8, 6))
            plt.vlines(x=daMUSIC_doa,  # x位置数组
                       ymin=0, ymax=1,  # 线的起始和结束y坐标
                       colors='gray',  # 颜色
                       linestyles='--',  # 线型
                       alpha=0.7,  # 透明度
                       linewidths=1.5,  # 线宽
                       label='daMUSIC'
                       )
            plt.plot(normalize_minmax(spectrum), label='BeamformNet', color='blue')
            plt.plot(normalize_minmax(lowSNRNet_probability), label='lowSNRNet', color='red', alpha=0.5)
            plt.plot(normalize_minmax(classicMUSIC_outputs), label='MUSIC', color='green', alpha=0.5)
            thetas = np.where(labels[index] == 1)[0]

            plt.vlines(x=daMUSIC_doa,  # x位置数组
                       ymin=0, ymax=1,  # 线的起始和结束y坐标
                       colors='gray',  # 颜色
                       linestyles='--',  # 线型
                       alpha=0.7,  # 透明度
                       linewidths=1.5,  # 线宽
                       )
            plt.scatter(thetas, np.ones(thetas.shape[0]), color='green', label='true DOA')
            plt.vlines(x=thetas,  # x位置数组
                       ymin=0, ymax=1,  # 线的起始和结束y坐标
                       colors='green',  # 颜色
                       linestyles='--',  # 线型
                       alpha=0.5,  # 透明度
                       linewidths=1,  # 线宽
                       )

            # plt.title('deepSFNS_energy')

            # 设置x轴范围为0-180
            plt.xlim(0 + 30, len(spectrum) - 1 - 29)

            # 修改x轴刻度和标签
            xticks = np.arange(0 + 30, len(spectrum) - 29, 30)  # 每30度一个刻度
            xticklabels = xticks - 90  # 将0-180转换为-90到90
            plt.xticks(xticks, xticklabels)

            plt.xlabel('Angle')
            plt.ylabel('Spectrum')
            plt.legend()
            plt.grid(True)
            plt.show()





