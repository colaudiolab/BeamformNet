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


def capon_getB(incident, arrangment, thetas, f, c, phi = 0):
    covariance = incident @ incident.conj().T
    covariance_inverse = np.linalg.inv(covariance)

    numSamples = len(thetas)
    B = np.zeros((numSamples, arrangment.shape[1]),np.complex128)
    for i in range(numSamples):
        # establish array steering vector
        a = get_steer_vector(arrangment, thetas[i], phi, f, c)
        b = covariance_inverse @ a / (a.conj().transpose() @ covariance_inverse.conj().T @ a)
        B[i] = b.conj().T

    return B


def CBF_getB(incident, arrangment, thetas, f, c, phi = 0):
    covariance = incident @ incident.conj().T
    covariance_inverse = np.linalg.inv(covariance)

    numSamples = len(thetas)
    B = np.zeros((numSamples, arrangment.shape[1]),np.complex128)
    for i in range(numSamples):
        # establish array steering vector
        a = get_steer_vector(arrangment, thetas[i], phi, f, c)
        b = a / arrangment.shape[1]
        B[i] = b.conj().T

    return B


# 使用numpy广播的向量化版本
def normalize_rows_vectorized(W):
    """向量化版本的行归一化"""
    row_mins = W.min(axis=1, keepdims=True)
    row_maxs = W.max(axis=1, keepdims=True)
    ranges = row_maxs - row_mins
    # 避免除零
    ranges[ranges == 0] = 1
    return (W - row_mins) / ranges

def normalize_2d_array_global(W):
    """将整个二维数组归一化到0-1"""
    W_min = W.min()
    W_max = W.max()
    if W_max - W_min == 0:
        return np.zeros_like(W)
    return (W - W_min) / (W_max - W_min)

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
batch_size = 256
test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,
                         worker_init_fn=create_worker_seed_val(5201314))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arrangement = get_arrangement(gen_data_config)


deepSFNS_model = DeepSFNS(gen_data_config, model_config, train_config).to(device)
# #加载预训练权重
deepSFNS_model.load_state_dict(torch.load('./best_model.pth', map_location=device))
deepSFNS_model.eval()
snr = -15

with torch.no_grad():
    for batch in tqdm(test_loader):
        Y = batch['signal'].to(device)
        labels = batch['label'].numpy()
        A = batch['steeringMatrix_A'].to(device)

        signal = batch['signal'].numpy()
        noise = batch['noise'].numpy()
        orignal_signal_power = np.mean(np.abs(signal-noise) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        SNR = 10 * np.log10(orignal_signal_power / noise_power)
        print(f'oringal SNR = {SNR} dB')

        A_real = A.real
        A_imag = A.imag
        A = torch.cat((A_real, A_imag), dim=1)  # out A (batch,2*m,n)
        Y_real = Y.real
        Y_imag = Y.imag
        Y = torch.cat((Y_real, Y_imag), dim=1)  # out Y (batch,2*m,snapshots)
        A = A.float()
        Y = Y.float()
        A = deepSFNS_model.Ann1(A)  # out A (batch,k,n)
        Y = deepSFNS_model.Ynn2(Y)  # out Y (batch,k,snapshots)
        A_v = deepSFNS_model.W_v(A)  # out A_v (batch,k,n)
        A_k = deepSFNS_model.W_k(A)  # out A_k (batch,k,n)
        Y_q = deepSFNS_model.W_q(Y)  # out Y_q (batch,k,n)
        # 计算注意力分数
        scores = torch.matmul(Y_q, A_k.transpose(-2, -1)) / deepSFNS_model.scale  # out scores (batch,k,k)
        attn_weights = torch.softmax(scores, dim=-1)
        # 加权求和
        B = torch.matmul(attn_weights, A_v)  # out B (batch_size, k, n)
        B = deepSFNS_model.Bnn3(B)  # out B (batch_size, k, n)
        B = B.permute(0, 2, 1)  # out B (batch_size, n, k)
        B = deepSFNS_model.W_b(B)  # out B (batch_size, n, 2 * m)
        B_real = B[:, :, :deepSFNS_model.m]
        B_imag = B[:, :, deepSFNS_model.m:]
        Y_origin = torch.complex(Y_real.float(), Y_imag.float())
        B = torch.complex(B_real, B_imag)
        A = torch.complex(A_real.float(), A_imag.float())


        new_noise = B.detach().cpu().numpy() @ noise
        noise_power = np.mean(np.abs(new_noise) ** 2)
        SNR = 10 * np.log10(orignal_signal_power / noise_power)
        print(f'BeamformNet SNR = {SNR} dB')

        W = torch.mean(torch.abs(B @ A),dim=0).detach().cpu().numpy()

        W = normalize_2d_array_global(W)

        plt.figure(figsize=(10, 8))
        plt.imshow(W, cmap='viridis', extent=[-90, 90, 90, -90], aspect='auto')
        plt.colorbar(label='Value')
        plt.title(F'BeamformNet SNR={snr}dB')
        plt.xlabel('Angle')
        plt.ylabel('Angle')
        plt.show()

        B_list = []
        for index in range(batch['label'].shape[0]):
            signal = batch['signal'].numpy()[index]
            B = capon_getB(signal,arrangement,list(
                np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']) * np.pi / 180),
                                                gen_data_config['f'], gen_data_config['c'], 0)
            B_list.append(B)

        B_list = np.array(B_list)

        new_noise = B_list @ noise
        noise_power = np.mean(np.abs(new_noise) ** 2)
        SNR = 10 * np.log10(orignal_signal_power / noise_power)
        print(f'MVDR SNR = {SNR} dB')

        W = np.mean(np.abs(B @ batch['steeringMatrix_A'].numpy()), axis=0)

        W = normalize_2d_array_global(W)

        plt.figure(figsize=(10, 8))
        plt.imshow(W, cmap='viridis', extent=[-90, 90, 90, -90], aspect='auto')
        plt.colorbar(label='Value')
        plt.title(f'MVDR SNR={snr}dB')
        plt.xlabel('Angle')
        plt.ylabel('Angle')
        plt.show()

        # plt.figure(figsize=(8, 8))
        # plt.plot(np.mean(np.mean(np.abs(B_list @ batch['signal'].numpy()),axis=0),axis=1))
        # plt.show()
        #
        #
        # for i in range(W.shape[0]):
        #
        #     plt.figure(figsize=(8, 8))
        #     plt.plot(W[i])
        #     plt.show()

        B_list = []
        for index in range(batch['label'].shape[0]):
            signal = batch['signal'].numpy()[index]
            B = CBF_getB(signal,arrangement,list(
                np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']) * np.pi / 180),
                                                gen_data_config['f'], gen_data_config['c'], 0)
            B_list.append(B)

        B_list = np.array(B_list)

        new_noise = B_list @ noise
        noise_power = np.mean(np.abs(new_noise) ** 2)
        SNR = 10 * np.log10(orignal_signal_power / noise_power)
        print(f'CBF SNR = {SNR} dB')

        W = np.mean(np.abs(B @ batch['steeringMatrix_A'].numpy()), axis=0)

        W = normalize_2d_array_global(W)

        plt.figure(figsize=(10, 8))
        plt.imshow(W, cmap='viridis', extent=[-90, 90, 90, -90], aspect='auto')
        plt.colorbar(label='Value')
        plt.title(f'CBF SNR={snr}dB')
        plt.xlabel('Angle')
        plt.ylabel('Angle')
        plt.show()

        print()



