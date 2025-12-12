import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.gen_data import get_arrangement
from models.IQResNet import IQResNet, IQResNet_val
from models.broadband_music import B_music_val
from models.daMUSIC import daMUSIC_val, DeepAugmentMusic
from models.daMUSIC_source_num_estimator import DaMUSICSourceNumEstimator, daMUSICSourceNumEstimator_val
from models.deepSFNS import DeepSFNS, deepSFNS_val
from models.deepSFNS_source_num_estimator import DeepSFNSSourceNumEstimator, deepSFNSSourceNumEstimator_val
from models.deepSSE import DeepSSE, deepSSE_val
from models.lowSNRNet import DOALowSNRNet, lowSNRNet_val
from models.music import music_val
from models.source_num_estimator import SourceNumEstimator, sourceNumEstimator_val
from models.subspaceNet import SubspaceNet, subspaceNet_val
from utils.dataset import DOADataset, RealMANDataset
from utils.metrics import RMSPE, Accuracy, RMSPE_padding_random, RMSPE_padding_zeros

import csv
from datetime import datetime

from utils.util import create_worker_seed_val




# 加载train.yaml
def val_main(gen_data_config, train_config,model_config,validate_config,folder_path,RealMAN_test_list=None):
    print("Run validating...")
    # rng_restored = np.random.default_rng()  # 新生成器
    # rng = load_rng_state(folder_path / 'rng_state.pkl', rng_restored)


    if gen_data_config['ar_tp'] == 'ULA' or gen_data_config['ar_tp'] == 'RealMAN':
        mod = np.pi
    else:
        mod = 2 * np.pi

    arrangement = get_arrangement(gen_data_config)

    data_num = gen_data_config['da_n']
    train_datanum = int(data_num*0.9)

    is_cov = False
    is_Rx_tou = False
    is_eigenvalue = False
    if train_config['model'] == 'deepSSE' or train_config['model'] == 'lowSNRNet':
        is_cov = True
    if train_config['model'] == 'subspaceNet':
        is_Rx_tou = True
    if train_config['model'] == 'SourceNumEstimator':
        is_eigenvalue = True

    val_dataset = DOADataset(arrangement = arrangement, num_samples = data_num - train_datanum, gen_data_config=gen_data_config, is_cov=is_cov, is_Rx_tou=is_Rx_tou,is_eigenvalue=is_eigenvalue)
    if RealMAN_test_list is not None:
        val_dataset = RealMANDataset(data_list = RealMAN_test_list,arrangement=arrangement, gen_data_config=gen_data_config,
                                   is_cov=is_cov, is_Rx_tou=is_Rx_tou)

    batch_size = train_config['bs']
    test_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=8,worker_init_fn=create_worker_seed_val(5201314))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取是什么模型
    if train_config['model'] == 'deepSFNS':
        model = DeepSFNS(gen_data_config, model_config, train_config).to(device)
        # #加载预训练权重
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        # 得到角度和标签
        doas_list,labels_list,doas_known_d_list = deepSFNS_val(model, test_loader, device,validate_config, gen_data_config)
    elif train_config['model'] == 'deepSSE':
        deepsse_config = {
            "steering_vetor": val_dataset.steeringMatrix_A,
            "num_class": len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']))),
            "num_antenna": gen_data_config['m'],
            "antenna_spacing": (gen_data_config['c'] / gen_data_config['f']) / 2,
            "img_channels": 3,
            "in_channels": 32,
            "out_channels": 32,
            "layers": [2, 2],
            "activation": "gelu",
            "d_model": 128,
            "nhead": 8,
            "num_ca_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.05
        }
        model = DeepSSE(**deepsse_config).to(device)
        # #加载预训练权重
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        # 得到角度和标签
        doas_list,labels_list,doas_known_d_list = deepSSE_val(model, test_loader, device,validate_config, gen_data_config)
    elif train_config['model'] == 'MUSIC':
        doas_list,labels_list = music_val(test_loader, arrangement, gen_data_config)
        doas_known_d_list = doas_list
    elif train_config['model'] == 'B_MUSIC':
        doas_list,labels_list = B_music_val(test_loader, arrangement, gen_data_config)
        doas_known_d_list = doas_list
    elif train_config['model'] == 'daMUSIC':
        model = DeepAugmentMusic(gen_data_config['m'], gen_data_config['c'] / (2 * gen_data_config['f']), gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        doas_list, labels_list, doas_known_d_list = daMUSIC_val(model, test_loader, device,gen_data_config)
    elif train_config['model'] == 'subspaceNet':
        model = SubspaceNet(tau = 8).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        doas_list, labels_list, doas_known_d_list = subspaceNet_val(model, test_loader, device,gen_data_config)
    elif train_config['model'] == 'lowSNRNet':
        model = DOALowSNRNet(len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p'])))).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #
        # print(f"总参数量: {total_params:,}")
        # print(f"可训练参数量: {trainable_params:,}")
        doas_list, labels_list, doas_known_d_list = lowSNRNet_val(model, test_loader, device, validate_config, gen_data_config)
    elif train_config['model'] == 'IQResNet':
        model = IQResNet(len(list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p']))), gen_data_config['m']).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        doas_list, labels_list, doas_known_d_list = IQResNet_val(model, test_loader, device, validate_config, gen_data_config)
    elif train_config['model'] == 'SourceNumEstimator':
        model = SourceNumEstimator(gen_data_config['m']).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        accuracy = sourceNumEstimator_val(model, test_loader, device,validate_config, gen_data_config)
    elif train_config['model'] == 'DaMUSICSourceNumEstimator':
        model = DaMUSICSourceNumEstimator(gen_data_config,device).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        accuracy = daMUSICSourceNumEstimator_val(model, test_loader, device,validate_config, gen_data_config)
    elif train_config['model'] == 'DeepSFNSSourceNumEstimator':
        model = DeepSFNSSourceNumEstimator(gen_data_config, model_config, train_config,device).to(device)
        model.load_state_dict(torch.load(folder_path / 'best_model.pth', map_location=device))
        accuracy = deepSFNSSourceNumEstimator_val(model, test_loader, device,validate_config, gen_data_config)

    else:
        return -1

    if train_config['model'] == 'SourceNumEstimator' or train_config['model'] == 'DaMUSICSourceNumEstimator' or train_config['model'] == 'DeepSFNSSourceNumEstimator':
        # 将结果保存为csv或txt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp, accuracy]

        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile(folder_path / 'results.csv')

        with open(folder_path / 'results.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 如果文件不存在，写入表头
            if not file_exists:
                header = ['Timestamp', 'Accuracy']
                writer.writerow(header)

            # 写入数据
            writer.writerow(row_data)

        print("results are saved to results.csv")
    else:
        # 两个指标验证
        rmspe = RMSPE(doas_list, labels_list, mod)
        accuracy, false_alarm_rate, false_alarm_mean_count, Missed_alarm_rate, Missed_alarm_mean_count = Accuracy(
            doas_list, labels_list)
        rmspe_padding_random = RMSPE_padding_random(doas_list, labels_list, gen_data_config, mod)
        rmspe_padding_random_known = RMSPE_padding_random(doas_known_d_list, labels_list, gen_data_config, mod)
        rmspe_padding_zeros = RMSPE_padding_zeros(doas_list, labels_list, mod)

        # 将结果保存为csv或txt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp, rmspe, rmspe_padding_random_known, rmspe_padding_zeros, rmspe_padding_random, accuracy,
                    false_alarm_rate, false_alarm_mean_count, Missed_alarm_rate,
                    Missed_alarm_mean_count]

        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile(folder_path / 'results.csv')

        with open(folder_path / 'results.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 如果文件不存在，写入表头
            if not file_exists:
                header = ['Timestamp', 'RMSPE', 'RMSPE_padding_random_known', 'RMSPE_padding_zeros',
                          'RMSPE_padding_random', 'Accuracy', 'False_Alarm_Rate',
                          'False_Alarm_Mean_Count', 'Missed_Alarm_Rate', 'Missed_Alarm_Mean_Count']
                writer.writerow(header)

            # 写入数据
            writer.writerow(row_data)

        print("results are saved to results.csv")

    return 0