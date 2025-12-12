import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.gen_data import gen_data
from utils.util import get_steer_vector, cal_cov, cal_Rx_tou, cal_eigenvalue


class DOADataset(Dataset):
    def __init__(self, arrangement, num_samples, gen_data_config,is_cov=False,is_Rx_tou=False,is_eigenvalue=False):
        self.arrangement = arrangement
        self.gen_data_config = gen_data_config
        self.num_samples = num_samples
        self.is_cov = is_cov
        self.is_Rx_tou = is_Rx_tou
        self.is_eigenvalue = is_eigenvalue

        degree_lower = gen_data_config['deg_l']
        degree_upper = gen_data_config['deg_u']
        degree_precision = gen_data_config['deg_p']
        degree_list = list(np.arange(degree_lower, degree_upper, degree_precision))
        c = gen_data_config['c']
        f = gen_data_config['f']

        steeringMatrix_A = np.zeros((arrangement.shape[1], len(degree_list)), dtype=complex)
        for i in range(len(degree_list)):
            theta = degree_list[i] * np.pi / 180
            steeringMatrix_A[:, i] = get_steer_vector(arrangement, theta, 0, f, c)
        self.steeringMatrix_A = steeringMatrix_A

        self._rng = None

    def _init_worker_rng(self, seed):
        """为每个worker初始化独立的RNG"""
        # 基于基础种子和worker_id创建唯一种子
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):
        signal, binary_label, noise = gen_data(np.copy(self.arrangement) , self.gen_data_config,self._rng)
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker_info.id, signal[0][0])

        cov = 0
        if self.is_cov:
            cov = torch.from_numpy(cal_cov(signal))

        Rx_tou = 0
        if self.is_Rx_tou:
            Rx_tou = cal_Rx_tou(torch.from_numpy(signal))

        eigenvalue = 0
        if self.is_eigenvalue:
            eigenvalue = torch.from_numpy(cal_eigenvalue(signal))

        return {
            'Rx_tou': Rx_tou,
            'cov': cov,
            'eigenvalue': eigenvalue,
            'signal': signal,
            'label': binary_label,
            'steeringMatrix_A':self.steeringMatrix_A,
            'noise': noise
        }


class RealMANDataset(Dataset):
    def __init__(self,data_list,arrangement,gen_data_config,is_cov=False,is_Rx_tou=False):
        self.data_list = data_list
        self.gen_data_config = gen_data_config
        self.is_cov = is_cov
        self.is_Rx_tou = is_Rx_tou

        degree_lower = gen_data_config['deg_l']
        degree_upper = gen_data_config['deg_u']
        degree_precision = gen_data_config['deg_p']
        degree_list = list(np.arange(degree_lower, degree_upper, degree_precision))
        c = gen_data_config['c']
        f = gen_data_config['f']

        steeringMatrix_A = np.zeros((arrangement.shape[1], len(degree_list)), dtype=complex)
        for i in range(len(degree_list)):
            theta = degree_list[i] * np.pi / 180
            steeringMatrix_A[:, i] = get_steer_vector(arrangement, theta, 0, f, c)
        self.steeringMatrix_A = steeringMatrix_A

    def _init_worker_rng(self, seed):
        """为每个worker初始化独立的RNG"""
        # 基于基础种子和worker_id创建唯一种子
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item_path = self.data_list[index]
        #获得标签
        data_item_path = Path(data_item_path)
        filename = data_item_path.stem

        pattern = r'^.*_(\d+)_(-?\d+\.\d+)$'
        match = re.match(pattern, filename)
        label_angle = float(match.group(2))

        degree_lower = self.gen_data_config['deg_l']
        degree_upper = self.gen_data_config['deg_u']
        degree_precision = self.gen_data_config['deg_p']

        degree_list = list(np.arange(degree_lower, degree_upper, degree_precision))
        binary_label = np.zeros(len(degree_list))
        binary_label[round((label_angle - degree_lower) / degree_precision) % int((degree_upper - degree_lower) / degree_precision)] = 1.0

        #加载数据
        signal = np.load(data_item_path)

        cov = 0
        if self.is_cov:
            cov = torch.from_numpy(cal_cov(signal))

        Rx_tou = 0
        if self.is_Rx_tou:
            Rx_tou = cal_Rx_tou(torch.from_numpy(signal))

        return {
            'Rx_tou': Rx_tou,
            'cov': cov,
            'signal': signal,
            'label': binary_label,
            'steeringMatrix_A':self.steeringMatrix_A
        }

