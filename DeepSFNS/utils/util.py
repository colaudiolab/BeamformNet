import random

import numpy as np
import pickle
from scipy import linalg
import torch



def get_steer_vector(arrangement, theta, phi, f, c):
    # arrangement 一个二维numpy数组 （2, M） M阵元数 2 第一个是横坐标 第二个是纵坐标 单位m
    # theta 方位角 弧度 标量
    # phi 仰角 弧度 标量
    # f 中心频率
    # c 声速
    arrangement_copy = np.zeros((arrangement.shape[0], arrangement.shape[1]))
    arrangement_copy[0, :] = arrangement[0, :] * np.cos(theta) * np.cos(phi)
    arrangement_copy[1, :] = arrangement[1, :] * np.sin(theta) * np.cos(phi)
    steer_vector = np.ones((1, 2)) @ arrangement_copy
    steer_vector = np.exp(- 1j * 2 * np.pi * f * steer_vector * (1 / c))

    return steer_vector[0]


def awgn(X, snr_db, method='measured', rng=None):
    signal_power = np.mean(np.abs(X) ** 2)  # 计算信号功率
    snr_linear = 10 ** (snr_db / 10)  # 将信噪比从分贝转换为线性比例

    # 如果使用'measured'方法，则使用信号的实际功率来计算噪声功率
    if method == 'measured':
        noise_power = signal_power / snr_linear
    else:
        noise_power = 1 / snr_linear  # 如果不是'measured'，则假定信号功率为1

    # 生成噪声，对于复数信号，实部和虚部都需要生成噪声
    noise_real = np.sqrt(noise_power / 2) * rng.normal(0, 1, X.shape)
    noise_imag = np.sqrt(noise_power / 2) * rng.normal(0, 1, X.shape)
    noise = noise_real + 1j * noise_imag

    Y = X + noise  # 将噪声添加到信号中

    return Y, noise



def cal_cov(data, use_phase=True, inference=False):
    """Calculate covariance matrix of data"""
    if data.shape[1] == 1:
        cov = data @ data.conj().T
    else:
        cov = np.cov(data)



    # normalize
    cov = cov / np.linalg.norm(cov)

    if use_phase:
        cov = np.concatenate(
            (
                np.real(cov)[np.newaxis],
                np.imag(cov)[np.newaxis],
                np.angle(cov)[np.newaxis],
            ),
            axis=0,
        ).astype(np.float32)
    else:
        cov = np.concatenate(
            (np.real(cov)[np.newaxis], np.imag(cov)[np.newaxis]), axis=0
        ).astype(np.float32)

    # if inference:
    #     cov = prepare_for_inf(cov)

    return cov


def save_rng_state(filename, rng):
    """保存随机数生成器状态到文件"""
    state = rng.bit_generator.state
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_rng_state(filename, rng):
    """从文件加载随机数生成器状态"""
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    rng.bit_generator.state = state
    return rng


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]), 1))
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X))
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


def cal_Rx_tou(X,tau = 8):
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr.float()

def create_worker_seed_train(seed):
    def worker_seed_train(worker_id):
        # 为每个worker设置不同的随机种子
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset._init_worker_rng(seed+worker_id)


    return worker_seed_train

def create_worker_seed_val(seed):
    def worker_seed_val(worker_id):
        # 为每个worker设置不同的随机种子
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset._init_worker_rng(seed+worker_id)

    return worker_seed_val

def cal_eigenvalue(data):
    if data.shape[1] == 1:
        cov = data @ data.conj().T
    else:
        cov = np.cov(data)

    eigenvalues, _ = linalg.eig(cov)
    #排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    return np.abs(sorted_eigenvalues).astype(np.float32)

