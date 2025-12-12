import argparse
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from utils.util import get_steer_vector, awgn

#半波长ULA
def ULA_half_wave_length_array(m, c, f):
    d = c / f / 2
    y_positions = (np.arange(m) - (m - 1) / 2) * d  # 对称排列，中心在原点
    x = np.zeros(m)
    return x , y_positions


#ULA
def ULA_equal_aperture(m, aperture):
    y_positions = np.linspace(-aperture / 2, aperture / 2, m)
    x = np.zeros(m)
    return x, y_positions


#RealMAN阵列
def RealMAN_x_array():
    y_positions = np.linspace(-0.12, 0.15, 10)
    x = np.zeros(10)
    return x, y_positions

def get_arrangement(gen_data_config):
    #根据阵列类型 m 阵列孔径 生成阵列坐标
    array_type = gen_data_config['ar_tp']
    m = gen_data_config['m']
    c = gen_data_config['c']
    f = gen_data_config['f']
    aperture = gen_data_config['apte']

    if array_type == 'ULA':
        if aperture == None:
            x,y = ULA_half_wave_length_array(m, c, f)
        else:
            x,y = ULA_equal_aperture(m, aperture)
    elif array_type == 'RealMAN':
        x, y = RealMAN_x_array()
    else:
        return -1

    arrangement = np.vstack((x, y))
    return arrangement

def gen_data(arrangement, gen_data_config,rng):
    if gen_data_config['p_err'] != None:
        dd = np.sqrt((arrangement[0, 1] - arrangement[0, 0])**2 + (arrangement[1, 1] - arrangement[1, 0])**2)
        sigma = gen_data_config['p_err'] * dd
        error = rng.uniform(-sigma, sigma, arrangement.shape)
        arrangement += error

    #根据读入的下限和上限参数 生成待选角度列表
    degree_lower = gen_data_config['deg_l']
    degree_upper = gen_data_config['deg_u']
    degree_precision = gen_data_config['deg_p']

    degree_list = list(np.arange(degree_lower, degree_upper, degree_precision))

    # degree_list_copy = list(np.arange(-60, 60, 1))

    #从待选角度列表中均匀随机选出d个目标角度
    d_list = gen_data_config['d']
    d = rng.choice(d_list, size=1, replace=False)[0]
    if gen_data_config['deg_g'] == None:
        degrees_label = np.array(rng.choice(degree_list, size = d, replace = False))
    else:
        degrees_label = np.zeros(d)
        degree_first = rng.choice(degree_list, size=1, replace=False)[0]
        degrees_label[0] = degree_first
        for i in range(d-1):
            degrees_label[i+1] = (degree_first+gen_data_config['deg_g']*(i+1) + degree_upper) % (degree_upper-degree_lower) - degree_upper
    thetas = degrees_label * np.pi / 180

    #计算信号接收的阵列流行矩阵Atorch.manual_seed(seed)
    c = gen_data_config['c']
    f = gen_data_config['f']
    A = np.array([get_steer_vector(arrangement, thetas[j], 0, f, c) for j in range(d)])

    # 根据d snapshots 生成信号
    #计算m个阵元接收的信号
    #根据snr加上噪声
    snapshots = gen_data_config['spshots']
    snr_list = gen_data_config['snr']
    snr = rng.choice(snr_list, size=1, replace=False)[0]
    if gen_data_config['is_co']:
        signal_orign = (rng.standard_normal((1, snapshots)) + 1j * rng.standard_normal((1, snapshots)))
        signal_orign = np.tile(signal_orign, (d, 1))
    else:
        signal_orign = (rng.standard_normal((d, snapshots)) + 1j * rng.standard_normal((d, snapshots)))

    # signal_orign = np.tile(signal_orign, (1, 200))
    # signal_orign = signal_orign[:,:200]

    signal = np.dot(A.T, signal_orign)
    signal, noise = awgn(signal, snr_db=snr, method="measured", rng = rng)


    #获得标签
    binary_labels = np.zeros(len(degree_list))
    for idx in degrees_label:
        binary_labels[int((idx-degree_lower)/degree_precision)] = 1.0

    return signal, binary_labels, noise




def gen_data_main(gen_data_config,folder_path):

    data_num = gen_data_config['da_n']
    arrangement = get_arrangement(gen_data_config)

    signal_list = []
    label_list = []
    for i in tqdm(range(data_num)):
        singals, labels = gen_data(arrangement, gen_data_config)
        signal_list.append(singals)
        label_list.append(labels)

    signals_data = np.array(signal_list)
    labels_data = np.array(label_list)

    gen_datas_dict = {
        'signals_data': signals_data,
        'labels_data': labels_data,
        'arrangement': arrangement,
        'gen_data_config': gen_data_config
    }

    # filename = ""
    # for key, value in gen_data_config.items():
    #     # 添加键值对到文件名
    #     filename += f"{key}_{value}_"
    # filename = filename.rstrip("_")


    #保存到npz
    np.savez_compressed(folder_path / f"data.npz", **gen_datas_dict)

    # # 加载数据
    # loaded = np.load("data.npz", allow_pickle=True)
    # signals_data = loaded['signals_data']
    # labels_data = loaded['labels_data']
    # arrangement = loaded['arrangement']
    # gen_data_config = loaded['gen_data_config'].item()
    # loaded.close()
    print()



    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gen data Script')
    parser.add_argument('--yaml_config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    yaml_config_path = Path(args.yaml_config_path)

    with open(yaml_config_path, 'r') as file:
        gen_data_config = yaml.safe_load(file)

    gen_data_main(gen_data_config,"./")