import lap
import numpy as np
from tqdm import tqdm

np.random.seed(seed=12)

#两个评价指标

def RMSE_mode(theta_1, theta_2, mod = np.pi):
    rmse = (((theta_1 - theta_2) + mod / 2) % mod - mod / 2)**2
    return rmse

def RMSPE(doas_list, labels_list,mod = np.pi):
    #使用Jonker-Volgenant algorithm 算法匹配
    num = len(doas_list)
    rmse_list = []
    print('calculating RMSPE...')
    for num_index in tqdm(range(num)):
        doas = np.array(doas_list[num_index])
        labels = np.array(labels_list[num_index])

        m = len(doas)
        n = len(labels)

        min_m_n = min(m,n)

        if min_m_n == 0:
            continue

        costs_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                costs_matrix[i, j] = RMSE_mode(doas[i], labels[j],mod)

        cost = lap.lapjv(costs_matrix, extend_cost=True)[0]

        rmse = np.sqrt(cost / min_m_n)
        rmse_list.append(rmse)

    return np.array(rmse_list).mean()

def RMSPE_padding_random(doas_list, labels_list,gen_data_config,mod = np.pi):
    #使用Jonker-Volgenant algorithm 算法匹配
    num = len(doas_list)
    rmse_list = []
    print('calculating RMSPE...')
    for num_index in tqdm(range(num)):
        doas = np.array(doas_list[num_index])
        labels = np.array(labels_list[num_index])

        m = len(doas)
        n = len(labels)

        # 确保长度一致
        max_m_n = max(m, n)
        if m < max_m_n:
            doas = np.concatenate([doas, np.random.uniform(gen_data_config['deg_l'] * np.pi / 180, gen_data_config['deg_u'] * np.pi / 180, max_m_n - m)])
        if n < max_m_n:
            labels = np.concatenate([labels, np.random.uniform(gen_data_config['deg_l'] * np.pi / 180, gen_data_config['deg_u'] * np.pi / 180, max_m_n - n)])


        costs_matrix = np.zeros((max_m_n, max_m_n))
        for i in range(max_m_n):
            for j in range(max_m_n):
                costs_matrix[i, j] = RMSE_mode(doas[i], labels[j],mod)

        cost = lap.lapjv(costs_matrix)[0]

        rmse = np.sqrt(cost / max_m_n)
        rmse_list.append(rmse)

    return np.array(rmse_list).mean()



def RMSPE_padding_zeros(doas_list, labels_list, mod = np.pi):
    #使用Jonker-Volgenant algorithm 算法匹配
    num = len(doas_list)
    rmse_list = []
    print('calculating RMSPE...')
    for num_index in tqdm(range(num)):
        doas = np.array(doas_list[num_index])
        labels = np.array(labels_list[num_index])

        m = len(doas)
        n = len(labels)

        # 确保长度一致
        max_m_n = max(m, n)
        if m < max_m_n:
            doas = np.concatenate([doas, np.zeros(max_m_n - m)])
        if n < max_m_n:
            labels = np.concatenate([labels, np.zeros(max_m_n - n)])


        costs_matrix = np.zeros((max_m_n, max_m_n))
        for i in range(max_m_n):
            for j in range(max_m_n):
                costs_matrix[i, j] = RMSE_mode(doas[i], labels[j], mod)

        cost = lap.lapjv(costs_matrix)[0]

        rmse = np.sqrt(cost / max_m_n)
        rmse_list.append(rmse)

    return np.array(rmse_list).mean()


def Accuracy(doas_list, labels_list):
    num = len(doas_list)
    print('calculating accuracy...')
    equal_num = 0
    more_num = 0
    more_count = 0
    less_num = 0
    less_cont = 0
    for num_index in tqdm(range(num)):
        doas = np.array(doas_list[num_index])
        labels = np.array(labels_list[num_index])

        doas_len = len(doas)
        labels_len = len(labels)

        if doas_len == labels_len:
            equal_num += 1
        elif doas_len < labels_len:
            less_num += 1
            less_cont += labels_len - doas_len
        else:
            more_num += 1
            more_count += doas_len - labels_len

    accuracy = equal_num / num if num > 0 else 0.0
    false_alarm_rate = more_num / num if num > 0 else 0.0
    false_alarm_mean_count = more_count / more_num if more_num > 0 else 0.0
    Missed_alarm_rate = less_num / num if num > 0 else 0.0
    Missed_alarm_mean_count = less_cont / less_num if less_num > 0 else 0.0

    return accuracy,false_alarm_rate,false_alarm_mean_count,Missed_alarm_rate,Missed_alarm_mean_count

if __name__ == '__main__':
    cost_matrix = np.array([
        [1, 2, 5],
        [3, 1, 4],
    ], dtype=np.float64)
    a = lap.lapjv(cost_matrix, extend_cost=True)
    print(a)