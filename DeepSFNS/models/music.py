import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import linalg
from tqdm import tqdm

from utils.util import get_steer_vector


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

    doas_peak = scipy.signal.find_peaks(spectrum)[0]
    doas_peak = doas_peak[np.argsort(spectrum[doas_peak])][::-1]
    doas_great = np.argsort(spectrum)[::-1]
    doas_known_d = np.concatenate([doas_peak, doas_great[~np.isin(doas_great, doas_peak)]])[:d]


    # plt.figure(figsize=(12, 6))
    # plt.plot(spectrum, label='result_MUSIC_energy', color='blue')
    # # plt.scatter((thetas / np.pi) * 180 + 180, np.ones(thetas.shape[0]) * max(spectrum), color='green')
    # plt.title('MUSIC_energy')
    # plt.xlabel('degree')
    # plt.ylabel('energy')
    # plt.grid(True)
    # plt.show()

    return doas_known_d


def music_val(val_loader, arrangement, gen_data_config):
    doas_list = []
    label_list = []
    for batch in tqdm(val_loader):
        signal = batch['signal'].numpy()
        labels = batch['label'].numpy()

        for index in range(batch['label'].shape[0]):
            d = len(np.where(labels[index] == 1)[0])
            doas = classicMUSIC(signal[index],arrangement,list(np.arange(gen_data_config['deg_l'], gen_data_config['deg_u'], gen_data_config['deg_p'])*np.pi/180),gen_data_config['f'],gen_data_config['c'],d,0)
            doas_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
            label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)

    return doas_list, label_list

