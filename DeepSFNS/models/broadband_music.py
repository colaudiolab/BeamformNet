import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import linalg,signal
from tqdm import tqdm

from utils.util import get_steer_vector


def classicMUSIC(incident, arrangment, thetas, f, c, sources=None, phi = 0):
    # # calculate EVD of covariance matrix
    # covariance = np.cov(incident)
    covariance = incident @ incident.conj().T

    eigenvalues, eigenvectors = linalg.eig(covariance)

    d = sources

    #排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # 对 eigenvalues 进行排序

    sorted_eigenvalues = eigenvalues[sorted_indices]

    # 对 eigenvectors 进行排序

    sorted_eigenvectors = eigenvectors[:,sorted_indices]


    # the noise matrix
    En = sorted_eigenvectors[:, d:]

    # calculate spatial spectrum
    numSamples = thetas.shape[1]
    spectrum = np.zeros(numSamples)
    for axis in thetas:
        for i in range(numSamples):
            # establish array steering vector
            a = get_steer_vector(arrangment, axis[i], phi, f, c)
            spectrum[i] = 1./(a.conj().transpose() @ En @ En.conj().transpose() @ a)

    DoA, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoA = DoA[np.argsort(spectrum[DoA])[-d:]]

    return DoA, spectrum

def B_music_val(val_loader, arrangement, gen_data_config):
    doas_list = []
    label_list = []
    for batch in tqdm(val_loader):
        signal = batch['signal'].numpy()
        labels = batch['label'].numpy()

        for index in range(batch['label'].shape[0]):
            # 扫描角度
            angles = np.array((np.linspace(- np.pi / 2, np.pi / 2, 180, endpoint=False),))  # angle continuum
            sample_rate = 16000
            fmin = 100
            fmax = 1000
            d = len(np.where(labels[index] == 1)[0])

            waveform_clip = signal[index]

            # 对信号进行FFT
            fft_values = np.fft.fft(waveform_clip, axis=1)
            fft_freqs = np.fft.fftfreq(waveform_clip.shape[1], 1 / sample_rate)

            # 只取前半部分频率
            half_n = len(fft_freqs) // 2

            spectrum_freq_list = []
            # 对音频做ISM宽带MUSIC
            for i in range(half_n):
                f = fft_freqs[i]
                if (f >= fmin and f <= fmax):  # or (f >= fmin1 and f <= fmax1):
                    DoAMUSIC, spectrum = classicMUSIC(fft_values[:, i: i + 1], arrangement, angles, f, gen_data_config['c'], sources=d,
                                                      phi=0)

                    max_ = spectrum.max()
                    min_ = spectrum.min()
                    spectrum = (spectrum - min_) / (max_ - min_)
                    spectrum_freq_list.append(spectrum)

            spectrum_freq_list = np.array(spectrum_freq_list)
            spectrum_freq_list = np.mean(spectrum_freq_list, axis=0)

            # plt.figure(figsize=(12, 6))
            # plt.plot(spectrum_freq_list, label='result_deepSFNS_energy', color='blue')
            # thetas = np.where(labels[index] == 1)[0]
            # plt.scatter(thetas, np.ones(thetas.shape[0]) * max(spectrum_freq_list), color='green')
            # plt.title('deepSFNS_energy')
            # plt.xlabel('degree')
            # plt.ylabel('energy')
            # plt.grid(True)
            # plt.show()

            doas_peak = scipy.signal.find_peaks(spectrum_freq_list)[0]
            doas_peak = doas_peak[np.argsort(spectrum_freq_list[doas_peak])][::-1]
            doas_great = np.argsort(spectrum_freq_list)[::-1]
            doas_known_d = np.concatenate([doas_peak, doas_great[~np.isin(doas_great, doas_peak)]])[:d]

            doas_list.append((doas_known_d * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
            label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)

    return doas_list, label_list

