import matplotlib
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import scipy.signal as ss
import librosa.display


def plot_data_distribution(array, title):
    # Convert from ['id1234', 'id1235', 'id1321', ...] to [0, 1, 87, ..]
    array = [int(elem[2:]) for elem in array]
    array = [elem - min(array) for elem in array]

    unique_elems = set(array)
    counts = dict.fromkeys(unique_elems, 0)
    for elem in unique_elems:
        for arr in array:
            if arr == elem:
                counts[elem] += 1
    
    lists = sorted(counts.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.figure()
    plt.bar(x, y)
    plt.xlabel('Person id')
    plt.ylabel('Number of recordings')
    plt.title('Data distribution')
    plt.savefig(title)


def read_wav_files(src):
    data = []
    for x in src:
        _, info = wavfile.read(x)
        info = np.float32(info / (2 ** 15))
        data.append(info)

    return data


def calc_mean_variance(data):
    means = []
    variance = []
    for info in data:
        means.append(np.mean(info))
        variance.append(np.var(info))

    return means, variance


def plot_mean(means, title):
    means.sort()
    plt.figure()
    plt.plot(means)
    plt.xlabel('Audio recording')
    plt.ylabel('Mean value')
    plt.title("Mean value of all audio recordings")
    plt.savefig(title)


def plot_variance(var, title):
    var.sort()
    plt.figure()
    plt.plot(var)
    plt.xlabel('Audio recording')
    plt.ylabel('Variance')
    plt.title("Dataset variance")
    plt.savefig(title)


def calc_fft(audio_fft, i, j):
    for length in range(i,j):
        audio_fft[length] = np.fft.fft(audio_fft[length])
    return audio_fft


def plot_mag_phase(data, fft):
    for i in range(10):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(data[i])
        plt.subplot(3,1,2)
        plt.plot(np.abs(fft[i]))
        plt.subplot(3,1,3)
        plt.plot(np.angle(fft[0]))
        plt.savefig('fft_analysis'+str(i)+'.jpg')

def power_spectral_density(data, fs):
    periodgrams = []
    for signal in data:
        f, S = ss.periodogram(signal, fs, scaling='density')
        periodgrams.append((f, S))

    return periodgrams

def plot_psd(psd, idx):
    plt.figure()
    plt.semilogy(psd[0], psd[1])
    plt.title("Power spectral density")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.savefig(f'psd{idx}.jpg')

def plot_acorr(acorr):
    plt.figure()
    plt.plot(acorr)
    plt.title("Real part of the inverse Fourier transform of the power spectrum")
    plt.savefig('acorr2.jpg')

def plot_acorrs(acorr1, acorr2):
    plt.figure()
    plt.plot(acorr1, 'b', label="Normal")
    plt.plot(acorr2, 'r', label="Using inverse FFT")
    plt.title("Autocorrelation")
    plt.legend()
    plt.savefig('acorrs.jpg')

def plot_mfcc(mfcc):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis="time", sr=16000)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig('mfcc.jpg')

def min_length(arr):
    min_length = 100000
    for elem in arr:
        if len(elem) < min_length:
            min_length = len(elem)

    return min_length

def crop_arrs(arr, length):
    for elem in arr:
        elem = elem[:length]

    return arr

def plot_scatter(x, y, labels, n_labels, title):
    fig, ax = plt.subplots(1,1, figsize=(12,12))

    # Define colormap
    cmap = plt.cm.jet
    bounds = np.linspace(0, n_labels, n_labels+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax.scatter(x, y, c=labels, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label("Person id")
    ax.set_title('Audio 2D mapping')
    plt.savefig(title)