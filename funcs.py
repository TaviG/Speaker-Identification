import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


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
