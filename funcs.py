from scipy.io import wavfile
import numpy as np
import scipy.signal as ss
import os
import random
import threading

Fs = 16000

def extract_ds(inputdir):
    audios = []
    people = []

    for folder in os.listdir(inputdir):
        for video in os.listdir(os.path.join(inputdir, folder)):
            for audio in os.listdir(os.path.join(inputdir, folder, video)):
                if(audio.endswith(".wav")):
                    audios.append(os.path.join(inputdir, folder, video, audio))
                    people.append(folder)

    return audios, people

def shuffle_equally(x, y):
    both = list(zip(x, y))
    random.seed(42)
    random.shuffle(both)
    # Get back arrays
    x, y = zip(*both)
    return x, y

def read_wav_file(src):
    _, info = wavfile.read(src)
    info = np.float32(info / (2 ** 15))
    return info

def calc_mean_variance(data):
    means = []
    variance = []
    for info in data:
        means.append(np.mean(info))
        variance.append(np.var(info))

    return means, variance

def calc_cov_mat(data, start_samples, end_samples):
    audio_cov = [elem[start_samples:end_samples] for elem in data]
    audio_cov = np.stack(audio_cov, axis=0)
    cov = np.matmul(audio_cov, audio_cov.T)

    return cov

def calc_fft(audio_fft, freqs, i, j):
    for length in range(i,j):
        audio_fft[length] = np.fft.fft(audio_fft[length])
        freqs[length] = np.fft.fftfreq(audio_fft[length].size, d=1/Fs)
    return audio_fft, freqs

def fft_of_ds(data, num_threads):
    audio_fft = data.copy()
    freqs = data.copy()

    threads = []
    for nr in range(num_threads):
        t = threading.Thread(target=calc_fft, args=(audio_fft, freqs, int(nr*len(audio_fft)/num_threads), int((nr+1)*len(audio_fft)/num_threads),))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return audio_fft, freqs

def power_spectral_density(data, fs):
    periodgrams = []
    for signal in data:
        f, S = ss.periodogram(signal, fs, scaling='density')
        periodgrams.append((f, S))

    return periodgrams

def calc_accor(signal, mean, var):
    ndata = np.array(signal) - np.array(mean)
    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:]
    acorr = acorr / var / len(ndata)

    return acorr

def wiener_hincin(signal, mean, var):
    ndata = np.array(signal) - np.array(mean)

    size = 2 ** np.ceil(np.log2(2*len(signal) - 1)).astype('int')
    fft = np.fft.fft(ndata, size)

    pwr = np.abs(fft) ** 2

    acorr = np.fft.ifft(pwr).real / var / len(ndata)
    acorr = acorr[:len(ndata)]

    return acorr

def min_length(arr):
    min_length = 100000
    for elem in arr:
        if len(elem) < min_length:
            min_length = len(elem)

    return min_length

def Kmeans_accuracy(official_labels, pred_labels):
    accuracies = {}
    for label in np.unique(official_labels):
        idxes = official_labels == label
        predicted_classes = pred_labels[idxes]

        counts = {}
        for pred_class in np.unique(predicted_classes):
            counts[pred_class] = np.count_nonzero(predicted_classes == pred_class)

        accuracy = max(counts.values()) / sum(counts.values())

        accuracies[label] = accuracy

    return accuracies
