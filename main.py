# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:10:17 2022
@author: Tavi
"""

import os
import sys
import random
import funcs
import threading
from IPython import get_ipython
import gc
import numpy as np

inputdir = r"D:\CPPSMS\dataset\test\\"

people = []
audios = []
num_threads = 4
threads = []

for folder in os.listdir(inputdir):
    for video in os.listdir(os.path.join(inputdir, folder)):
        for audio in os.listdir(os.path.join(inputdir, folder, video)):
            if(audio.endswith(".wav")):
                audios.append(os.path.join(inputdir, folder, video, audio))
                people.append(folder)

# Shuffle both arrays in the same way
both = list(zip(audios, people))
random.seed(42)
random.shuffle(both)
# Get back arrays
audios, people = zip(*both)

n_audios = len(audios)

# Split into 80% train 20% test
n_train = round(n_audios * 0.8)
    
X_train = audios[:n_train]
Y_train = people[:n_train]

X_test = audios[n_train:]
Y_test = people[n_train:]

funcs.plot_data_distribution(Y_train, 'train_distribution.jpg')
funcs.plot_data_distribution(Y_test, 'test_distribution.jpg')
funcs.plot_data_distribution(people, 'people_distribution.jpg')

#Audio file read
audio_data = funcs.read_wav_files(audios)

# #Statistical Moments
# #Mean + Variance
mean, variance, covariance = funcs.calc_mean_variance(audio_data)

# Plot means and variances
funcs.plot_mean(mean, "mean.jpg")        
funcs.plot_variance(variance, "var.jpg")

# Power spectral density
psds = funcs.power_spectral_density(audio_data[:10], 16000)
for idx, psd in enumerate(psds):
    funcs.plot_psd(psd, idx)

# Autocorrelation of one signal
ndata = np.array(audio_data[0]) - np.array(mean[0])
acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:]
acorr = acorr / (variance[0] + 0.000000001) / len(ndata)
funcs.plot_acorr(acorr)

# Wiener Khinchin test
size = 2 ** np.ceil(np.log2(2*len(audio_data[0]) - 1)).astype('int')
fft = np.fft.fft(ndata, size)
pwr = np.abs(fft) ** 2
acorr2 = np.fft.ifft(pwr).real / (variance[0] + 0.000000001) / len(ndata)
acorr2 = acorr2[:len(ndata)]
funcs.plot_acorr(acorr2)


# Fourier Transform
audio_fft = audio_data.copy()
freqs = audio_data.copy()
for nr in range(num_threads):
    t = threading.Thread(target=funcs.calc_fft, args=(audio_fft, freqs, int(nr*len(audio_fft)/num_threads), int((nr+1)*len(audio_fft)/num_threads),))
    threads.append(t)
    t.start()
for t in threads:
    t.join()    

# Plot dataset before and after fft.
funcs.plot_mag_phase(audio_data, audio_fft, freqs)
   
    
# Garbage collector
get_ipython().magic('reset -sf')
del audio_fft
del audio_data
del freqs
gc.collect()