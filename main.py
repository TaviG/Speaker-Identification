# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:10:17 2022

@author: Tavi
"""

import os
import sys
import random
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
#import threading

inputdir = sys.argv[1]

people = []
audios = []
num_threads = 4


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

#Audio file read
data = []
for x in X_train:
    _, info = wavfile.read(x)        
    data.append(info)

#Statistical Moments

#Mean + Variance

means = [] 
variance = [] 
datafft = np.zeros(len(data)) 
threads = []

for info in data:
    means.append(np.mean(info))
    variance.append(np.var(info))

#Plot means and variances
plt.plot(means)
plt.figure(), plt.plot(variance)        

# Fourier Transform

def fft(data, i, j):
    return
        
'''
for nr in num_threads:
    t = threading.Thread(target=fft, args=(data, int(nr*len(data)/num_threads), int((nr+1)*len(data)/num_threads),))
    threads.append(t)
'''
for info in data:
    x = np.fft.fft(info)
    datafft.append(x) 

# Plot dataset before and after fft.

plt.figure(), plt.plot(data[0])
plt.figure(), plt.plot(datafft[0]) # cast to real part



