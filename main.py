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
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import accuracy_score

inputdir = sys.argv[1]

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

official_labels = [int(elem[2:]) for elem in people]
official_labels = np.array([elem - min(official_labels) for elem in official_labels])

n_audios = len(audios)

# Split into 80% train 20% test
n_train = round(n_audios * 0.8)
    
X_train = audios[:n_train]
Y_train = people[:n_train]

X_test = audios[n_train:]
Y_test = people[n_train:]

# funcs.plot_data_distribution(Y_train, 'train_distribution.jpg')
# funcs.plot_data_distribution(Y_test, 'test_distribution.jpg')
# funcs.plot_data_distribution(people, 'people_distribution.jpg')

#Audio file read
audio_data = funcs.read_wav_files(audios)

# audio_cov = audio_data[32:64]
min_length = funcs.min_length(audio_data)
# audio_cov = [elem[10000:20000] for elem in audio_data]
# audio_cov = np.stack(audio_cov, axis=0)
# cov = np.matmul(audio_cov, audio_cov.T)

# plt.figure()
# plt.imshow(cov)
# plt.colorbar()
# plt.savefig('covariance.jpg')

mfcc = [librosa.feature.mfcc(y=elem[:min_length], sr=16000, n_mfcc=16, hop_length=500) for elem in audio_data]

scaler = StandardScaler()
mfcc_scaled = [scaler.fit_transform(elem) for elem in mfcc]
mfcc_scaled_flat = np.array([elem.flatten() for elem in mfcc_scaled])

# funcs.plot_scatter(mfcc_scaled_flat[:, 1000], mfcc_scaled_flat[:, 1001], labels=official_labels, n_labels=40, title='scatter.jpg')

# get within cluster sum of squares for each value of k
# wcss = []
# max_clusters = 40
# for i in range(1, max_clusters):
#     kmeans_pca = KMeans(i, init='k-means++', random_state=42)
#     kmeans_pca.fit(mfcc_scaled_flat)
#     wcss.append(kmeans_pca.inertia_)
#     print(f"KMeans for {i} clusters done")

# # programmatically locate the elbow
# n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
# print("Optimal number of clusters", n_clusters)
    
# # visualize the curve in order to locate the elbow
# fig = plt.figure(figsize=(10,8))
# plt.plot(range(1, max_clusters), wcss, marker='o', linestyle='--')
# # plt.vlines(n_clusters, ymin=0, ymax=max(wcss), linestyles='dashed')
# plt.xlabel('Number of Clusters', fontsize=18)
# plt.ylabel('Within Cluster Sum of Squares (WCSS)', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig('wcss.jpg')

kmeans_40 = KMeans(40, init='k-means++', random_state=42)
kmeans_40.fit(mfcc_scaled_flat)
pred_labels = kmeans_40.labels_

accuracies = {}
for label in np.unique(official_labels):
    idxes = official_labels == label
    predicted_classes = pred_labels[idxes]

    counts = {}
    for pred_class in np.unique(predicted_classes):
        counts[pred_class] = np.count_nonzero(predicted_classes == pred_class)

    accuracy = max(counts.values()) / sum(counts.values())

    accuracies[label] = accuracy


# funcs.plot_scatter(kmeans_40.cluster_centers_[:, 1000], kmeans_40.cluster_centers_[:, 1001], labels=range(40), n_labels=40, title='kmeans_centers.jpg')

# score = accuracy_score(official_labels, pred_labels)
# print(score)

# funcs.plot_mfcc(mfcc[10])

# # #Statistical Moments
# # #Mean + Variance
# mean, variance = funcs.calc_mean_variance(audio_data)

# # Plot means and variances
# funcs.plot_mean(mean, "mean.jpg")        
# funcs.plot_variance(variance, "var.jpg")

# # Power spectral density
# psds = funcs.power_spectral_density(audio_data[:10], 16000)
# for idx, psd in enumerate(psds):
#     funcs.plot_psd(psd, idx)

# # Autocorrelation of one signal
# ndata = np.array(audio_data[0]) - np.array(mean[0])
# acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:]
# acorr = acorr / variance[0] / len(ndata)
# funcs.plot_acorr(acorr)

# # Wiener Khinchin test
# size = 2 ** np.ceil(np.log2(2*len(audio_data[0]) - 1)).astype('int')
# fft = np.fft.fft(ndata, size)
# pwr = np.abs(fft) ** 2
# acorr2 = np.fft.ifft(pwr).real / variance[0] / len(ndata)
# acorr2 = acorr2[:len(ndata)]
# funcs.plot_acorr(acorr2)

# # Fourier Transform
# audio_fft = audio_data.copy()
# for nr in range(num_threads):
#     t = threading.Thread(target=funcs.calc_fft, args=(audio_fft, int(nr*len(audio_fft)/num_threads), int((nr+1)*len(audio_fft)/num_threads),))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()    

# # Plot dataset before and after fft.
# funcs.plot_mag_phase(audio_data, audio_fft)

# # Garbage collector
# get_ipython().magic('reset -sf')
# del audio_fft
# del audio_data
# gc.collect()

print("Done")