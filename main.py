# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:10:17 2022
@author: Tavi
"""
import sys

import funcs
import plots

from IPython import get_ipython
import gc
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

inputdir = sys.argv[1]
num_threads = 4
Fs = 16000

audios, people = funcs.extract_ds(inputdir)

# Shuffle both arrays in the same way
audios, people = funcs.shuffle_equally(audios, people)

# Convert labels from ['id1234', 'id1235', 'id1321', ...] to [0, 1, 87, ..]
official_labels = [int(elem[2:]) for elem in people]
official_labels = np.array([elem - min(official_labels) for elem in official_labels])

n_audios = len(audios)

# Split into 80% train 20% test
n_train = round(n_audios * 0.8)

X_train = audios[:n_train]
Y_train = people[:n_train]

X_test = audios[n_train:]
Y_test = people[n_train:]

plots.data_distribution(official_labels, 'people_distribution.jpg')

#Audio file read
audio_data = [funcs.read_wav_file(elem) for elem in audios]

# Statistical Moments
# Mean + Variance
mean, variance = funcs.calc_mean_variance(audio_data)

# Plot means and variances
plots.mean(mean, "mean.jpg")
plots.variance(variance, "var.jpg")

# Fourier Transform
audio_fft, freqs = funcs.fft_of_ds(audio_data, num_threads)

# Plot dataset before and after fft.
plots.fft_mag_phase(audio_data, audio_fft, freqs)

# Power spectral density
n_psds = 6
psds = funcs.power_spectral_density(audio_data[:n_psds], Fs)
for idx, psd in enumerate(psds):
    plots.psd(psd, idx)

# Autocorrelation of one signal
which = 0
acorr = funcs.calc_accor(audio_data[which], mean[which], variance[which])
plots.acorr(acorr, 'Autocorrelation', 'accor1.jpg')

# Wiener Khinchin test
wh = funcs.wiener_hincin(audio_data[which], mean[which], variance[which])
plots.acorr(wh, 'Real part of the inverse Fourier transform of the power spectrum', 'accor2.jpg')

# Covariance matrix
covariance_matrix = funcs.calc_cov_mat(audio_data[32:64], 10000, 20000)
plots.covariance_matrix(covariance_matrix)

# MFCCs
min_length = funcs.min_length(audio_data)
mfcc = [librosa.feature.mfcc(y=elem[:min_length], sr=Fs, n_mfcc=16, hop_length=500) for elem in audio_data]

plots.mfcc(mfcc[10], Fs)

# K-Means with 40 classes
scaler = StandardScaler()
mfcc_scaled = [scaler.fit_transform(elem) for elem in mfcc]
mfcc_scaled_flat = np.array([elem.flatten() for elem in mfcc_scaled])

plots.scatter(mfcc_scaled_flat[:, 1000], mfcc_scaled_flat[:, 1001], labels=official_labels, n_labels=40, title='scatter.jpg')

kmeans_40 = KMeans(40, init='k-means++', random_state=42)
kmeans_40.fit(mfcc_scaled_flat)
pred_labels = kmeans_40.labels_

plots.scatter(kmeans_40.cluster_centers_[:, 1000], kmeans_40.cluster_centers_[:, 1001], labels=range(40), n_labels=40, title='kmeans_centers.jpg')

accuracies = funcs.Kmeans_accuracy(official_labels, pred_labels)
print(accuracies)

# Finding optimum number of classes
# get within cluster sum of squares for each value of k
wcss = []
max_clusters = 40
for i in range(1, max_clusters):
    kmeans_pca = KMeans(i, init='k-means++', random_state=42)
    kmeans_pca.fit(mfcc_scaled_flat)
    wcss.append(kmeans_pca.inertia_)
    print(f"KMeans for {i} clusters done")

# programmatically locate the elbow
n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
print("Optimal number of clusters", n_clusters)

# visualize the curve in order to locate the elbow
plots.kmeans_elbow(max_clusters, wcss, n_clusters)

# # Garbage collector
# get_ipython().magic('reset -sf')
# del audio_fft
# del audio_data
# gc.collect()

print("Done")