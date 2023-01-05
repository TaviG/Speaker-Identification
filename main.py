# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:10:17 2022
@author: Tavi
"""
import sys
import os
import shutil

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
DESTINATION_FOLDER = "result"

audios, people = funcs.extract_ds(inputdir)

# Shuffle both arrays in the same way
audios, people = funcs.shuffle_equally(audios, people)

# Convert labels from ['id1234', 'id1235', 'id1321', ...] to [0, 1, 87, ..]
official_labels = [int(elem[2:]) for elem in people]
official_labels = np.array(
    [elem - min(official_labels) for elem in official_labels]
)

n_audios = len(audios)

# plots.data_distribution(official_labels, "people_distribution.jpg")

# Audio file read
audio_data = [funcs.read_wav_file(elem) for elem in audios]

# Statistical Moments
# Mean + Variance
mean, variance = funcs.calc_mean_variance(audio_data)

# # Plot means and variances
# plots.mean(mean, "mean.jpg")
# plots.variance(variance, "var.jpg")

# # Fourier Transform
# audio_fft, freqs = funcs.fft_of_ds(audio_data, num_threads)

# # Plot dataset before and after fft.
# plots.fft_mag_phase(audio_data, audio_fft, freqs)

# # Power spectral density
# n_psds = 6
# psds = funcs.power_spectral_density(audio_data[:n_psds], Fs)
# for idx, psd in enumerate(psds):
#     plots.psd(psd, idx)

# # Autocorrelation of one signal
# which = 0
# acorr = funcs.calc_accor(audio_data[which], mean[which], variance[which])
# plots.acorr(acorr, "Autocorrelation", "accor1.jpg")

# # Wiener Khinchin test
# wh = funcs.wiener_hincin(audio_data[which], mean[which], variance[which])
# plots.acorr(
#     wh,
#     "Real part of the inverse Fourier transform of the power spectrum",
#     "accor2.jpg",
# )

# # Covariance matrix
# covariance_matrix = funcs.calc_cov_mat(audio_data[32:64], 10000, 20000)
# plots.covariance_matrix(covariance_matrix)

# MFCCs
# min_length = funcs.min_length(audio_data)
# mfcc = [
#     librosa.feature.mfcc(
#         y=elem[:min_length], sr=Fs, n_mfcc=16, n_fft=400, hop_length=160
#     )
#     for elem in audio_data
# ]
# plots.mfcc(mfcc[10], Fs, "mfcc.jpg")

# mfcc_16_64 = [
#     librosa.feature.mfcc(
#         y=elem[64 * 160 : 2 * 64 * 159],
#         sr=Fs,
#         n_mfcc=16,
#         n_fft=400,
#         hop_length=160,
#     )
#     for elem in audio_data
# ]
# plots.mfcc(mfcc_16_64[10], Fs, "mfcc_16_64.jpg", "16x64")

mfcc_16_32 = [
    librosa.feature.mfcc(
        y=elem[5 * 32 * 160 : 6 * 32 * 159],
        sr=Fs,
        n_mfcc=16,
        n_fft=400,
        hop_length=160,
    )
    for elem in audio_data
]
plots.mfcc(mfcc_16_32[10], Fs, "mfcc_16_32.jpg", "16x32")

# mfcc_16_16 = [
#     librosa.feature.mfcc(
#         y=elem[16 * 160 : 2 * 16 * 159],
#         sr=Fs,
#         n_mfcc=16,
#         n_fft=400,
#         hop_length=160,
#     )
#     for elem in audio_data
# ]
# plots.mfcc(mfcc_16_16[10], Fs, "mfcc_16_16.jpg", "16x16")

# Scatter plots
scaler = StandardScaler()
# mfcc_scaled = [scaler.fit_transform(elem) for elem in mfcc]
# mfcc_scaled_flat = np.array([elem.flatten() for elem in mfcc_scaled])

# plots.scatter(
#     mfcc_scaled_flat[:, 1000],
#     mfcc_scaled_flat[:, 1001],
#     labels=official_labels,
#     n_labels=40,
#     title="scatter.jpg",
# )

# mfcc_scaled_16_16 = [scaler.fit_transform(elem) for elem in mfcc_16_16]
# mfcc_scaled_flat_16_16 = np.array(
#     [elem.flatten() for elem in mfcc_scaled_16_16]
# )

# plots.scatter(
#     mfcc_scaled_flat_16_16[:, 200],
#     mfcc_scaled_flat_16_16[:, 201],
#     labels=official_labels,
#     n_labels=40,
#     title="scatter_16_16.jpg",
#     dimension="16x16",
# )

mfcc_scaled_16_32 = [scaler.fit_transform(elem) for elem in mfcc_16_32]
mfcc_scaled_flat_16_32 = np.array(
    [elem.flatten() for elem in mfcc_scaled_16_32]
)

plots.scatter(
    mfcc_scaled_flat_16_32[:, 200],
    mfcc_scaled_flat_16_32[:, 201],
    labels=official_labels,
    n_labels=40,
    title="scatter_16_32.jpg",
    dimension="16x32",
)

# mfcc_scaled_16_64 = [scaler.fit_transform(elem) for elem in mfcc_16_64]
# mfcc_scaled_flat_16_64 = np.array(
#     [elem.flatten() for elem in mfcc_scaled_16_64]
# )

# plots.scatter(
#     mfcc_scaled_flat_16_64[:, 200],
#     mfcc_scaled_flat_16_64[:, 201],
#     labels=official_labels,
#     n_labels=40,
#     title="scatter_16_64.jpg",
#     dimension="16x64",
# )

# K-Means with 40 classes
accuracies = []

# kmeans_40 = KMeans(40, init="k-means++", random_state=42)
# kmeans_40.fit(mfcc_scaled_flat)
# pred_labels = kmeans_40.labels_

# plots.scatter(
#     kmeans_40.cluster_centers_[:, 1000],
#     kmeans_40.cluster_centers_[:, 1001],
#     labels=range(40),
#     n_labels=40,
#     title="kmeans_centers.jpg",
# )

# accuracies.append(funcs.Kmeans_accuracy(official_labels, pred_labels))

# kmeans_40_16_16 = KMeans(40, init="k-means++", random_state=42)
# kmeans_40_16_16.fit(mfcc_scaled_flat_16_16)
# pred_labels_16_16 = kmeans_40_16_16.labels_

# plots.scatter(
#     kmeans_40_16_16.cluster_centers_[:, 200],
#     kmeans_40_16_16.cluster_centers_[:, 201],
#     labels=range(40),
#     n_labels=40,
#     title="kmeans_centers_16_16.jpg",
#     dimension="16x16",
# )

# accuracies.append(funcs.Kmeans_accuracy(official_labels, pred_labels_16_16))

kmeans_40_16_32 = KMeans(40, init="k-means++", random_state=42)
kmeans_40_16_32.fit(mfcc_scaled_flat_16_32)
pred_labels_16_32 = kmeans_40_16_32.labels_

plots.scatter(
    kmeans_40_16_32.cluster_centers_[:, 200],
    kmeans_40_16_32.cluster_centers_[:, 201],
    labels=range(40),
    n_labels=40,
    title="kmeans_centers_16_32.jpg",
    dimension="16x32",
)

accuracies.append(funcs.Kmeans_accuracy(official_labels, pred_labels_16_32))

# kmeans_40_16_64 = KMeans(40, init="k-means++", random_state=42)
# kmeans_40_16_64.fit(mfcc_scaled_flat_16_64)
# pred_labels_16_64 = kmeans_40_16_64.labels_

# plots.scatter(
#     kmeans_40_16_64.cluster_centers_[:, 200],
#     kmeans_40_16_64.cluster_centers_[:, 201],
#     labels=range(40),
#     n_labels=40,
#     title="kmeans_centers_16_64.jpg",
#     dimension="16x64",
# )

# accuracies.append(funcs.Kmeans_accuracy(official_labels, pred_labels_16_64))

print(accuracies)

# # Finding optimum number of classes
# # get within cluster sum of squares for each value of k
# wcss = []
# max_clusters = 40
# for i in range(1, max_clusters):
#     kmeans_pca = KMeans(i, init="k-means++", random_state=42)
#     kmeans_pca.fit(mfcc_scaled_flat_16_32)
#     wcss.append(kmeans_pca.inertia_)
#     print(f"KMeans for {i} clusters done")

# # programmatically locate the elbow
# n_clusters = KneeLocator(
#     [i for i in range(1, max_clusters)],
#     wcss,
#     curve="convex",
#     direction="decreasing",
# ).knee
# print("Optimal number of clusters", n_clusters)

# # visualize the curve in order to locate the elbow
# plots.kmeans_elbow(max_clusters, wcss, n_clusters)

kmeans_9_16_32 = KMeans(9, init="k-means++", random_state=42)
kmeans_9_16_32.fit(mfcc_scaled_flat_16_32)
new_labels = kmeans_9_16_32.labels_

plots.scatter(
    mfcc_scaled_flat_16_32[:, 200],
    mfcc_scaled_flat_16_32[:, 201],
    labels=new_labels,
    n_labels=9,
    title="scatter_kmeans9_16_32.jpg",
    dimension="16x32",
)

# Store new class association in folder
if os.path.exists(DESTINATION_FOLDER):
    shutil.rmtree(DESTINATION_FOLDER)

for label in range(9):
    os.makedirs(os.path.join(DESTINATION_FOLDER, str(label)))

itemcounts = dict.fromkeys(range(9))
for i in range(9):
    itemcounts[i] = dict.fromkeys(range(40), 1)

for index, label in enumerate(new_labels):
    shutil.copy(
        audios[index],
        os.path.join(
            DESTINATION_FOLDER,
            str(label),
            f"{official_labels[index]}_{itemcounts[label][official_labels[index]]}.wav",
        ),
    )
    itemcounts[label][official_labels[index]] += 1

# # Garbage collector
# get_ipython().magic("reset -sf")
# del audio_fft
# del audio_data
# del mfcc
# del mfcc_scaled
# del mfcc_scaled_flat
# del freqs
# gc.collect()

print("Done")
