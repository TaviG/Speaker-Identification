import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def data_distribution(array, title):
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
    plt.close()

def covariance_matrix(cov):
    plt.figure()
    plt.imshow(cov)
    plt.colorbar()
    plt.savefig('covariance.jpg')
    plt.close()

def mean(means, title):
    m = means.copy()
    m.sort()
    plt.figure()
    plt.plot(m)
    plt.xlabel('Audio recording')
    plt.ylabel('Mean value')
    plt.title("Mean value of all audio recordings")
    plt.savefig(title)
    plt.close()

def variance(var, title):
    v = var.copy()
    v.sort()
    plt.figure()
    plt.plot(v)
    plt.xlabel('Audio recording')
    plt.ylabel('Variance')
    plt.title("Dataset variance")
    plt.savefig(title)
    plt.close()

def fft_mag_phase(data, fft, freqs):
    for i in range(6):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(data[i])
        plt.title("Audio recording")
        plt.subplot(2,1,2)
        plt.plot(freqs[i], np.abs(fft[i]))
        plt.title("FFT")
        plt.savefig('fft_analysis'+str(i)+'.jpg')
        plt.close()

def psd(psd, idx):
    plt.figure()
    plt.semilogy(psd[0], psd[1])
    plt.title("Power spectral density")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.savefig(f'psd{idx}.jpg')
    plt.close()

def acorr(acorr, title, filename):
    plt.figure()
    plt.plot(acorr)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def mfcc(mfcc, Fs):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis="time", sr=Fs)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig('mfcc.jpg')
    plt.close()

def scatter(x, y, labels, n_labels, title):
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
    plt.close()

def kmeans_elbow(max_clusters, wcss, n_clusters):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, max_clusters), wcss, marker='o', linestyle='--')
    plt.vlines(n_clusters, ymin=0, ymax=max(wcss), linestyles='dashed')
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('Within Cluster Sum of Squares (WCSS)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('wcss.jpg')
    plt.close()