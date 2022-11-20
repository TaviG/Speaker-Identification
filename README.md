# Speaker-Identification
Speaker Identification on VoxCeleb dataset using statistical moments, covariance analysis, PCA based on covariance matrix computation and K-means

## Prerequisite:

sudo apt install ffmpeg

## In virtual environment:

python3 -m venv

venv . venv/bin/activate

pip install requirements.txt

## To run extract_ds.py:

python extract_ds.py /path/to/dataset

## To run mp42wav.py

python mp42wav.py /path/to/dataset

## Dataset must be downloaded and stored locally from

https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

## Audio files

All the audio files in the dataset are stored as *.wav with the following characteristics:

- 16 kHz sampling rate
- mono format
- audio codec =  pcm_s16le

## Runtime of main.py script

To run everything in the script it took 2.44s
