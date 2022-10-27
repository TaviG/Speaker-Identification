from asyncio import exceptions
import os
import sys
from pytube import YouTube, exceptions
from scipy.io.wavfile import read, write
import numpy as np

DATASET_FILEPATH = sys.argv[1]

print(os.listdir(DATASET_FILEPATH))

for folder in os.listdir(DATASET_FILEPATH):
    videos = os.listdir(DATASET_FILEPATH + folder)
    for video in videos:
        yt = YouTube(f"https://www.youtube.com/watch?v={video}")
        try:
            streams = yt.streams.filter(file_extension='mp4')
        except exceptions.VideoUnavailable:
            continue

        streams[0].download("videos", filename=f"video.mp4")
        os.system("ffmpeg -i videos/video.mp4 -r 25 videos/video_25.mp4")

        if not os.path.exists(f"dataset/test/{folder}/{video}"):
            os.makedirs(f"dataset/test/{folder}/{video}")
        
        for file in os.listdir(DATASET_FILEPATH + folder + '/' + video + '/'):
            f = open(DATASET_FILEPATH + folder + '/' + video + '/' + file, "r")
            lines = f.readlines()

            first_frame = int(lines[7][:6])
            last_frame = int(lines[-1][:6])

            f.close()

            os.system(f"ffmpeg -i videos/video_25.mp4 -ss {first_frame/25} -to {last_frame/25} -c:v copy -c:a copy dataset/test/{folder}/{video}/{file[:-4]}.mp4")

        os.remove("videos/video.mp4")
        os.remove("videos/video_25.mp4")

# first_folder = os.listdir(DATASET_FILEPATH)[0]

# videos = os.listdir(DATASET_FILEPATH + first_folder)

# video = videos[0]

# yt = YouTube(f"https://www.youtube.com/watch?v={video}")
# streams = yt.streams.filter(file_extension='mp4')

# streams[0].download("videos", filename=f"video.mp4")

# os.system("ffmpeg -i videos/video.mp4 -r 25 videos/video_25.mp4")

print("Done")
