# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:20:26 2022

@author: Tavi
"""
from __future__ import unicode_literals
import os
import youtube_dl
import threading

base_path = r"C:\Users\Tavi\Desktop\CPPSMS\txt\\"
file_list = []
paths = []
ids = []
threads = []
nr_threads = 4


def download(videos):
    for video in videos:
        try:
            download_video(video)
        except Exception:
            pass

def download_video(video):
    ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video])


for path, subdirs, files in os.walk(base_path):
    for name in files:
        file_list.append(os.path.join(path, name))
        paths.append(path)
    
for path in paths:
    ids.append(path.split("\\")[-1])

ids = list(dict.fromkeys(ids))

videos = ["https://www.youtube.com/watch?v=" + s for s in ids]



lung = len(videos) 

#download(videos)


for nr in range(nr_threads):
    t = threading.Thread(target=download, args=(videos[int(nr*lung/nr_threads):int((nr+1)*lung/nr_threads)],))
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()