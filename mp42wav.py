# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:50:19 2022

@author: Tavi
"""

import os
import sys


inputdir = sys.argv[1]

filenames = []

for path, subdirs, files in os.walk(inputdir):
    for name in files:
        filenames.append(os.path.join(path, name))
        
for filename in filenames:
    if(filename.endswith(".mp4")):
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 -ac 1 -y {}.wav'.format(filename, filename[:-4]))
    else:
        continue

print("Done")