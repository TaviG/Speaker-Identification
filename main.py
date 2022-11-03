# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:10:17 2022

@author: Tavi
"""

import os
import sys


inputdir = sys.argv[1]

people = []

for folder in os.listdir(inputdir):
    people.append(os.path.join(inputdir, folder))
    
train = []
test = []

for person in people:
   l = os.listdir(person)
   for i in l[:int(0.8 * len(l))]:
       train.append(i)
   for i in l[int(0.8 * len(l)):]:
       test.append(i)

print("train:")
print(train)
print("test:")
print(test)
        