#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:59 2022

@author: jbarry

Purpose: Script to create four unique lists of stimuli
"""
import os
import random
import shutil


files_list = []

for root, dirs, files in os.walk("<SOURCE_DIR>"):
    for file in files:
        #all 
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            files_list.append(os.path.join(root, file))


#print images
#lets me count and print the amount of jpeg,jpg,pmg 
file_count = len(files_list)
print file_count

# print files_list   
filesToCopy = random.sample(files_list, 2)  #prints two random files from list 

destPath = "<DESTINATION_DIR>"

# if destination dir does not exists, create it
if os.path.isdir(destPath) == False:
        os.makedirs(destPath)

# iteraate over all random files and move them
for file in filesToCopy:
    shutil.move(file, destPath)