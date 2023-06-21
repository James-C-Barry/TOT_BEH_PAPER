# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:49:10 2022

@author: jericho
"""

import PIL
import os
import os.path
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = r'/bcbl/home/home_g-m/jbarry/TOT_MRI_NEW/Stimuli/non_famous_pictures/places/local' # Replace this with the directory that contains the images
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    maxsize = (500, 500) # Edit these numbers for the maximum size you want
    img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    img.save("/bcbl/home/home_g-m/jbarry/TOT_MRI_NEW/Stimuli/non_famous_pictures/places/resized/local/" + file + ".png") # Replace this for where you want to save the images

# Rename files with the name and a seqential numbering system

directory = "path_to_directory"  # Replace with the path to your target directory
new_name = "new_filename"  # Replace with the desired new filename

count = 1

for filename in os.listdir(directory):
    if filename.endswith(".png"):  # Change the file extension if necessary
        new_filename = f"{new_name}_{count}.txt"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        count += 1





#paths = (os.path.join(root, filename)
        #for root, _, filenames in os.walk('/bcbl/home/home_g-m/jbarry/TOT_MRI_NEW/Stimuli/non_famous_pictures/places/resized/local') #This was just to rename them
        #for filename in filenames)

#for path in paths:
    # the '#' in the example below will be replaced by the '-' in the filenames in the directory
    #newname = path.replace('.jpg','').replace('.jpeg','').replace ('.JPG','')
    #if newname != path:
        #os.rename(path, newname)
        
        
#path = '/bcbl/home/home_g-m/jbarry/TOT_MRI_NEW/Stimuli/non_famous_pictures/places/resized/local'
#for filename in os.listdir(path):
    #print(filename)
    #os.rename(os.path.join(path,filename),os.path.join(path, filename.replace(' ', '_').lower()))

