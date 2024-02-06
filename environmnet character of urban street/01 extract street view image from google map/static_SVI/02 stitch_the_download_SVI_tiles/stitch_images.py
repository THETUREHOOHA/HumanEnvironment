#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
from stitching import Stitcher
import numpy as np

# Set the directory containing the images to be stitched
folder = 'saved_files/'

# Get all files in alphabetic order
all_files = sorted(os.listdir(folder))

# Add folder to filename to have full path
all_files = [os.path.join(folder, name) for name in all_files]

print(all_files)

# Set the output directory for stitched images
output_path = 'stitch_files_output/'

# Create a dictionary to store filenames based on numbers
dictionary = {}

# Extract numbers from filenames and store them in the dictionary
for filename in all_files:
    number = int(filename.split("/")[-1].split("_")[0])
    if number not in dictionary:
        dictionary[number] = []
    dictionary[number].append(filename)

# Sort the dictionary by keys
sorted_dict = dict(sorted(dictionary.items(), key=lambda x: x[0]))
print(sorted_dict.keys())

# Stitch the images and handle exceptions if any errors occur
stitcher = Stitcher()
failed_images = []

for i in range(0, 84):
    try:
        panorama = stitcher.stitch(dictionary[i])
        cv2.imwrite(os.path.join(output_path, str(i) + '.jpg'), panorama)
        print(str(i) + '_' + '1' + '.jpg')
    except Exception as e:
        print(f"Stitching failed for image {i}: {str(e)}")
        failed_images.append(str(i) + '.jpg')

print(f"Failed to stitch {len(failed_images)} images: {failed_images}")

# Find the images with incorrect size
# Create a dictionary with file name and shape value
path = 'stitch_files_output/'
image_dict = {}

for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = cv2.imread(os.path.join(path, file))
        image_dict[file] = img.shape

print(image_dict)

# Define the size threshold
height_threshold = 1000
width_threshold = 1700

# Check the size of each image and print the filenames if they are below the threshold
for filename, shape in image_dict.items():
    if shape[0] < height_threshold or shape[1] < width_threshold:
        print(filename)

