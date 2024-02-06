#!/usr/bin/env python
# coding: utf-8

# # Preparation: load libraries

# In[11]:


# 0.Load all required libraries
import numpy as np
import matplotlib
import pandas as pd

from scipy.stats import mode
import sys
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time


# # 1. Read the CSV file 
# 
# 

# In[12]:


# 1.Read the CSV file

# Make a new variable named "df", which is a dataframe that loads the csv data
df = pd.read_csv('location_of_observation_spots.csv', encoding='unicode_escape')

# Display the first 2 rows of df
df.head(10)


# # 2. Download the Google Street View Image

# ## 2.1 Acquire a Google API Key
# 
# https://developers.google.com/maps/documentation/javascript/get-api-key

# In[4]:


# 3.1 Paste your Google API Key here, pay attention to the charge rules
key='AIzaSyAgl0nhKsCGcwfgvU9-lGWoveO7-HksfFM'


# # 2.2 Request metaData and download static image 

# In[16]:


import pandas as pd
import urllib,xmltodict,time,os,os.path

# Where to save the images
SavLoc = 'saved_files/'  #replace with your own location

SkipList=[]
atp=0
# Create an empty list to save the image names
downloaded_jpg = []
for f in files:
    print(f)
    if f.endswith(".jpg"):
        a,b = f.split('.',1)
        downloaded_jpg.append(a)
downloaded_jpg.sort()

print("downloaded SVI number:",len(downloaded_jpg))
print(downloaded_jpg)
print(downloaded_jpg)

indices = [int(i) for i in downloaded_jpg]

print(indices)

 ### 1.0 Read coordinates from the dataframe
list1 = [1,2,3,4,5]
list2 = [0,30,-30]
for i in range(len(df)):
    FID=i
#     for a,p in zip(list1,list2):
    for a in list1:
        lat,lon,heading=(str(df['Lat'][i]),str(df['Lng'][i]),str(df["back"+str(a)][i]))
        for p in list2:

            base  = "https://maps.googleapis.com/maps/api/streetview?size=800x800&location="
            fov   = '60' # focus of view
            pitch = str(p) # Vertical camera view angle towards road
            source='outdoor'
              # 2.2 Generate a urlSVI for the StreetViewImage SVI
            urlSVI = base+lat+','+lon+'&key='+key+'&fov='+fov+'&pitch='+pitch+'&source='+source +'&heading='+heading
#    ###        urlSVI = https://maps.googleapis.com/maps/api/streetview?size=600x300&location=46.414382,10.013988&heading=151.78&pitch=-0.76&key=AIzaSyAgt9qu13G4Js7F-xvGfOKc83Zp7ASBuos
#                 urlSVI = base+lat+','+lon+'&key='+key+'&fov='+fov+'&pitch='+pitch+'&heading='+heading+'&source='+source 
#             print(urlSVI)
            # 2.3 Save the SVI image
            filename=str(FID)+"_"+str(a)+"_"+str(p)+"_"+"2"+'.jpg'
            print(filename)
            urllib.request.urlretrieve(urlSVI, os.path.join(SavLoc,filename))
    atp+=1
    print("Attempt:%s, download photo%s"%(atp,FID))
     
print('Done saving the photo')   

