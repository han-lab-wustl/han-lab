# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:47:19 2023

@author: Han
"""

import os, numpy as np, tifffile as tif, shutil

dr = r'H:\imaging_data'
concattifs = [] #in order of days
concattifnm = [] #also names of days
days = os.listdir(dr); days.sort()
dst = r'H:\imaging_data\concat'
for day in days:
    print(day)
    daydr = os.path.join(dr, day)
    # find images from session, split up in 3000 frame tifs
    tifs = [os.path.join(daydr, tiff) for tiff in os.listdir(daydr) if '.tif' in tiff]; tifs.sort()
    [shutil.copy(tiff, dst) for tiff in tifs]
    tifs = None