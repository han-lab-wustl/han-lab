# Zahra
# preprocess axon images for deepcad

import tifffile, os, shutil, SimpleITK as stik
import numpy as np

src = r'D:\zahra_axon_analysis\230509_TI\230509_TI_E192\230509_TI_E192_000_000\suite2p\plane0\reg_tif'
# make into big tif
imgs = [os.path.join(src, xx) for xx in os.listdir(src)]
imgs.sort()
bigarr = [stik.GetArrayFromImage(stik.ReadImage(img)) for img in imgs]
bigarr = np.concatenate(np.array(bigarr))[:,:,:650] # crop
dst = r'D:\zahra_axon_analysis\230509_TI\230509_TI_E192.tif'
tifffile.imwrite(dst, bigarr)
