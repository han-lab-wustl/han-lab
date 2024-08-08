"""process xyz tiles from scanbox
zahra
july 2024
"""
    
import os, sys, shutil, tifffile as tif, numpy as np, pandas as pd, re, tarfile, scipy
from datetime import datetime
from pathlib import Path

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from utils.utils import convert_zstack_sbx_to_tif
sbxsrcs = [r"X:\rna_fish_alignment_zstacks\240709\e218_head_only\240709_ZD_001_004\240709_ZD_001_004.sbx"]
for sbxsrc in sbxsrcs:
    convert_zstack_sbx_to_tif(sbxsrc)
#%%
# specs from imaging notes        
frames = 20; steps = 5 #um
volume = 150 #z, um
fldnm = '240702_ZD_001_001'
fldpth = rf"X:\rna_fish_alignment_zstacks\240702\e217\{fldnm}\{fldnm}.tif"
dst = rf'X:\rna_fish_alignment_zstacks\240702\e217\{fldnm}'
arr = tifffile.imread(fldpth)
subvol = int(((volume/steps)+1)*(frames))
for i in range(int(arr.shape[0]/subvol)):
    subvol_tif = arr[i*subvol:(subvol*(i+1))]
    savedst = os.path.join(dst, f'tile_{i:03d}.tif')
    tifffile.imwrite(savedst,subvol_tif)
#%%
# then run motion corr with gerardo's pipeline
# then remake tifs
arr = tifffile.imread(fldpth)
for i in range(int(arr.shape[0]/subvol)):
    tilepath = os.path.join(dst, f'tile_{i:03d}_registered.mat')
    tile = scipy.io.loadmat(tilepath)
    tile = tile['chtemp']
    savedst = os.path.join(dst, f'tile_{i:03d}_registered.tif')
    tifffile.imwrite(savedst,np.fliplr(np.rot90(tile.T,axes=(1,2))))
    tile_nonreg = os.path.join(dst, f'tile_{i:03d}.tif')
    if os.path.exists(savedst): os.remove(tilepath); os.remove(tile_nonreg)
#%%
# split into z for bigstitcher
# crop in x
fldnm = '240702_ZD_001_001'
src = rf'X:\rna_fish_alignment_zstacks\240702\e217\{fldnm}'
fls = [os.path.join(src, fl) for fl in os.listdir(src) if 'registered.tif' in fl]
for fl in fls:
    arr = tif.imread(fl)
    arr = arr[:,:,70:732]
    print(fl)
    tif.imwrite(fl, arr)
#%%    
# make stack
fls = [os.path.join(src, fl) for fl in os.listdir(src) if 'fused' in fl]
fls.sort()
# to accoutn for imperfectly shaped individual tiles
dims=np.array([np.array(tif.imread(fl).shape) for fl in fls])
maxy,maxx = np.max(dims,axis=0)
stack = np.zeros((len(fls), maxy,maxx))
for i,fl in enumerate(fls):
    arr = tif.imread(fl)
    stack[i,:arr.shape[0],:arr.shape[1]] = arr
    print(fl)
tif.imwrite(os.path.join(src, os.path.basename(src)+'_stitched_stack.tif'), stack)
