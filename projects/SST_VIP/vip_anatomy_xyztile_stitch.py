"""process xyz tiles from scanbox
zahra
july 2024
"""
    
import os, sys, shutil, tifffile, numpy as np, pandas as pd, re, tarfile, scipy
from datetime import datetime
from pathlib import Path

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from utils.utils import convert_zstack_sbx_to_tif
sbxsrcs = [r"X:\rna_fish_alignment_zstacks\240709\e218_head_only\240709_ZD_001_004\240709_ZD_001_004.sbx"]
for sbxsrc in sbxsrcs:
    convert_zstack_sbx_to_tif(sbxsrc)

# specs from imaging notes        
frames = 20; steps = 5 #um
volume = 200 #z, um
fldnm = '240709_ZD_001_004'
fldpth = rf"X:\rna_fish_alignment_zstacks\240709\e218_head_only\{fldnm}\{fldnm}_red.tif"
dst = rf'X:\rna_fish_alignment_zstacks\240709\e218_head_only\{fldnm}'
arr = tifffile.imread(fldpth)
subvol = int(((volume/steps)+1)*(frames))
for i in range(int(arr.shape[0]/subvol)):
    subvol_tif = arr[i*subvol:(subvol*(i+1))]
    savedst = os.path.join(dst, f'tile_{i:03d}.tif')
    tifffile.imwrite(savedst,subvol_tif)

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

# then run grid stitching in imagej using the tiles    