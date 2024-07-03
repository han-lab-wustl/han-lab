"""process xyz tiles from scanbox
zahra
july 2024
"""
    
import os, sys, shutil, tifffile, numpy as np, pandas as pd, re, tarfile, scipy
from datetime import datetime
from pathlib import Path

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from utils.utils import convert_zstack_sbx_to_tif
sbxsrcs = [r"X:\rna_fish_alignment_zstacks\240702\240702_ZD_001_001\240702_ZD_001_001.sbx",
    r"X:\rna_fish_alignment_zstacks\240702\240702_ZD_001_002\240702_ZD_001_002.sbx"]
for sbxsrc in sbxsrcs:
    convert_zstack_sbx_to_tif(sbxsrc)

# specs from imaging notes        
frames = 20; steps = 5 #um
volume = 150 #z, um
fldnm = '240702_ZD_002_001'
fldpth = rf"X:\rna_fish_alignment_zstacks\240702\{fldnm}\{fldnm}_red.tif"
arr = tifffile.imread(fldpth)
dst = rf'X:\rna_fish_alignment_zstacks\240702\{fldnm}'
subvol = int(((volume/steps)+1)*(frames))
for i in range(int(arr.shape[0]/subvol)):
    subvol_tif = arr[i*subvol:(subvol*(i+1))]
    savedst = os.path.join(dst, f'tile_{i:03d}.tif')
    tifffile.imwrite(savedst,subvol_tif)

# then run motion corr with gerardo's pipeline
# then remake tifs
fldnm = '240702_ZD_002_001'
fldpth = rf"X:\rna_fish_alignment_zstacks\240702\{fldnm}\{fldnm}_red.tif"
arr = tifffile.imread(fldpth)
dst = rf'X:\rna_fish_alignment_zstacks\240702\{fldnm}'

for i in range(int(arr.shape[0]/subvol)):
    tilepath = os.path.join(dst, f'tile_{i:03d}_registered.mat')
    tile = scipy.io.loadmat(tilepath)
    tile = tile['chtemp']
    savedst = os.path.join(dst, f'tile_{i:03d}_registered.tif')
    tifffile.imwrite(savedst,np.fliplr(np.rot90(tile.T,axes=(1,2))))
    tile_nonreg = os.path.join(dst, f'tile_{i:03d}.tif')
    if os.path.exists(savedst): os.remove(tilepath); os.remove(tile_nonreg)

# then run grid stitching in imagej using the tiles    