    """process xyz tiles from scanbox
    zahra
    july 2024
    """
    
import os, sys, shutil, tifffile, numpy as np, pandas as pd, re, tarfile, scipy
from datetime import datetime
from pathlib import Path

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from utils.utils import convert_zstack_sbx_to_tif
sbxsrcs = [r"X:\zstacks\240701_ZD_001_005\240701_ZD_001_005.sbx",
    r"X:\zstacks\240701_ZD_002_004\240701_ZD_002_004.sbx",
    r"X:\zstacks\240701_ZD_002_005\240701_ZD_002_005.sbx"]
for sbxsrc in sbxsrcs:
    convert_zstack_sbx_to_tif(sbxsrc)

# specs from imaging notes        
frames = 20; steps = 5 #um
volume = 150 #z, um
fldnm = '240701_ZD_002_005'
fldpth = rf"X:\zstacks\{fldnm}\{fldnm}_red.tif"
arr = tifffile.imread(fldpth)
dst = rf'X:\zstacks\{fldnm}'
subvol = int(((volume/steps)+1)*(frames))
for i in range(int(arr.shape[0]/subvol)):
    subvol_tif = arr[i*subvol:(subvol*(i+1))]
    savedst = os.path.join(dst, f'tile_{i:03d}.tif')
    tifffile.imsave(savedst,subvol_tif)

# then run motion corr with gerardo's pipeline
# then remake tifs
for i in range(int(arr.shape[0]/subvol)):
    tilepath = os.path.join(dst, f'tile_{i:03d}_registered.mat')
    tile = scipy.io.loadmat(tilepath)
    tile = tile['chtemp']
    savedst = os.path.join(dst, f'tile_{i:03d}_registered.tif')
    tifffile.imwrite(savedst,np.fliplr(np.rot90(tile.T,axes=(1,2))))
    tile_nonreg = os.path.join(dst, f'tile_{i:03d}.tif')
    if os.path.exists(savedst): os.remove(tilepath); os.remove(tile_nonreg)
    