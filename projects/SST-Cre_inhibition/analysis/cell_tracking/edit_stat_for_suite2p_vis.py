# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:24:28 2023

@author: Han


take cells extracted/ tracked from suite2p and cell reg and make borders of ROIs to align to movies
"""

import os, numpy as np, tifffile as tif, shutil, scipy, cv2
from scipy import ndimage
import matplotlib.pyplot as plt

src = r"H:\imaging_data"
dst = r"Z:\20230124_run"
commoncells = scipy.io.loadmat('Z:/20230124_run/commoncells.mat')

days = [os.path.join(src, xx) for xx in os.listdir(src) if "22" in xx]
for i,day in enumerate(days):
    print(day)
    statpth = os.path.join(day, "suite2p", "plane0", "stat.npy")
    stat = np.load(statpth, allow_pickle=True)
    ops = np.load(os.path.join(day, "suite2p", "plane0", "ops.npy"), allow_pickle=True)
    commoncells = commoncells["commoncells"]
    statcc = stat[commoncells[:,i]-1] # -1 to account for python index    
    # dilate = [] #gonna take the max proj of this
    # how much to dilate
    kernel = np.asarray([[False, False, True, False, False],
                     [False, True, False, True, False],
                     [True, False, False, False, True],
                     [False, True, False, True, False],
                     [False, False, True, False, False]])
    
    for cl, cell in enumerate(statcc): # makes a mask of all cells
        mask = np.zeros((ops.all()['Ly'], ops.all()['Lx']))
        for x,ypix in enumerate(cell['ypix']):
            try:
                mask[ypix-1,cell['xpix'][x]-1]=1
                d = ndimage.binary_dilation(mask, kernel)
            except:
                print('cell does not fit')
        #save individual borders of cells
        tif.imsave(r'Z:\20230124_run\cell_borders_per_session\%s_cellno%05d.tif' % (os.path.basename(day),cl), (d-mask).astype("uint8"))
        print(r'Z:\20230124_run\cell_borders_per_session\%s_cellno%05d.tif' % (os.path.basename(day),cl))

#%%
src = r'Z:\20230124_run\cell_borders_per_session'
# make an array of frame, cell border, and trace per cell below (multiplot) and export as tif
#test
border=r'Z:/20230124_run/cell_borders_per_session/221206_cellno00008.tif'
frames=r'H:/imaging_data/221206/suite2p/plane0/reg_tif/file000_chan0.tif'
framestif=tif.imread(frames)
behavior=r'Z:/cellregtest_Fmats/221206_Fall.mat'
dFF = r'Z:/dff_221206-11.mat'
dff = scipy.io.loadmat(dFF)
dff=dff['dff'][0] #reformat
behav=scipy.io.loadmat(behavior)

fig,axes=plt.subplots(nrows=2, ncols=1)
ax=axes[0]
plt.imshow()