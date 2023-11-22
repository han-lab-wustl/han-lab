import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy, matplotlib.pyplot as plt, re
import h5py, pickle
import matplotlib
matplotlib.use('TkAgg')
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification.preprocessing import VRalign

if __name__ == "__main__":
    dlccsv = r'Y:\DLC\dlc_mixedmodel2\230406_E201DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
    vrfl = r'Y:\DLC\dlc_mixedmodel2\E201_06_Apr_2023_time(08_23_28).mat'
    savedst = r'Y:\DLC\dlc_mixedmodel2'
    VRalign(vrfl, dlccsv, savedst)

    # example on how to open the pickle file
    pdst = r"Y:\DLC\dlc_mixedmodel2\E201_06_Apr_2023_vr_dlc_align.p"
    with open(pdst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
    
    # print all keys
    vralign.keys()
    # you have to do this weird thing in matplotlib to make the plots pop out
    matplotlib.use('TkAgg')
    %matplotlib inline
    # plot hrz behavior with paw    
    # reformatting
    ypos = np.hstack(vralign['ybinned'])
    licks = np.hstack(vralign['licks'])
    # plots whisker and nose
    # converts low likelihood points to nans
    vralign['WhiskerUpper_y'][vralign['WhiskerUpper_likelihood'].astype('float32')<0.99]=np.nan
    vralign['NoseTopPoint_y'][vralign['NoseTopPoint_likelihood'].astype('float32')<0.99]=np.nan
    vralign['NoseTip_y'][vralign['NoseTip_likelihood'].astype('float32')<0.99]=np.nan
    vralign['NoseTip_y'][vralign['NoseTip_likelihood'].astype('float32')<0.99]=np.nan
    vralign['NoseBottomPoint_y'][vralign['NoseBottomPoint_likelihood'].astype('float32') < 0.99] = 0
    vralign['PawTop_x'][vralign['PawTop_likelihood'].astype('float32') < 0.99] = 0
    vralign['PawTop_y'][vralign['PawTop_likelihood'].astype('float32') < 0.99] = 0
    vralign['PawMiddle_x'][vralign['PawMiddle_likelihood'].astype('float32') < 0.99] = np.nan
    vralign['PawMiddle_y'][vralign['PawMiddle_likelihood'].astype('float32') < 0.99] = np.nan
    vralign['PawBottom_x'][vralign['PawBottom_likelihood'].astype('float32') < 0.99] = np.nan
    vralign['PawBottom_y'][vralign['PawBottom_likelihood'].astype('float32') < 0.99] = np.nan
    paw_y = np.nanmean(np.array([vralign['PawTop_y'],vralign['PawBottom_y'],
                     vralign['PawMiddle_y']]).astype('float32'), axis=0)
    paw_x = np.nanmean(np.array([vralign['PawTop_x'],vralign['PawBottom_x'],
                     vralign['PawMiddle_x']]).astype('float32'), axis=0)

    whisker = vralign['WhiskerUpper_y']#[vralign['WhiskerUpper_likelihood'].astype('float32')>0.99]
    nose = vralign['NoseTip_y']#[vralign['NoseTip_likelihood'].astype('float32')>0.99]
    fig, axs = plt.subplots()
    axs.plot(ypos, color='slategray',
            linewidth=0.5)
    axs.scatter(np.argwhere(licks>0).T[0], ypos[licks>0], color='r', marker='.')
    axs.plot(scipy.ndimage.gaussian_filter(whisker/3,1), label = 'whisker')
    axs.plot(scipy.ndimage.gaussian_filter(nose/5,1), label = 'nose')

    axs.plot(scipy.ndimage.gaussian_filter(paw_y/2,1), color = 'olive', 
             label = 'paw_y', alpha=0.5)
    axs.legend()
    axs.set_title(os.path.basename(pdst))
