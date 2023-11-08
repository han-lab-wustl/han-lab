import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy.io as sio, matplotlib.pyplot as plt, re
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
    fig, axs = plt.subplots()
    # reformatting
    ypos = np.hstack(vralign['ybinned'])
    licks = np.hstack(vralign['licks'])
    whisker = vralign['WhiskerUpper_y']#[vralign['WhiskerUpper_likelihood'].astype('float32')>0.99]
    nose = vralign['NoseTip_y']#[vralign['NoseTip_likelihood'].astype('float32')>0.99]
    axs.plot(ypos, color='slategray',
            linewidth=0.5)
    axs.scatter(np.argwhere(licks>0).T[0], ypos[licks>0], color='r', marker='.')
    axs.plot(whisker/3)
    axs.plot(nose/5)
    
    plt.title(os.path.basename(matfl))
