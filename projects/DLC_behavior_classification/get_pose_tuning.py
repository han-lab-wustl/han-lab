# zahra
# pose tuning curve examples

import os, pickle, matplotlib.pyplot as plt, pandas as pd, numpy as np, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from projects.DLC_behavior_classification.eye import get_pose_tuning_curve
src = r"Y:\DLC\dlc_mixedmodel2"
pths = ['E201_05_May_2023_vr_dlc_align.p', 
        'E189_02_May_2023_vr_dlc_align.p',
        'T11_23_Aug_2023__vr_dlc_align.p', 
        'E186_23_Dec_2022_vr_dlc_align.p', 
        'E168_30_Mar_2023_vr_dlc_align.p']
for pth in pths:
    with open(os.path.join(src, pth),'rb') as fp: #unpickle
        vralign = pickle.load(fp)
    vralign['NoseTopPoint_y'][vralign['NoseTopPoint_likelihood'].astype('float32')<0.99]=np.nan
    vralign['NoseTip_y'][vralign['NoseTip_likelihood'].astype('float32')<0.99]=np.nan    
    vralign['NoseBottomPoint_y'][vralign['NoseBottomPoint_likelihood'].astype('float32') < 0.99] = 0
    pose = np.nanmean(np.array([vralign['NoseTopPoint_y'],vralign['NoseTip_y'],
                    vralign['NoseBottomPoint_y']]).astype('float32'), axis=0)
    pose_name = 'Nose Y Position'
    # vralign['WhiskerUpper_x'][vralign['WhiskerUpper_likelihood'].astype('float32')<0.99]=np.nan
    # vralign['WhiskerUpper1_x'][vralign['WhiskerUpper1_likelihood'].astype('float32')<0.99]=np.nan
    # vralign['WhiskerUpper3_x'][vralign['WhiskerUpper3_likelihood'].astype('float32')<0.99]=np.nan
    # vralign['WhiskerLower1_x'][vralign['WhiskerLower1_likelihood'].astype('float32')<0.99]=np.nan
    # vralign['WhiskerLower_x'][vralign['WhiskerLower_likelihood'].astype('float32') < 0.99] = 0
    # vralign['WhiskerLower3_x'][vralign['WhiskerLower3_likelihood'].astype('float32') < 0.99] = np.nan
    # whisker = np.nanmean(np.array([vralign['WhiskerUpper_x'],vralign['WhiskerUpper1_x'],
    #                 vralign['WhiskerUpper3_x'],
    #                 vralign['WhiskerLower_x'],vralign['WhiskerLower1_x'],
    #                 vralign['WhiskerLower3_x']]).astype('float32'), axis=0)

    gainf=3/2
    rewsize=15
    get_pose_tuning_curve(pth, vralign, pose, \
        gainf, rewsize, \
        pose_name, success=True)

