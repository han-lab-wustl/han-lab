import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy, matplotlib.pyplot as plt, re
import h5py, pickle
import matplotlib
matplotlib.use('TkAgg')
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification.preprocessing import VRalign

if __name__ == "__main__":
        # make sure you convert the behavior mat file first!!!
        # e.g. in matlab
        # load('D:\adina_vr_files\E218_09_Nov_2023_time(10_40_41).mat')
        # save('D:\adina_vr_files\E218_09_Nov_2023_time(10_40_41).mat', 'VR', '-v7.3')                        
        dlccsv = [r"D:\Tail_E186\Tail_221029_E186-Adina-2023-01-19\230508_E200DLC_resnet50_Tail_221029_E186Jan19shuffle1_250000.csv"]                
        vrfl = [r"D:\adina_vr_files\VR_data\E200_08_May_2023_time(08_54_41).mat"]
        dlccsv = [r"I:\dlc_inference\230502_E201DLC_resnet50_PupilTrainingJul7shuffle1_500000.csv"]
        vrfl = [r"D:\adina_vr_files\VR_data\E201_02_May_2023_time(09_16_02).mat"]
        savedst = r"D:\Tail_E186"
        savedst = r"I:\pupil_pickles"
        for i in range(len(dlccsv)): # align beh with video data
                VRalign(vrfl[i], dlccsv[i], savedst)

        # example on how to open the pickle file
        pdst = os.path.join(savedst, "E200_14_Mar_2023_vr_dlc_align.p")
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

#tail video (Adina)
        vralign['LowerBack_y'][vralign['LowerBack_likelihood'].astype('float32')<0.99]=np.nan
        vralign['MidBack_y'][vralign['MidBack_likelihood'].astype('float32')<0.99]=np.nan
        vralign['UpperBack_y'][vralign['UpperBack_likelihood'].astype('float32')<0.99]=np.nan
        
        back_y = np.nanmean(np.array([vralign['LowerBack_y'],vralign['MidBack_y'],
                        vralign['UpperBack_y']]).astype('float32'), axis=0)
        fig, axs = plt.subplots()
        axs.plot(ypos, color='slategray',
                linewidth=0.5)
        axs.plot(vel/20, label = 'forwardvel')
        axs.plot(reward*20)
        axs.plot(scipy.ndimage.gaussian_filter(back_y/3,1), label = 'back_y')
        axs.scatter(np.argwhere(licks>0).T[0], ypos[licks>0], color='r', marker='.')

        axs.legend()
        axs.set_title(os.path.basename(pdst))
        ypos = np.hstack(vralign['ybinned'])
        vel = np.hstack(vralign['forwardvel'])
        reward = np.hstack(vralign['rewards'])