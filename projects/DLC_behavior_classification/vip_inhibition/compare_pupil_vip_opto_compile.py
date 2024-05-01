"""
zahra
compare correct and incorrect trials of pupil size during vip inhibition
"""
#%%
import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy, pandas as pd, seaborn as sns
import glob, shutil, h5py
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab')
from projects.DLC_behavior_classification.eye import consecutive_stretch_vralign, get_area_circumference_opto, perireward_binned_activity
from projects.DLC_behavior_classification import eye
import inhibition
# copy vr file to folder
vrsrc = r'\\storage1.ris.wustl.edu\ebhan\Active\all_vr_data'
picklesrc = r'I:\vip_inhibition'
#%% - step 1
# optional if you don't have the vr fl in the folder
inhibition.copyvrfl_matching_pickle(picklesrc, vrsrc)
#%% - step 2 - align to start of reward zone
range_val=8 # seconds before and after align
binsize=0.05
fs = 31.25 # frame rate
# path to pickle
dct_pre_reward = {}
for fl in os.listdir(picklesrc):
    if fl[-2:]=='.p':
        # read files
        pdst = os.path.join(picklesrc,fl)        
        with open(pdst, "rb") as fp: #unpickle
                vralign = pickle.load(fp)
        vrfl = glob.glob(pdst[:-15]+'*.mat')[0]
        f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
        VR = f['VR']
        # fix vars
        trialnum = vralign['trialnum']
        ybinned = vralign['ybinned']
        rewlocs = VR['changeRewLoc'][:][VR['changeRewLoc'][:]>0]
        rewards = vralign["rewards"]
        changeRewLoc = vralign["changeRewLoc"]
        time = vralign['timedFF']
        licks_threshold = vralign['lickVoltage']<=-0.065 # manually threshold licks
        crl = consecutive_stretch_vralign(np.where(changeRewLoc>0)[0])
        crl = np.array([min(xx) for xx in crl])
        eps = np.array([xx for ii,xx in enumerate(crl[1:]) if np.diff(np.array([crl[ii],xx]))[0]>5000])
        eps = np.append(eps, 0)
        eps = np.append(eps, len(changeRewLoc))
        eps = np.sort(eps)
        velocity = np.hstack(vralign['forwardvel'])
        # run glm
        areas, areas_res = get_area_circumference_opto(pdst, range_val, binsize)
        # init
        areas_per_ep = []; peri_reward_fail_after_success = []; peri_reward_fail_after_fail = []
        peri_reward_s_after_s = []; peri_reward_s_after_ss = []        
        # for each epoch
        for i in range(len(eps)-1):
            ypos = ybinned[eps[i]:eps[i+1]]            
            input_peri = areas_res[eps[i]:eps[i+1]][ypos>2]
            areas_per_ep.append(input_peri)
            rewards_ = rewards[eps[i]:eps[i+1]]            
            time_ = time[eps[i]:eps[i+1]]            
            # get trial type
            success, fail, str_trials, ftr_trials, ttr, total_trials, \
            fail_after_success, fail_after_fail, succ_after_one_succ, \
            succ_after_two_succ = inhibition.get_success_failure_trials(trialnum[eps[i]:eps[i+1]], rewards[eps[i]:eps[i+1]])         
            # success after 1 perirew
            # success after 2 perirew
            rewall_ep_ss = inhibition.get_peri_signal_of_fail_trial_types(succ_after_one_succ, \
                trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            rewall_ep_sss = inhibition.get_peri_signal_of_fail_trial_types(succ_after_two_succ, \
                trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            # failed - different trial types            
            rewall_ep_ff = inhibition.get_peri_signal_of_fail_trial_types(fail_after_fail, trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            rewall_ep_fs = inhibition.get_peri_signal_of_fail_trial_types(fail_after_success, trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            # save peri reward
            peri_reward_fail_after_fail.append(rewall_ep_ff)
            peri_reward_fail_after_success.append(rewall_ep_fs)
            peri_reward_s_after_s.append(rewall_ep_ss)
            peri_reward_s_after_ss.append(rewall_ep_sss)
            dct_pre_reward[fl] = [areas_per_ep, 
                peri_reward_fail_after_success, peri_reward_fail_after_fail, 
                peri_reward_s_after_s, peri_reward_s_after_ss] # trial per ep
#%%
# plot mean and standard error of different trial types

arrfs_all = []
arrff_all = []
arrss_all = []
arrsss_all = []
for k,v in dct_pre_reward.items():    
    arrfs_all.append(np.hstack(v[1]))
    arrff_all.append(np.hstack(v[2]))
    arrss_all.append(np.hstack(v[3]))
    arrsss_all.append(np.hstack(v[4]))

arrff=np.hstack(arrff_all)
arrfs=np.hstack(arrfs_all)
arrss=np.hstack(arrss_all)
arrsss=np.hstack(arrsss_all)

fig, ax = plt.subplots()
meanrewdFF = np.nanmean(arrsss,axis=1)
ax.plot(meanrewdFF, color='k',label='success_after_2success')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,int(range_val/binsize)*2), 
        meanrewdFF-scipy.stats.sem(arrsss,axis=1,nan_policy='omit'),
        meanrewdFF+scipy.stats.sem(arrsss,axis=1,nan_policy='omit'), color='k',alpha=0.2)

meanrewdFF = np.nanmean(arrss,axis=1)
ax.plot(meanrewdFF, color='darkslategray',label='success_after_1success')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,int(range_val/binsize)*2), 
        meanrewdFF-scipy.stats.sem(arrss,axis=1,nan_policy='omit'),
        meanrewdFF+scipy.stats.sem(arrss,axis=1,nan_policy='omit'), color='darkslategray',alpha=0.2)
    
meanrewdFF = np.nanmean(arrfs,axis=1)
ax.plot(meanrewdFF, color='darkorchid',label='fail_after_success')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,int(range_val/binsize)*2), 
        meanrewdFF-scipy.stats.sem(arrfs,axis=1,nan_policy='omit'),
        meanrewdFF+scipy.stats.sem(arrfs,axis=1,nan_policy='omit'), color='darkorchid',alpha=0.2)

meanrewdFF = np.nanmean(arrff,axis=1)
ax.plot(meanrewdFF, color='r',label='fail_after_fail')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,int(range_val/binsize)*2), 
        meanrewdFF-scipy.stats.sem(arrff,axis=1,nan_policy='omit'),
        meanrewdFF+scipy.stats.sem(arrff,axis=1,nan_policy='omit'), color='r',alpha=0.2)

ax.axvline(int(range_val/binsize), color='k', linestyle='--')
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
ax.set_xticklabels(range(-range_val, range_val+1, 1))
ax.set_title(f'Sessions: {len(list(dct_pre_reward.keys()))}, Animals: 4')
ax.set_ylabel('Pupil residual')
ax.set_xlabel('Time from start of reward zone (s)')
ax.legend()
