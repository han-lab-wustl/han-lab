"""
zahra
compare correct and incorrect trials during vip inhibition
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
#%% - step 2
range_val=5
binsize=0.05
fs = 31.25 # frame rate
# path to pickle
dct_pre_reward = {}
for fl in os.listdir(picklesrc):
    if fl[-2:]=='.p':
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
        areas, areas_res = get_area_circumference_opto(pdst, range_val, binsize)
        pre_reward = []; pre_reward_fail_after_success = []; pre_reward_fail_after_fail = []
        areas_per_ep = []; peri_reward = [];  peri_reward_fail_after_success = []; peri_reward_fail_after_fail = []
        for i in range(len(eps)-1):
            ypos = ybinned[eps[i]:eps[i+1]]            
            input_peri = areas_res[eps[i]:eps[i+1]][ypos>2]
            areas_per_ep.append(input_peri)
            rewards_ = rewards[eps[i]:eps[i+1]]            
            time_ = time[eps[i]:eps[i+1]]            
            # success
            normmeanrew_t, meanrew, normrewall_t, \
            rewall = perireward_binned_activity(np.array(input_peri), \
                                rewards_[ypos>2].astype(int),
                                time_[ypos>2], range_val, binsize)
            # failed - different trial types
            success, fail, str_trials, \
            ftr_trials, ttr, total_trials, \
            fail_after_success, fail_after_fail = inhibition.get_success_failure_trials(trialnum[eps[i]:eps[i+1]], rewards_)
            rewall_ep_ff = inhibition.get_peri_signal_of_fail_trial_types(fail_after_fail, trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            rewall_ep_fs = inhibition.get_peri_signal_of_fail_trial_types(fail_after_success, trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res)
            # mean per trials
            pre_reward.append(np.nanmean(rewall[:int(range_val/binsize),:],axis=0))
            pre_reward_fail_after_fail.append(np.nanmean(rewall_ep_ff[:int(range_val/binsize),:],axis=0))
            pre_reward_fail_after_success.append(np.nanmean(rewall_ep_fs[:int(range_val/binsize),:],axis=0))
            # save peri reward
            peri_reward.append(rewall); peri_reward_fail_after_fail.append(rewall_ep_ff)
            peri_reward_fail_after_success.append(rewall_ep_fs)
            dct_pre_reward[fl] = [areas_per_ep, peri_reward, 
                peri_reward_fail_after_success, peri_reward_fail_after_fail, 
                pre_reward, pre_reward_fail_after_success, pre_reward_fail_after_fail] # trial per ep
#%%
# mean_prerew_areas_per_ep = [np.concatenate(v[4]) for k,v in dct_pre_reward.items()]
# std_prerew_areas_per_ep = [np.concatenate(v[4]) for k,v in dct_pre_reward.items()]

# fails_mean_prerew_areas_per_ep = [np.concatenate(v[5]) for k,v in dct_pre_reward.items()]
# fails_std_prerew_areas_per_ep = [np.concatenate(v[5]) for k,v in dct_pre_reward.items()]

# # distribution of success vr. fails
# for i,an in enumerate(mean_prerew_areas_per_ep):
#     fig, ax = plt.subplots()
#     ax.hist(an, color = 'k', alpha=0.3)
#     ax.hist(fails_mean_prerew_areas_per_ep[i], color = 'r', alpha=0.3)
#     fig.suptitle(f'{list(dct_pre_reward.items())[i][0]}')
#%%
# heat map successes vs. fails
# plot mean and standard error
# peri_reward.append(rewall); peri_reward_fail.append(rewall_ep)
# dct_pre_reward[fl] = [areas_per_ep, peri_reward, peri_reward_fail, pre_reward, pre_reward_fail] # trial per ep

for k,v in dct_pre_reward.items():
    arr = np.hstack(v[1])
    arrfs = np.hstack(v[2])
    arrff = np.hstack(v[3])
    fig, ax = plt.subplots()
    meanrewdFF = np.nanmean(arr,axis=1)
    ax.plot(meanrewdFF, color='k',label='success')   
    xmin,xmax = ax.get_xlim()     
    ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(arr,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(arr,axis=1,nan_policy='omit'), color='k',alpha=0.2)
    meanrewdFF = np.nanmean(arrfs,axis=1)
    ax.plot(meanrewdFF, color='r',label='fail_after_success')   
    xmin,xmax = ax.get_xlim()     
    ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(arrfs,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(arrfs,axis=1,nan_policy='omit'), color='rosybrown',alpha=0.2)
    meanrewdFF = np.nanmean(arrff,axis=1)
    ax.plot(meanrewdFF, color='r',label='fail_after_fail')   
    xmin,xmax = ax.get_xlim()     
    ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(arrff,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(arrff,axis=1,nan_policy='omit'), color='r',alpha=0.2)
    ax.axvline(int(range_val/binsize), color='k', linestyle='--')
    ax.axvline(int(range_val/binsize)+10, color='aqua', linestyle='--')
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(range(-range_val, range_val+1, 1))
    ax.set_title(k)
    ax.set_ylabel('Pupil residual')
    ax.set_xlabel('Time from CS (s)')
    
