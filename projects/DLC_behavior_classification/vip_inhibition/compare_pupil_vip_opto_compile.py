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
        pre_reward = []; pre_reward_fail = []
        areas_per_ep = []; peri_reward = []
        for i in range(len(eps)-1):
            input_peri = areas_res[eps[i]:eps[i+1]]
            areas_per_ep.append(input_peri)
            rewards_ = rewards[eps[i]:eps[i+1]]            
            time_ = time[eps[i]:eps[i+1]]
            # success
            normmeanrew_t, meanrew, normrewall_t, \
            rewall = perireward_binned_activity(np.array(input_peri), \
                                rewards_.astype(int),
                                time_, range_val, binsize)
            # failed
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = eye.get_success_failure_trials(trialnum[eps[i]:eps[i+1]], rewards_)
            failtr_bool = np.array([any(yy.astype(int)==xx for yy in ftr_trials) for xx in trialnum[eps[i]:eps[i+1]]])
            failed_trialnum = trialnum[eps[i]:eps[i+1]][failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            ypos = ybinned[eps[i]:eps[i+1]]
            rews_centered[(ypos[failtr_bool] >= rewlocs[i]-5) & (ypos[failtr_bool] <= rewlocs[i]+5)]=1
            rews_iind = eye.consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            rewards_ep = rews_centered
            # fake time var
            time_ep = np.arange(0,rewards_ep.shape[0]/fs,1/fs)
            licks_threshold_ep = licks_threshold[eps[i]:eps[i+1]][failtr_bool]
            velocity_ep = velocity[eps[i]:eps[i+1]][failtr_bool]
            input_peri = areas_res[eps[i]:eps[i+1]][failtr_bool]
            normmeanrew_t, meanrew_ep, normrewall_t, \
            rewall_ep = perireward_binned_activity(np.array(input_peri), \
                                    rewards_ep.astype(int),
                                    time_ep, range_val, binsize)
            # mean per trials
            pre_reward.append(np.nanmean(rewall[:int(range_val/binsize),:],axis=0))
            pre_reward_fail.append(np.nanmean(rewall_ep[:int(range_val/binsize),:],axis=0))
            # save peri reward
            peri_reward.append([rewall, rewall_ep])
        dct_pre_reward[fl] = [areas_per_ep, peri_reward, pre_reward, pre_reward_fail] # trial per ep
#%%
# [areas_per_ep, peri_reward, pre_reward, pre_reward_fail] # trial per ep
mean_prerew_areas_per_ep = [np.concatenate(v[2]) for k,v in dct_pre_reward.items()]
std_prerew_areas_per_ep = [np.concatenate(v[2]) for k,v in dct_pre_reward.items()]

fails_mean_prerew_areas_per_ep = [np.concatenate(v[3]) for k,v in dct_pre_reward.items()]
fails_std_prerew_areas_per_ep = [np.concatenate(v[3]) for k,v in dct_pre_reward.items()]

# distribution of success vr. fails
for i,an in enumerate(mean_prerew_areas_per_ep):
    fig, ax = plt.subplots()
    ax.hist(an, color = 'k', alpha=0.3)
    ax.hist(fails_mean_prerew_areas_per_ep[i], color = 'r', alpha=0.3)
    fig.suptitle(f'{list(dct_pre_reward.items())[i][0]}')
#%%
# heat map successes vs. fails
for k,v in dct_pre_reward.items():
    arr = np.hstack([np.hstack(v[1][0]), np.hstack(v[1][1])])
    plt.figure()
    plt.imshow(arr.T)
    plt.axhline(np.hstack(v[1][0]).shape[1], color='k')
    plt.title(k)
    
