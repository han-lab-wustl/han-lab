"""
zahra
compare correct and incorrect trials of pupil size during vip inhibition
"""
#%%
import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy, pandas as pd, seaborn as sns, random
import glob, shutil, h5py, re
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.patches as patches

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=16)          # controls default text sizes
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab')
from projects.DLC_behavior_classification.eye import consecutive_stretch_vralign, get_area_circumference_opto, perireward_binned_activity # type: ignore
from projects.DLC_behavior_classification import eye # type: ignore
import inhibition
# copy vr file to folder
vrsrc = r'\\storage1.ris.wustl.edu\ebhan\Active\all_vr_data'
picklesrc = r'I:\vip_inhibition'
#%% - step 1
# optional if you don't have the vr fl in the folder
# inhibition.copyvrfl_matching_pickle(picklesrc, vrsrc)
#%% - step 2 - align to start of reward zone
# import csv
df = pd.read_csv(r"I:\pupil_conddf.csv", index_col=None)
range_val=8 # seconds before and after align
binsize=0.05
fs = 31.25 # frame rate
# path to pickle
dct_pre_reward = {}
for fl in os.listdir(picklesrc):
    if fl[-2:]=='.p':
        # read files
        pdst = os.path.join(picklesrc,fl)
        # match file with condition df  
        mouse_nm = re.split(r"_", os.path.basename(pdst))[0].lower()
        match_str = re.search(r'\d{2}_\w{3}_\d{4}', os.path.basename(pdst))
        date = datetime.strptime(match_str.group(), '%d_%b_%Y').date()
        date = str(date)
        if sum((df.animals==mouse_nm) & (df.date==date))>0: # get opto animals
            print(pdst)
            with open(pdst, "rb") as fp: #unpickle
                    vralign = pickle.load(fp)
            optoep = df.loc[(df.animals==mouse_nm) & (df.date==date), 'optoep'].values[0]            
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
            if optoep<2: 
                optoep = random.randint(2,3)
                if len(eps)<4: optoep=2 # if not more than 2 ep or small ep3
            velocity = np.hstack(vralign['forwardvel'])
            # run glm
            areas, areas_res = get_area_circumference_opto(pdst, range_val, binsize)
            # init
            areas_per_ep = []; peri_reward_fail_after_success = []; peri_reward_fail_after_fail = []
            peri_reward_s_after_s = []; peri_reward_s_after_ss = []        
            # for each epoch
            i = optoep-2 # ctrl ep
            # for i in range(len(eps)-1):
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
            i = optoep-1 # only opto ep
            # for i in range(len(eps)-1):
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
            # remake for control var saving
            optoep = df.loc[(df.animals==mouse_nm) & (df.date==date), 'optoep'].values[0]            
            cond = df.loc[(df.animals==mouse_nm) & (df.date==date), 'in_type'].values[0]            
            dct_pre_reward[fl] = [areas_per_ep, 
                peri_reward_fail_after_success, peri_reward_fail_after_fail, 
                peri_reward_s_after_s, peri_reward_s_after_ss,
                mouse_nm, date, optoep, cond] # trial per ep
#%%
# plot mean and standard error of different trial types
conditions = ['vip','sst']
for condition in conditions:
    if condition=='vip': y1=-100; y2=140 # ylim
    else: y1=-200; y2=400
    arrfs_all = []
    arrff_all = []
    arrss_all = []
    arrsss_all = []
    for k,v in dct_pre_reward.items():  # prev epoch
        if v[7]>1 and v[8]==condition: # opto days
            arrfs_all.append(v[1][0])
            arrff_all.append(v[2][0])
            arrss_all.append(v[3][0])
            arrsss_all.append(v[4][0])

    arrff=np.hstack(arrff_all)
    arrfs=np.hstack(arrfs_all)
    arrss=np.hstack(arrss_all)
    arrsss=np.hstack(arrsss_all)

    fig, axes = plt.subplots(ncols=2,nrows=2,
                        figsize=(10,10))
    ax = axes[0,0]
    meanrewdFF = np.nanmean(arrsss,axis=1)
    ax.plot(meanrewdFF, color='k',label='success_after_2success')   
    xmin,xmax = ax.get_xlim()     
    ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(arrsss,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(arrsss,axis=1,nan_policy='omit'), 
            color='k',alpha=0.2)

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
    ax.set_title('Control Ep.')
    ax.set_ylabel('Pupil residual')
    ax.set_ylim(y1,y2)
    ax.spines[['right', 'top']].set_visible(False)

    arrfs_all = []
    arrff_all = []
    arrss_all = []
    arrsss_all = []
    for k,v in dct_pre_reward.items():  # opto epoch  
        if v[7]>1 and v[8]==condition:
            arrfs_all.append(v[1][1])
            arrff_all.append(v[2][1])
            arrss_all.append(v[3][1])
            arrsss_all.append(v[4][1])

    if len(arrff_all)>0:
        arrff=np.hstack(arrff_all)
    else:
        arrff=[]
    if len(arrfs_all)>0:
        arrfs=np.hstack(arrfs_all)
    else:
        arrfs = []
    arrss=np.hstack(arrss_all)
    arrsss=np.hstack(arrsss_all)

    ax = axes[0,1]
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

    ax.add_patch(patches.Rectangle(
        xy=(0,y1),  # point of origin.
        width=160, height=abs(y1)+abs(y2), linewidth=1,
        color='red', alpha=0.15))

    ax.axvline(int(range_val/binsize), color='k', linestyle='--')
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(range(-range_val, range_val+1, 1))
    ax.set_title(f'{condition.upper()} Inhibition \n Animals: 3')
    ax.set_ylim(y1,y2)
    ax.spines[['right', 'top']].set_visible(False)

    # control days
    arrfs_all = []
    arrff_all = []
    arrss_all = []
    arrsss_all = []
    for k,v in dct_pre_reward.items():  # prev epoch
        if v[7]<2 and v[8]==condition: 
            arrfs_all.append(v[1][0])
            arrff_all.append(v[2][0])
            arrss_all.append(v[3][0])
            arrsss_all.append(v[4][0])

    arrff=np.hstack(arrff_all)
    arrfs=np.hstack(arrfs_all)
    arrss=np.hstack(arrss_all)
    arrsss=np.hstack(arrsss_all)

    ax = axes[1,0]
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
    ax.set_title('Control Ep.')
    ax.set_ylim(y1,y2)
    ax.spines[['right', 'top']].set_visible(False)

    arrfs_all = []
    arrff_all = []
    arrss_all = []
    arrsss_all = []
    for k,v in dct_pre_reward.items():  # opto epoch
        if v[7]<2 and v[8]==condition: 
            arrfs_all.append(v[1][1])
            arrff_all.append(v[2][1])
            arrss_all.append(v[3][1])
            arrsss_all.append(v[4][1])

    arrff=np.hstack(arrff_all)
    arrfs=np.hstack(arrfs_all)
    arrss=np.hstack(arrss_all)
    arrsss=np.hstack(arrsss_all)

    ax = axes[1,1]
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
    ax.set_title('Comparison Ep.')
    ax.set_xlabel('Time from start of reward zone (s)')
    ax.set_ylim(y1,y2)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.spines[['right', 'top']].set_visible(False)