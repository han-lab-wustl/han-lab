""" 
Ziyi's dopamine hrz with opto analysis
For each day, only one epoch has opto, in this one epoch, every other trial has opto at the oppsite 
location of the reward zone
Adapted from zahra's dopamine hrz analysis
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
import scipy.stats
import matplotlib.gridspec as gridspec

from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
from projects.opto.behavior.behavior import smooth_lick_rate, get_lick_selectivity
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity, get_position_first_lick_after_rew
from Ziyi.python_processing_code.functions.function_code import find_start_points, has_internal_nan_gap, remove_trials_with_internal_nan_gap, nan_after_reward_all_epochs
from projects.DLC_behavior_classification import eye
plt.rcParams["font.family"] = "Arial"

#%%
plt.close('all')
# save to pdf
animal = ''
src = r"E:\Ziyi\Data\VTA_mice\hrz\E277"
src = os.path.join(src,animal)
dst = r"E:\Ziyi"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"hrz_{os.path.basename(src)}.pdf"))
save_dir = r"C:\Users\HanLab\Downloads\per_day"
save_path = 'lick_rate_block_consumption_licks_new.npz'
#days = np.arange(19,23)
# opto days [19,20,22,23]
# control days [21,24]
days = [28,29,32,34,35,39,41,43,45,47]
save_npz = False
save_figs = False

opto_day = True
epsALL_mask = np.array([True, False, False])

opto_start = True
if opto_start:
    num = 1
else:
    num = 0	


'''
i = epsALL_mask.index(True)
if i == len(arr) - 1:
    ep_after_opto = np.nan
else:
    arr[i] = False
    arr[i + 1] = True
    ep_after_opto = arr
'''

#opto_day = False

range_val=10; binsize=0.2
numtrialsstim=2 # every 10 trials stim w 1 trial off
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
# figs = True # print out per day figs

if opto_day:
    behavior_opto_shade = 'darkorange'
    opto_shade = 'lightgreen'
    day_label = 'Opto Day'
else:
    behavior_opto_shade = 'salmon'  # or just 'salmon'
    opto_shade = 'skyblue'
    day_label = 'Control Day'


day_date_dff_opto ={}
day_date_dff_nonopto ={}
day_date_dff_nonopto_eps = {}
day_date_dff_nonopto_after_opto = {}

# opto centered lickrate in opto epoch
day_lickrate_opto_centered_opto = {}
day_lickrate_opto_centered_nonopto = {}
#day_lickerate_opto_centered_




for day in days: 
    plndff_opto = []
    plndff_nonopto = []
    plndff_nonopto_eps = []
    lick_rate_opto_centered_opto = []
    lick_rate_opto_centered_nonopto = []
    stimspth = list(Path(os.path.join(src, str(day))).rglob('*000*.mat'))[0]
    #stimspth = Path(os.path.join(src, str(day))).rglob('*000*.mat')
    stims = scipy.io.loadmat(stimspth)
    stims = np.hstack(stims['stims']) # nan out stims
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        VR = params['VR'][0][0][()]
        gainf = VR['scalingFACTOR'][0][0]
        try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/gainf        
        except:
            rewsize = 10
        

        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/basemean

        
        if old:
            dff = np.hstack(params['params'][0][0][7][0][0])/np.nanmean(np.hstack(params['params'][0][0][7][0][0]))
            # dff = np.hstack(params['params'][0][0][10])/np.nanmean(np.hstack(params['params'][0][0][10]))
        else:
            dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))
        
        # plt.close(fig)
        framerate = 7.8
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(5).mean().values)
        rewards = np.hstack(params['solenoid2'])
        #rewards = rewards.astype(bool)
        velocity = np.hstack(params['forwardvel'])
        veldf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(veldf.rolling(10).mean().values)
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        timedFF = np.hstack(params['timedFF'])
        changeRewLoc = np.hstack(params['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
        eps = np.append(eps, len(changeRewLoc))
        lick_rate = smooth_lick_rate(licks, 1/framerate, sigma_sec=0.5)

        lick_rate = nan_after_reward_all_epochs(lick_rate, rewards, trialnum, eps)
        
        #eps_opto_mask = np.zeros(len(stims), dtype=bool)

        
        changeRewLocALL = np.hstack(params['changeRewLocALL'])
        epsALL = np.where(changeRewLocALL>0)[0];rewlocs = changeRewLocALL[epsALL]/gainf
        epsALL = np.append(epsALL, len(changeRewLocALL))

        # Initialize eps_opto_mask
        eps_opto_mask = np.zeros(eps[-1], dtype=bool)


        # Precompute which epsALL segments have stim
           
        if opto_day:
            epsALL_has_stim = np.array([ np.any(stims[epsALL[j]:epsALL[j+1]] == 1)
            for j in range(len(epsALL) - 1)])
        else:
            epsALL_has_stim = epsALL_mask
        #epsALL_has_stim = [False,True,False]	
        # For each epsALL segment that has stim, mark the corresponding eps segment


        for j in range(len(epsALL_has_stim)):
            if epsALL_has_stim[j] and j < len(eps) - 1:
                eps_opto_mask[eps[j]:eps[j+1]] = True

        # find the epoch after the opto 
        arr = epsALL_has_stim.copy()
        i = np.where(arr)[0][0]   # first True index

        if i == len(arr) - 1:
            ep_after_opto = np.nan
        else:
            ep_after_opto = np.zeros_like(arr, dtype=bool)
            ep_after_opto[i+1] = True


        ep_after_opto_mask = np.zeros(eps[-1], dtype=bool)
        #success_rate = np.nan
        if np.isnan(ep_after_opto).any():
            success_rate_ep_after = np.nan
        else:
            for j in range(len(ep_after_opto)):
                if ep_after_opto[j] and j < len(eps) - 1:
                    ep_after_opto_mask[eps[j]:eps[j+1]] = True
            
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[ep_after_opto_mask],rewards[ep_after_opto_mask])
            if success + fail < 10:
                success_rate_ep_after = np.nan
            else:
                success_rate_ep_after = round((success/(success+fail)*100),1)

        
        
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eps_opto_mask], rewards[eps_opto_mask])
        success_rate_opto_ep = round((success/(success+fail)*100),1)



        eps_nonopto_mask = ~eps_opto_mask    
        newrewloc = rewlocs[epsALL_has_stim]
        
        # split into opto vs. non opto
        # opto
        #trialnumvr = VR[8][0]
        #catchtrialsnum = trialnumvr[VR[16][0].astype(bool)]
        ybinned_opto_eps = ybinned[eps_opto_mask]
        trialnum_opto_eps = trialnum[eps_opto_mask]
        rewards_opto_eps = rewards[eps_opto_mask]
        dff_opto_eps = dff[eps_opto_mask]
        timedFF_opto_eps = timedFF[eps_opto_mask]
        velocity_opto_eps = velocity[eps_opto_mask]
        licks_opto_eps = licks[eps_opto_mask]
        lick_rate_opto_eps = lick_rate[eps_opto_mask]
       
 
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum_opto_eps, rewards_opto_eps)

        all_tr = sorted(str_trials + ftr_trials)
        str_bool_opto_all = np.array([(xx in ttr) and 
                 (xx%numtrialsstim==1) for xx in trialnum_opto_eps])

        # Get trial categories for each trial stream
        success_opto, fail_opto, str_trials_opto, ftr_trials_opto, ttr_opto, total_trials_opto = get_success_failure_trials(trialnum_opto_eps, rewards_opto_eps)
       

        all_tr = sorted(str_trials_opto + ftr_trials_opto)

        str_bool_opto_all = np.array([(xx in all_tr) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])

        # STR
        str_bool_opto = np.array([(xx in str_trials_opto) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])

        str_trialnum_opto = trialnum_opto_eps[str_bool_opto]

        rews_centered_opto_str = np.zeros_like(str_trialnum_opto)
        rews_centered_opto_str[(ybinned_opto_eps[str_bool_opto] >= newrewloc - 5) &
                            (ybinned_opto_eps[str_bool_opto] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto_str)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_opto_str = np.zeros_like(str_trialnum_opto)
        rews_centered_opto_str[min_iind] = 1


        normmeanrewdFF_opto_str, meanrewdFF_opto_str, normrewdFF_opto_str, rewdFF_opto_str = perireward_binned_activity(
            dff_opto_eps[str_bool_opto], rews_centered_opto_str, timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], range_val, binsize)
        _, meanvel_opto_str, __, vel_opto_str = perireward_binned_activity(
            velocity_opto_eps[str_bool_opto], rews_centered_opto_str, timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], range_val, binsize)
        _, meanlick_opto_str, __, licktr_opto_str = perireward_binned_activity(
            lick_rate_opto_eps[str_bool_opto], rews_centered_opto_str, timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], range_val, binsize)

        # FTR
        ftr_bool_opto = np.array([(xx in ftr_trials_opto) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])
        
        failed_trialnum = trialnum_opto_eps[ftr_bool_opto]
        rews_centered_opto_ftr = np.zeros_like(failed_trialnum)
        rews_centered_opto_ftr[( ybinned_opto_eps[ftr_bool_opto] >= newrewloc-5) & ( ybinned_opto_eps[ftr_bool_opto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto_ftr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_opto_ftr = np.zeros_like(failed_trialnum)
        rews_centered_opto_ftr[min_iind]=1   
        
        normmeanrewdFF_opto_ftr, meanrewdFF_opto_ftr, normrewdFF_opto_ftr, rewdFF_opto_ftr = perireward_binned_activity(
            dff_opto_eps[ftr_bool_opto], rews_centered_opto_ftr, timedFF_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], range_val, binsize)
        _, meanvel_opto_ftr, __, vel_opto_ftr = perireward_binned_activity(
            velocity_opto_eps[ftr_bool_opto], rews_centered_opto_ftr, timedFF_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], range_val, binsize)
        _, meanlick_opto_ftr, __, licktr_opto_ftr = perireward_binned_activity(
            lick_rate_opto_eps[ftr_bool_opto], rews_centered_opto_ftr, timedFF_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], range_val, binsize)

        # TTR
        ttr_bool_opto = np.array([(xx in ttr_opto) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])
        total_trialnum = trialnum_opto_eps[ttr_bool_opto]
        rews_centered_opto_ttr = np.zeros_like(total_trialnum)
        rews_centered_opto_ttr[( ybinned_opto_eps[ttr_bool_opto] >= newrewloc-5) & (ybinned_opto_eps[ttr_bool_opto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto_ttr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_opto_ttr = np.zeros_like(total_trialnum)
        rews_centered_opto_ttr[min_iind]=1        





        normmeanrewdFF_opto_ttr, meanrewdFF_opto_ttr, normrewdFF_opto_ttr, rewdFF_opto_ttr = perireward_binned_activity(
            dff_opto_eps[ttr_bool_opto], rews_centered_opto_ttr, timedFF_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto],range_val, binsize)
        _, meanvel_opto_ttr, __, vel_opto_ttr = perireward_binned_activity(
            velocity_opto_eps[ttr_bool_opto], rews_centered_opto_ttr, timedFF_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto], range_val, binsize)
        _, meanlick_opto_ttr, __, licktr_opto_ttr = perireward_binned_activity(
            lick_rate_opto_eps[ttr_bool_opto], rews_centered_opto_ttr, timedFF_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto], range_val, binsize)

    

       # --- STR TRIALS (nonopto) ---
        str_bool_nonopto = np.array([(xx in str_trials) and (xx % numtrialsstim == 0) for xx in trialnum_opto_eps])
        str_trialnum_nonopto = trialnum_opto_eps[str_bool_nonopto]

        rews_centered_nonopto_str = np.zeros_like(str_trialnum_nonopto)
        rews_centered_nonopto_str[(ybinned_opto_eps[str_bool_nonopto] >= newrewloc - 5) &
                            (ybinned_opto_eps[str_bool_nonopto] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_str)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_str = np.zeros_like(str_trialnum_nonopto)
        rews_centered_nonopto_str[min_iind] = 1

        normmeanrewdFF_nonopto_str, meanrewdFF_nonopto_str, normrewdFF_nonopto_str, rewdFF_nonopto_str = perireward_binned_activity(
            dff_opto_eps[str_bool_nonopto], rews_centered_nonopto_str, timedFF_opto_eps[str_bool_nonopto],trialnum_opto_eps[str_bool_nonopto], range_val, binsize)
        _, meanvel_nonopto_str, __, vel_nonopto_str = perireward_binned_activity(
            velocity_opto_eps[str_bool_nonopto], rews_centered_nonopto_str, timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], range_val, binsize)
        _, meanlick_nonopto_str, __, licktr_nonopto_str = perireward_binned_activity(
            lick_rate_opto_eps[str_bool_nonopto], rews_centered_nonopto_str, timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], range_val, binsize)

        # --- FTR TRIALS (nonopto) ---
        ftr_bool_nonopto = np.array([(xx in ftr_trials) and (xx % numtrialsstim == 0) for xx in trialnum_opto_eps])
        ftr_trialnum_nonopto = trialnum_opto_eps[ftr_bool_nonopto]
        rews_centered_nonopto_ftr = np.zeros_like(ftr_trialnum_nonopto)
        rews_centered_nonopto_ftr[(ybinned_opto_eps[ftr_bool_nonopto] >= newrewloc - 5) &
                            (ybinned_opto_eps[ftr_bool_nonopto] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_ftr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_ftr = np.zeros_like(ftr_trialnum_nonopto)
        rews_centered_nonopto_ftr[min_iind] = 1


        normmeanrewdFF_nonopto_ftr, meanrewdFF_nonopto_ftr, normrewdFF_nonopto_ftr, rewdFF_nonopto_ftr = perireward_binned_activity(
            dff_opto_eps[ftr_bool_nonopto], rews_centered_nonopto_ftr, timedFF_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto],range_val, binsize)
        _, meanvel_nonopto_ftr, __, vel_nonopto_ftr = perireward_binned_activity(
            velocity_opto_eps[ftr_bool_nonopto], rews_centered_nonopto_ftr, timedFF_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], range_val, binsize)
        _, meanlick_nonopto_ftr, __, licktr_nonopto_ftr = perireward_binned_activity(
            lick_rate_opto_eps[ftr_bool_nonopto], rews_centered_nonopto_ftr, timedFF_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], range_val, binsize)

        # --- TTR TRIALS (nonopto) ---
        ttr_bool_nonopto = np.array([(xx in ttr) and (xx % numtrialsstim == 0) for xx in trialnum_opto_eps])
        
        ttr_trialnum_nonopto = trialnum_opto_eps[ttr_bool_nonopto]
        rews_centered_nonopto_ttr = np.zeros_like(ttr_trialnum_nonopto)
        rews_centered_nonopto_ttr[(ybinned_opto_eps[ttr_bool_nonopto] >= newrewloc - 5) &
                            (ybinned_opto_eps[ttr_bool_nonopto] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_ttr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_ttr = np.zeros_like(ttr_trialnum_nonopto)
        rews_centered_nonopto_ttr[min_iind] = 1  
        
        normmeanrewdFF_nonopto_ttr, meanrewdFF_nonopto_ttr, normrewdFF_nonopto_ttr, rewdFF_nonopto_ttr = perireward_binned_activity(
            dff_opto_eps[ttr_bool_nonopto], rews_centered_nonopto_ttr, timedFF_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto],range_val, binsize)
        _, meanvel_nonopto_ttr, __, vel_nonopto_ttr = perireward_binned_activity(
            velocity_opto_eps[ttr_bool_nonopto],rews_centered_nonopto_ttr, timedFF_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto], range_val, binsize)
        _, meanlick_nonopto_ttr, __, licktr_nonopto_ttr = perireward_binned_activity(
            lick_rate_opto_eps[ttr_bool_nonopto], rews_centered_nonopto_ttr, timedFF_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto], range_val, binsize)

    
        # nonopto trial in the non epoches, can aligned to cs
        ybinned_nonopto_eps = ybinned[eps_nonopto_mask]
        trialnum_nonopto_eps = trialnum[eps_nonopto_mask]
        rewards_nonopto_eps = rewards[eps_nonopto_mask]
        dff_nonopto_eps = dff[eps_nonopto_mask]
        timedFF_nonopto_eps = timedFF[eps_nonopto_mask]
        lick_rate_nonopto_eps = lick_rate[eps_nonopto_mask]


        # Trial classification for non-opto trials in non-opto epochs
        success_nonopto, fail_nonopto, str_trials_nonopto, ftr_trials_nonopto, ttr_nonopto, _ = get_success_failure_trials(
            trialnum_nonopto_eps, rewards_nonopto_eps)
        
        # Success trials rew id in nonopto epoches 
        str_bool_nonopto_eps = np.array([(xx in str_trials_nonopto) for xx in trialnum_nonopto_eps])
        str_trialnum_nonopto_eps = trialnum_nonopto_eps[str_bool_nonopto_eps]
        rews_centered_nonopto_eps_str = np.zeros_like(str_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_str[(ybinned_nonopto_eps[str_bool_nonopto_eps] >= newrewloc - 5) &
                            (ybinned_nonopto_eps[str_bool_nonopto_eps] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_eps_str)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_eps_str = np.zeros_like(str_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_str[min_iind] = 1
        
        # Fail trials rew id in nonopto epoches 
        ftr_bool_nonopto_eps = np.array([(xx in ftr_trials_nonopto) for xx in trialnum_nonopto_eps])
        ftr_trialnum_nonopto_eps = trialnum_nonopto_eps[ftr_bool_nonopto_eps]
        rews_centered_nonopto_eps_ftr = np.zeros_like(ftr_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_ftr[(ybinned_nonopto_eps[ftr_bool_nonopto_eps] >= newrewloc - 5) &
                            (ybinned_nonopto_eps[ftr_bool_nonopto_eps] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_eps_ftr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_eps_ftr = np.zeros_like(ftr_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_ftr[min_iind] = 1

        # Total trials rew id in nonopto epoches 
        ttr_bool_nonopto_eps = np.array([(xx in ttr_nonopto) for xx in trialnum_nonopto_eps])
        ttr_trialnum_nonopto_eps = trialnum_nonopto_eps[ttr_bool_nonopto_eps]
        rews_centered_nonopto_eps_ttr = np.zeros_like(ttr_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_ttr[(ybinned_nonopto_eps[ttr_bool_nonopto_eps] >= newrewloc - 5) &
                            (ybinned_nonopto_eps[ttr_bool_nonopto_eps] <= newrewloc + 5)] = 1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto_eps_ttr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        rews_centered_nonopto_eps_ttr = np.zeros_like(ttr_trialnum_nonopto_eps)
        rews_centered_nonopto_eps_ttr[min_iind] = 1

        # STR
        normmeanrewdFF_nonopto_eps_str, meanrewdFF_nonopto_eps_str, normrewdFF_nonopto_eps_str, rewdFF_nonopto_eps_str = perireward_binned_activity(
            dff_nonopto_eps[str_bool_nonopto_eps], rews_centered_nonopto_eps_str, 
            timedFF_nonopto_eps[str_bool_nonopto_eps], trialnum_nonopto_eps[str_bool_nonopto_eps], range_val, binsize)

        _, meanvel_nonopto_eps_str, __, vel_nonopto_eps_str = perireward_binned_activity(
            velocity[eps_nonopto_mask][str_bool_nonopto_eps], rews_centered_nonopto_eps_str,
            timedFF[eps_nonopto_mask][str_bool_nonopto_eps], trialnum[eps_nonopto_mask][str_bool_nonopto_eps], range_val, binsize)

        _, meanlick_nonopto_eps_str, __, licktr_nonopto_eps_str = perireward_binned_activity(
            lick_rate_nonopto_eps[str_bool_nonopto_eps], rews_centered_nonopto_eps_str,
            timedFF[eps_nonopto_mask][str_bool_nonopto_eps], trialnum[eps_nonopto_mask][str_bool_nonopto_eps], range_val, binsize)

        # FTR
        normmeanrewdFF_nonopto_eps_ftr, meanrewdFF_nonopto_eps_ftr, normrewdFF_nonopto_eps_ftr, rewdFF_nonopto_eps_ftr = perireward_binned_activity(
            dff_nonopto_eps[ftr_bool_nonopto_eps],  rews_centered_nonopto_eps_ftr, 
            timedFF_nonopto_eps[ftr_bool_nonopto_eps], trialnum_nonopto_eps[ftr_bool_nonopto_eps], range_val, binsize)

        _, meanvel_nonopto_eps_ftr, __, vel_nonopto_eps_ftr = perireward_binned_activity(
            velocity[eps_nonopto_mask][ftr_bool_nonopto_eps],  rews_centered_nonopto_eps_ftr,
            timedFF[eps_nonopto_mask][ftr_bool_nonopto_eps], trialnum[eps_nonopto_mask][ftr_bool_nonopto_eps], range_val, binsize)

        _, meanlick_nonopto_eps_ftr, __, licktr_nonopto_eps_ftr = perireward_binned_activity(
            lick_rate_nonopto_eps[ftr_bool_nonopto_eps],  rews_centered_nonopto_eps_ftr,
            timedFF[eps_nonopto_mask][ftr_bool_nonopto_eps], trialnum[eps_nonopto_mask][ftr_bool_nonopto_eps], range_val, binsize)

        # TTR
        normmeanrewdFF_nonopto_eps_ttr, meanrewdFF_nonopto_eps_ttr, normrewdFF_nonopto_eps_ttr, rewdFF_nonopto_eps_ttr = perireward_binned_activity(
            dff_nonopto_eps[ttr_bool_nonopto_eps], rews_centered_nonopto_eps_ttr, 
            timedFF_nonopto_eps[ttr_bool_nonopto_eps], trialnum_nonopto_eps[ttr_bool_nonopto_eps], range_val, binsize)

        _, meanvel_nonopto_eps_ttr, __, vel_nonopto_eps_ttr = perireward_binned_activity(
            velocity[eps_nonopto_mask][ttr_bool_nonopto_eps], rews_centered_nonopto_eps_ttr,
            timedFF[eps_nonopto_mask][ttr_bool_nonopto_eps], trialnum[eps_nonopto_mask][ttr_bool_nonopto_eps], range_val, binsize)

        _, meanlick_nonopto_eps_ttr, __, licktr_nonopto_eps_ttr = perireward_binned_activity(
            lick_rate_nonopto_eps[ttr_bool_nonopto_eps], rews_centered_nonopto_eps_ttr,
            timedFF[eps_nonopto_mask][ttr_bool_nonopto_eps], trialnum[eps_nonopto_mask][ttr_bool_nonopto_eps], range_val, binsize)



        # Opto trials in Opto epoch!! This is aligned to when the opto happens 
        #stimzone = ((newrewloc*gainf-((rewsize*gainf)/2)+90)%180)/gainf
        # Define the opto-centered stim zone
        stimzone = 20 / gainf
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[(ybinned_opto_eps >= stimzone - 5) & (ybinned_opto_eps <= stimzone + 5)] = 1
        rews_iind = consecutive_stretch(np.where(opto_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx) > 0]
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[min_iind] = 1

        opto_centered = opto_centered.astype(np.uint8)
        #opto_centered = opto_centered*str_bool_opto
        # STR
        normmeanOptodFF_str, meanOptodFF_str, normOptodFF_str, optodFF_str = perireward_binned_activity(
            dff_opto_eps[str_bool_opto], opto_centered[str_bool_opto], timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto],range_val, binsize)
        _, meanlick_rate_opto_centered_opto_str, __, lick_rate_opto_centered_opto_str = perireward_binned_activity(
            lick_rate_opto_eps[str_bool_opto], opto_centered[str_bool_opto], timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto],range_val, binsize)

        #lick_rate_opto_centered_opto_str = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_opto_str, gap_len=1)

        normmeanOptodFF_ftr, meanOptodFF_ftr, normOptodFF_ftr, optodFF_ftr = perireward_binned_activity(
            dff_opto_eps[ftr_bool_opto], opto_centered[ftr_bool_opto], timedFF_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto],range_val, binsize)
        _, meanlick_rate_opto_centered_opto_ftr, __, lick_rate_opto_centered_opto_ftr = perireward_binned_activity(
            lick_rate_opto_eps[ftr_bool_opto], opto_centered[ftr_bool_opto], timedFF_opto_eps[ftr_bool_opto],trialnum_opto_eps[ftr_bool_opto], range_val, binsize)
        
        #lick_rate_opto_centered_opto_ftr = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_opto_ftr, gap_len=1)
        
        # TTR
        normmeanOptodFF_ttr, meanOptodFF_ttr, normOptodFF_ttr, optodFF_ttr = perireward_binned_activity(
            dff_opto_eps[ttr_bool_opto], opto_centered[ttr_bool_opto], timedFF_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto],range_val, binsize)
        _, meanlick_rate_opto_centered_opto_ttr, __, lick_rate_opto_centered_opto_ttr = perireward_binned_activity(
            lick_rate_opto_eps[ttr_bool_opto], opto_centered[ttr_bool_opto], timedFF_opto_eps[ttr_bool_opto],trialnum_opto_eps[ttr_bool_opto], range_val, binsize)

        #lick_rate_opto_centered_opto_ttr = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_opto_ttr, gap_len=1)  
    
    
        # Non Opto trials in Opto epoch!! This is aligned to when the opto happens 

        # STR
        normmeannonOptodFF_str, meannonOptodFF_str, normnonOptodFF_str, NonoptodFF_str = perireward_binned_activity(
            dff_opto_eps[str_bool_nonopto], opto_centered[str_bool_nonopto], timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], range_val, binsize)
        _, meanlick_rate_opto_centered_nonopto_str, __, lick_rate_opto_centered_nonopto_str = perireward_binned_activity(
            lick_rate_opto_eps[str_bool_nonopto], opto_centered[str_bool_nonopto], timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto],range_val, binsize)

        #lick_rate_opto_centered_nonopto_str = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_nonopto_str, gap_len=1)  
        # FTR
        normmeannonOptodFF_ftr, meannonOptodFF_ftr, normnonOptodFF_ftr, NonoptodFF_ftr = perireward_binned_activity(
            dff_opto_eps[ftr_bool_nonopto], opto_centered[ftr_bool_nonopto], timedFF_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], range_val, binsize)
        _, meanlick_rate_opto_centered_nonopto_ftr, __, lick_rate_opto_centered_nonopto_ftr = perireward_binned_activity(
            lick_rate_opto_eps[ftr_bool_nonopto], opto_centered[ftr_bool_nonopto], timedFF_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto],range_val, binsize)
      
        #lick_rate_opto_centered_nonopto_ftr = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_nonopto_ftr, gap_len=1)  

        # TTR
        normmeannonOptodFF_ttr, meannonOptodFF_ttr, normnonOptodFF_ttr, NonoptodFF_ttr = perireward_binned_activity(
            dff_opto_eps[ttr_bool_nonopto], opto_centered[ttr_bool_nonopto], timedFF_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto],range_val, binsize)
        _, meanlick_rate_opto_centered_nonopto_ttr, __, lick_rate_opto_centered_nonopto_ttr = perireward_binned_activity(
            lick_rate_opto_eps[ttr_bool_nonopto], opto_centered[ttr_bool_nonopto], timedFF_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto],range_val, binsize)

        #lick_rate_opto_centered_nonopto_ttr = remove_trials_with_internal_nan_gap(lick_rate_opto_centered_nonopto_ttr, gap_len=1)  

    # Construct trial category dictionaries using your variable names
        reward_summary = {
            'opto': {
                'str': {'dff': rewdFF_opto_str, 'lick': licktr_opto_str},
                'ftr': {'dff': rewdFF_opto_ftr, 'lick': licktr_opto_ftr},
                'ttr': {'dff': rewdFF_opto_ttr, 'lick': licktr_opto_ttr},
            },
            'nonopto_opto_epoch': {
                'str': {'dff': rewdFF_nonopto_str, 'lick': licktr_nonopto_str},
                'ftr': {'dff': rewdFF_nonopto_ftr, 'lick': licktr_nonopto_ftr},
                'ttr': {'dff': rewdFF_nonopto_ttr, 'lick': licktr_nonopto_ttr},
            },
            'nonopto_nonopto_epoch': {
                'str': {'dff': rewdFF_nonopto_eps_str, 'lick': licktr_nonopto_eps_str},
                'ftr': {'dff': rewdFF_nonopto_eps_ftr, 'lick': licktr_nonopto_eps_ftr},
                'ttr': {'dff': rewdFF_nonopto_eps_ttr, 'lick': licktr_nonopto_eps_ttr},
            }
        }

        opto_summary = {
            'opto': {
                'str': {'dff': optodFF_str, 'lick': lick_rate_opto_centered_opto_str},
                'ftr': {'dff': optodFF_ftr, 'lick': lick_rate_opto_centered_opto_ftr},
                'ttr': {'dff': optodFF_ttr, 'lick': lick_rate_opto_centered_opto_ttr},
            },
            'nonopto_opto_epoch': {
                'str': {'dff': NonoptodFF_str, 'lick': lick_rate_opto_centered_nonopto_str},
                'ftr': {'dff': NonoptodFF_ftr, 'lick': lick_rate_opto_centered_nonopto_ftr},
                'ttr': {'dff': NonoptodFF_ttr, 'lick': lick_rate_opto_centered_nonopto_ttr},
            }
        }


        success_summary = {'opto_ep': success_rate_opto_ep, 'ep_after_opto':success_rate_ep_after} 


        if save_npz:
            save_key = f'day_{day}'  # or f'day_{day_index}' if that's your day label

            if os.path.exists(save_path):
                # Load existing data
                existing = dict(np.load(save_path, allow_pickle=True))
            else:
                existing = {}

            # Add or update the current day
            existing[save_key] = opto_summary

            # Save all to npz
            
        save_path = os.path.join(save_dir, 'success_rate_summary.npz')
        
        save_key = f'day_{day}'
        if os.path.exists(save_path):
            # Load existing data
            existing = dict(np.load(save_path, allow_pickle=True))
        else:
            existing = {}

        # Add or update the current day
        existing[save_key] = success_summary

        # Save all to npz
        np.savez(save_path, **existing)
        
        # plot pre-first reward dop activity  
        timedFF = np.hstack(params['timedFF'])
        # plot behavior
        if pln == 0:
            fig, ax = plt.subplots(figsize=(15,6))            
            ax.plot(ybinned, zorder=1)
            ax.scatter(np.where(rewards > 0)[0], ybinned[np.where(rewards > 0)[0]], 
                    color='cyan', s=30, zorder=3)
            ax.scatter(np.where(licks > 0)[0], ybinned[np.where(licks > 0)[0]], 
                    color='k', marker='.', s=100, zorder=2)
            
            import matplotlib.patches as patches
            
            # Add gray patches for reward windows
            for ep in range(len(eps) - 1):
                ax.add_patch(
                    patches.Rectangle(
                        xy=(eps[ep], rewlocs[ep] - rewsize / 2),
                        width=len(ybinned[eps[ep]:eps[ep + 1]]),
                        height=rewsize,
                        linewidth=1,
                        color='slategray',
                        alpha=0.3
                    )
                )

            opto_stim_tr = np.zeros_like(eps_opto_mask, dtype=int)
            opto_stim_tr[eps_opto_mask] = str_bool_opto_all
            opto_stim_trs = (ybinned > 2.25) & (opto_stim_tr)
            # Add light yellow patches where result == 1
            in_patch = False
            for i in range(len(opto_stim_trs)):
                if opto_stim_trs[i] == 1 and not in_patch:
                    start = i
                    in_patch = True
                elif opto_stim_trs[i] == 0 and in_patch:
                    end = i
                    ax.axvspan(start, end, color=behavior_opto_shade, alpha=0.5, zorder=0)
                    in_patch = False
            if in_patch:  # in case the last segment goes to the end
                ax.axvspan(start, len(opto_stim_trs), color= behavior_opto_shade, alpha=0.5, zorder=0)

            ax.set_title(f'Behavior, Day {day} {day_label}')
            ax.set_ylabel('Position (cm)')
            ax.set_xticks(np.arange(0, len(timedFF) + 1000, 1000))
            ax.set_xticklabels(np.round(np.append(timedFF[::1000]/60, timedFF[-1]/60), 1))
            ax.set_xlabel('Time (minutes)')
            fig.tight_layout()
            
            if save_figs:
                fig.savefig(os.path.join(figpath, f'day{day}_{day_label}_behavior.png'))
            #pdf.savefig(fig)


            trial_types = ['str', 'ftr', 'ttr']
            titles = {
                'str': 'Success Trials (STR)',
                'ftr': 'Failure Trials (FTR)',
                'ttr': 'Total Trials (TTR)',
            }
            time_axis = np.linspace(-range_val, range_val, optodFF_str.shape[0])

            def pad_with_nan(arr, target_shape):
                padded = np.full(target_shape, np.nan)
                padded[:arr.shape[0], :arr.shape[1]] = arr
                return padded

            print("Trial types:", trial_types)

            for ttype in trial_types:
                fig = plt.figure(figsize=(9, 10))
                gs = gridspec.GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[1, 0.5, 1.2, 1.2], wspace=0.05, hspace=0.3)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1b = fig.add_subplot(gs[1, 0], sharex=ax1)  # signed difference subplot
                ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
                ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
                cax = fig.add_subplot(gs[2:, 1])  # shared colorbar

                # Get lick traces
                lick_opto = opto_summary['opto'][ttype]['lick']
                lick_nonopto = opto_summary['nonopto_opto_epoch'][ttype]['lick']

                # Match shapes by padding with NaNs
                max_timebins = max(lick_opto.shape[0], lick_nonopto.shape[0])
                max_trials = max(lick_opto.shape[1], lick_nonopto.shape[1])

                lick_opto_padded = pad_with_nan(lick_opto, (max_timebins, max_trials))
                lick_nonopto_padded = pad_with_nan(lick_nonopto, (max_timebins, max_trials))

                # Mean ± SEM
                mean_opto = np.nanmean(lick_opto, axis=1)
                sem_opto = scipy.stats.sem(lick_opto, axis=1, nan_policy='omit')
                mean_nonopto = np.nanmean(lick_nonopto, axis=1)
                sem_nonopto = scipy.stats.sem(lick_nonopto, axis=1, nan_policy='omit')

                # --- ax1: Mean traces
                ax1.plot(time_axis, mean_opto, label='Opto Trials (Opto Epoch)', color='royalblue')
                ax1.fill_between(time_axis, mean_opto - sem_opto, mean_opto + sem_opto, alpha=0.3, color='royalblue')
                ax1.plot(time_axis, mean_nonopto, label='Non-Opto Trials (Opto Epoch)', color='darkorange')
                ax1.fill_between(time_axis, mean_nonopto - sem_nonopto, mean_nonopto + sem_nonopto, alpha=0.3, color='darkorange')
                ax1.axvline(0, linestyle='--', color='gray')
                ax1.axvspan(0, 2, color=opto_shade, alpha=0.3, label='Opto Period')
                ax1.set_ylabel('Lick Rate')
                ax1.set_title(f'{day} {day_label} | Opto-Aligned Lick Rate\n{titles[ttype]}')
                ax1.legend()
                ax1.tick_params(labelbottom=True)

                # --- ax1b: Signed difference
                signed_diff = mean_opto - mean_nonopto
                ax1b.plot(time_axis, signed_diff, color='black')
                ax1b.axhline(0, linestyle='--', color='gray', linewidth=1)
                ax1b.set_ylabel('Δ Lick Rate\n(Signed)')
                ax1b.spines[['top', 'right']].set_visible(False)

                # Shared color scale
                combined_lick = np.vstack([lick_opto_padded, lick_nonopto_padded])
                vmin = np.nanmin(combined_lick)
                vmax = np.nanmax(combined_lick)

                # --- ax2: Heatmap Opto
                im1 = ax2.imshow(lick_opto_padded.T, aspect='auto', cmap='viridis',
                                extent=[time_axis[0], time_axis[-1], 0, max_trials],
                                vmin=vmin, vmax=vmax)
                ax2.set_ylabel('Trial #')
                ax2.set_title('Lick Heatmap: Opto Trials')
                ax2.axvline(0, linestyle='--', color='gray')
                ax2.invert_yaxis()
                # Get current y-ticks
                yticks = ax2.get_yticks()

                # Reverse the order of labels
                ax2.set_yticklabels(yticks[::-1])


                # --- ax3: Heatmap Non-Opto
                im2 = ax3.imshow(lick_nonopto_padded.T, aspect='auto', cmap='viridis',
                                extent=[time_axis[0], time_axis[-1], 0, max_trials],
                                vmin=vmin, vmax=vmax)
                ax3.set_ylabel('Trial #')
                ax3.set_xlabel('Time (s)')
                ax3.set_title('Lick Heatmap: Non-Opto Trials')
                ax3.axvline(0, linestyle='--', color='gray')
                ax3.invert_yaxis()
                yticks = ax3.get_yticks()

                # Reverse the order of labels
                ax3.set_yticklabels(yticks[::-1])

                # --- Shared colorbar
                cbar = fig.colorbar(im2, cax=cax)
                cbar.set_label('Lick Rate')

                # --- Aesthetics
                for ax in [ax1, ax1b, ax2, ax3]:
                    ax.tick_params(direction='out')
                    ax.spines[['top', 'right']].set_visible(False)

                print(f"Plotting for trial type: {ttype}")
                if save_figs:
                    plt.savefig(os.path.join(save_dir, f"D{days[0]}_Opto_aligned_lickrate_{ttype}.png"), dpi=300)
                plt.show()
                plt.close()
# Define save paths

#lick_save_path = "lickrate_opto_aligned_all_days.npz"



'''
        for summary, alignment_type in zip([reward_summary, opto_summary], ['Reward-Aligned', 'Opto-Aligned']):
            for cond, trialtypes in summary.items():
                for ttype, data in trialtypes.items():
                    dff = data['dff']
                    lick = data['lick']

                    if dff.size == 0 or lick.size == 0:
                        continue

                    num_bins = dff.shape[0]
                    num_trials = dff.shape[1]
                    time_axis = np.linspace(-range_val, range_val, num_bins)

                    panel_name = planelut.get(pln, f'Panel {pln}')

                    # === Line Plots ===
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

                    ax[0].plot(time_axis, np.nanmean(dff, axis=1), label='dF/F', color='blue')
                    ax[0].fill_between(time_axis,
                                    np.nanmean(dff, axis=1) - scipy.stats.sem(dff, axis=1, nan_policy='omit'),
                                    np.nanmean(dff, axis=1) + scipy.stats.sem(dff, axis=1, nan_policy='omit'),
                                    alpha=0.3, color='blue')
                    ax[0].axvline(0, color='k', linestyle='--')
                    ax[0].set_title(f'{panel_name} | {day} | {cond.upper()}-{ttype.upper()} | {alignment_type} dF/F')
                    ax[0].set_xlabel('Time (s)')
                    ax[0].set_ylabel('dF/F')
                    ax[0].legend()
                    ax[0].spines[['top', 'right']].set_visible(False)

                    ax[1].plot(time_axis, np.nanmean(lick, axis=1), label='Lick Rate', color='gray')
                    ax[1].fill_between(time_axis,
                                    np.nanmean(lick, axis=1) - scipy.stats.sem(lick, axis=1, nan_policy='omit'),
                                    np.nanmean(lick, axis=1) + scipy.stats.sem(lick, axis=1, nan_policy='omit'),
                                    alpha=0.3, color='gray')
                    ax[1].axvline(0, color='k', linestyle='--')
                    ax[1].set_title(f'{panel_name} | {day} | {cond.upper()}-{ttype.upper()} | {alignment_type} Lick Rate')
                    ax[1].set_xlabel('Time (s)')
                    ax[1].set_ylabel('Lick Rate')
                    ax[1].legend()
                    ax[1].spines[['top', 'right']].set_visible(False)

                    fig.tight_layout()
                    plt.show()

                    # === Heatmaps ===
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                    im0 = ax[0].imshow(dff.T, aspect='auto', cmap='viridis',
                                    extent=[time_axis[0], time_axis[-1], 0, num_trials])
                    ax[0].set_title(f'{panel_name} | {day} | {cond.upper()}-{ttype.upper()} | {alignment_type} dF/F Heatmap')
                    ax[0].set_xlabel('Time (s)')
                    ax[0].set_ylabel('Trial #')
                    ax[0].axvline(0, color='w', linestyle='--')
                    plt.colorbar(im0, ax=ax[0], label='dF/F')

                    im1 = ax[1].imshow(lick.T, aspect='auto', cmap='gray',
                                    extent=[time_axis[0], time_axis[-1], 0, num_trials])
                    ax[1].set_title(f'{panel_name} | {day} | {cond.upper()}-{ttype.upper()} | {alignment_type} Lick Heatmap')
                    ax[1].set_xlabel('Time (s)')
                    ax[1].set_ylabel('Trial #')
                    ax[1].axvline(0, color='r', linestyle='--')
                    plt.colorbar(im1, ax=ax[1], label='Lick Rate')

                    fig.tight_layout()
                    plt.show()

'''