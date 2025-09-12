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

from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
from projects.opto.behavior.behavior import smooth_lick_rate, get_lick_selectivity
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
from Ziyi.python_processing_code.functions.function_code import spatial_binned_activity,nan_after_reward_all_epochs
from projects.DLC_behavior_classification import eye
plt.rcParams["font.family"] = "Arial"

#%%
#plt.close('all')
# save to pdf
animal = ''
src = r"E:\Ziyi\Data\VTA_mice\hrz\E277"
src = os.path.join(src,animal)
dst = r"E:\Ziyi"
save_dir = r'C:\Users\HanLab\Downloads\figure_1'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"hrz_{os.path.basename(src)}.pdf"))
#days = np.arange(19,23)
# opto days [19,20,22,23]
# control days [21,24]
save_figs = False
days = [37]
#range_val=20; binsize=0.2
numtrialsstim=2 # every 10 trials stim w 1 trial off
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
# figs = True # print out per day figs

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

        dt = np.nanmedian(np.diff(timedFF))
        lick_rate = smooth_lick_rate(licks, dt, sigma_sec=0.7)
        lick_rate = nan_after_reward_all_epochs(lick_rate, rewards, trialnum, eps)
        
        #eps_opto_mask = np.zeros(len(stims), dtype=bool)


        changeRewLocALL = np.hstack(params['changeRewLocALL'])
        epsALL = np.where(changeRewLocALL>0)[0];rewlocs = changeRewLocALL[epsALL]/gainf
        epsALL = np.append(epsALL, len(changeRewLocALL))

        # Initialize eps_opto_mask
        eps_opto_mask = np.zeros(eps[-1], dtype=bool)

        # Precompute which epsALL segments have stim
        
        epsALL_has_stim = np.array([
            np.any(stims[epsALL[j]:epsALL[j+1]] == 1)
            for j in range(len(epsALL) - 1)
        ])
        
        epsALL_has_stim = [True, False, False]
        # For each epsALL segment that has stim, mark the corresponding eps segment
        for j in range(len(epsALL_has_stim)):
            if epsALL_has_stim[j] and j < len(eps) - 1:
                eps_opto_mask[eps[j]:eps[j+1]] = True


        eps_nonopto_mask = ~eps_opto_mask

        #for ii in range(len(eps)-1):
        #    epoch_mask = np.arange(eps[ii],eps[ii+1])
        
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

        normmeanrewdFF_opto_str, meanrewdFF_opto_str, normrewdFF_opto_str, rewdFF_opto_str = spatial_binned_activity(
            dff_opto_eps[str_bool_opto], ybinned_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], binsize=9, 
            track_length=270)
        _, meanvel_opto_str, __, vel_opto_str = spatial_binned_activity(
            velocity_opto_eps[str_bool_opto], ybinned_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], binsize=9, track_length=270)
        _, meanlick_opto_str, __, licktr_opto_str = spatial_binned_activity(
            licks_opto_eps[str_bool_opto], ybinned_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto], binsize=9, track_length=270)

        # FTR
        ftr_bool_opto = np.array([(xx in ftr_trials_opto) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])
        
        failed_trialnum = trialnum_opto_eps[ftr_bool_opto]
        rews_centered_opto_ftr = np.zeros_like(failed_trialnum)
        rews_centered_opto_ftr[( ybinned_opto_eps[ftr_bool_opto] >= newrewloc-5) & ( ybinned_opto_eps[ftr_bool_opto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto_ftr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_opto_ftr = np.zeros_like(failed_trialnum)
        rews_centered_opto_ftr[min_iind]=1   
        
        normmeanrewdFF_opto_ftr, meanrewdFF_opto_ftr, normrewdFF_opto_ftr, rewdFF_opto_ftr = spatial_binned_activity(
            dff_opto_eps[ftr_bool_opto], ybinned_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], binsize=9, 
            track_length=270)
        _, meanvel_opto_ftr, __, vel_opto_ftr = spatial_binned_activity(
        velocity_opto_eps[ftr_bool_opto], ybinned_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], binsize=9, 
        track_length=270)
        _, meanlick_opto_ftr, __, licktr_opto_ftr = spatial_binned_activity(
            licks_opto_eps[ftr_bool_opto], ybinned_opto_eps[ftr_bool_opto], trialnum_opto_eps[ftr_bool_opto], binsize=9, 
            track_length=270)

        # TTR
        ttr_bool_opto = np.array([(xx in ttr_opto) and (xx % numtrialsstim == 1) for xx in trialnum_opto_eps])
        total_trialnum = trialnum_opto_eps[ttr_bool_opto]
        rews_centered_opto_ttr = np.zeros_like(total_trialnum)
        rews_centered_opto_ttr[( ybinned_opto_eps[ttr_bool_opto] >= newrewloc-5) & (ybinned_opto_eps[ttr_bool_opto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto_ttr)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_opto_ttr = np.zeros_like(total_trialnum)
        rews_centered_opto_ttr[min_iind]=1        


        normmeanrewdFF_opto_ttr, meanrewdFF_opto_ttr, normrewdFF_opto_ttr, rewdFF_opto_ttr = spatial_binned_activity(
            dff_opto_eps[ttr_bool_opto], ybinned_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto], binsize=9, 
            track_length=270)
        _, meanvel_opto_ttr, __, vel_opto_ttr =spatial_binned_activity(
        velocity_opto_eps[ttr_bool_opto], ybinned_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto], binsize=9, 
        track_length=270)
        _, meanlick_opto_ttr, __, licktr_opto_ttr = spatial_binned_activity(
            licks_opto_eps[ttr_bool_opto], ybinned_opto_eps[ttr_bool_opto], trialnum_opto_eps[ttr_bool_opto], binsize=9, 
            track_length=270)

    

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

        normmeanrewdFF_nonopto_str, meanrewdFF_nonopto_str, normrewdFF_nonopto_str, rewdFF_nonopto_str = spatial_binned_activity(
            dff_opto_eps[str_bool_nonopto], ybinned_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_str, __, vel_nonopto_str = spatial_binned_activity(
            velocity_opto_eps[str_bool_nonopto], ybinned_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_str, __, licktr_nonopto_str = spatial_binned_activity(
            licks_opto_eps[str_bool_nonopto], ybinned_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto], binsize=9, 
            track_length=270)
        
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


        normmeanrewdFF_nonopto_ftr, meanrewdFF_nonopto_ftr, normrewdFF_nonopto_ftr, rewdFF_nonopto_ftr = spatial_binned_activity(
            dff_opto_eps[ftr_bool_nonopto], ybinned_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_ftr, __, vel_nonopto_ftr = spatial_binned_activity(
            velocity_opto_eps[ftr_bool_nonopto], ybinned_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_ftr, __, licktr_nonopto_ftr = spatial_binned_activity(
            licks_opto_eps[ftr_bool_nonopto], ybinned_opto_eps[ftr_bool_nonopto], trialnum_opto_eps[ftr_bool_nonopto], binsize=9, 
            track_length=270)

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
        
        normmeanrewdFF_nonopto_ttr, meanrewdFF_nonopto_ttr, normrewdFF_nonopto_ttr, rewdFF_nonopto_ttr = spatial_binned_activity(
            dff_opto_eps[ttr_bool_nonopto], ybinned_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_ttr, __, vel_nonopto_ttr = spatial_binned_activity(
            velocity_opto_eps[ttr_bool_nonopto], ybinned_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_ttr, __, licktr_nonopto_ttr = spatial_binned_activity(
            licks_opto_eps[ttr_bool_nonopto], ybinned_opto_eps[ttr_bool_nonopto], trialnum_opto_eps[ttr_bool_nonopto], binsize=9, 
            track_length=270)

    
        # nonopto trial in the non epoches, can aligned to cs
        ybinned_nonopto_eps = ybinned[eps_nonopto_mask]
        trialnum_nonopto_eps = trialnum[eps_nonopto_mask]
        rewards_nonopto_eps = rewards[eps_nonopto_mask]
        dff_nonopto_eps = dff[eps_nonopto_mask]
        timedFF_nonopto_eps = timedFF[eps_nonopto_mask]
        lick_rate_nonopto_eps = lick_rate[eps_nonopto_mask]
        velocity_nonopto_eps = velocity[eps_nonopto_mask]
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
        normmeanrewdFF_nonopto_eps_str, meanrewdFF_nonopto_eps_str, normrewdFF_nonopto_eps_str, rewdFF_nonopto_eps_str = spatial_binned_activity(
            dff_nonopto_eps[str_bool_nonopto_eps], ybinned_nonopto_eps[str_bool_nonopto_eps], trialnum_nonopto_eps[str_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_eps_str, __, vel_nonopto_eps_str = spatial_binned_activity(
            velocity_nonopto_eps[str_bool_nonopto_eps], ybinned_nonopto_eps[str_bool_nonopto_eps], trialnum_nonopto_eps[str_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_eps_str, __, licktr_nonopto_eps_str = spatial_binned_activity(
            lick_rate_nonopto_eps[str_bool_nonopto_eps], ybinned_nonopto_eps[str_bool_nonopto_eps], trialnum_nonopto_eps[str_bool_nonopto_eps], binsize=9, 
            track_length=270)

        # FTR
        normmeanrewdFF_nonopto_eps_ftr, meanrewdFF_nonopto_eps_ftr, normrewdFF_nonopto_eps_ftr, rewdFF_nonopto_eps_ftr =  spatial_binned_activity(
            dff_nonopto_eps[ftr_bool_nonopto_eps], ybinned_nonopto_eps[ftr_bool_nonopto_eps], trialnum_nonopto_eps[ftr_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_eps_ftr, __, vel_nonopto_eps_ftr = spatial_binned_activity(
            velocity_nonopto_eps[ftr_bool_nonopto_eps], ybinned_nonopto_eps[ftr_bool_nonopto_eps], trialnum_nonopto_eps[ftr_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_eps_ftr, __, licktr_nonopto_eps_ftr = spatial_binned_activity(
            lick_rate_nonopto_eps[ftr_bool_nonopto_eps], ybinned_nonopto_eps[ftr_bool_nonopto_eps], trialnum_nonopto_eps[ftr_bool_nonopto_eps], binsize=9, 
            track_length=270)
 
        # TTR
        normmeanrewdFF_nonopto_eps_ttr, meanrewdFF_nonopto_eps_ttr, normrewdFF_nonopto_eps_ttr, rewdFF_nonopto_eps_ttr = spatial_binned_activity(
            dff_nonopto_eps[ttr_bool_nonopto_eps], ybinned_nonopto_eps[ttr_bool_nonopto_eps], trialnum_nonopto_eps[ttr_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanvel_nonopto_eps_ttr, __, vel_nonopto_eps_ttr = spatial_binned_activity(
            velocity_nonopto_eps[ttr_bool_nonopto_eps], ybinned_nonopto_eps[ttr_bool_nonopto_eps], trialnum_nonopto_eps[ttr_bool_nonopto_eps], binsize=9, 
            track_length=270)
        _, meanlick_nonopto_eps_ttr, __, licktr_nonopto_eps_ttr = spatial_binned_activity(
            lick_rate_nonopto_eps[ttr_bool_nonopto_eps], ybinned_nonopto_eps[ttr_bool_nonopto_eps], trialnum_nonopto_eps[ttr_bool_nonopto_eps], binsize=9, 
            track_length=270)




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
                    ax.axvspan(start, end, color='pink', alpha=0.5, zorder=0)
                    in_patch = False
            if in_patch:  # in case the last segment goes to the end
                ax.axvspan(start, len(opto_stim_trs), color='pink', alpha=0.5, zorder=0)

            ax.set_title(f'Behavior, Day {day}')
            ax.set_ylabel('Position (cm)')
            ax.set_xticks(np.arange(0, len(timedFF) + 1000, 1000))
            ax.set_xticklabels(np.round(np.append(timedFF[::1000]/60, timedFF[-1]/60), 1))
            ax.set_xlabel('Time (minutes)')
            fig.tight_layout()

            if save_figs:
                fig.savefig(os.path.join(save_dir, f"behavior_day_{day}.png"))
            plt.show()
            
        # === Step 1: Build reward_summary using existing aligned variables ===
        position_binned_data = {
            'opto': {
                'str': {
                    'dff': rewdFF_opto_str,
                    'lick': licktr_opto_str,
                    'velocity': vel_opto_str
                },
                'ftr': {
                    'dff': rewdFF_opto_ftr,
                    'lick': licktr_opto_ftr,
                    'velocity': vel_opto_ftr
                },
                'ttr': {
                    'dff': rewdFF_opto_ttr,
                    'lick': licktr_opto_ttr,
                    'velocity': vel_opto_ttr
                }
            },
            'nonopto_opto_epoch': {
                'str': {
                    'dff': rewdFF_nonopto_str,
                    'lick': licktr_nonopto_str,
                    'velocity': vel_nonopto_str
                },
                'ftr': {
                    'dff': rewdFF_nonopto_ftr,
                    'lick': licktr_nonopto_ftr,
                    'velocity': vel_nonopto_ftr
                },
                'ttr': {
                    'dff': rewdFF_nonopto_ttr,
                    'lick': licktr_nonopto_ttr,
                    'velocity': vel_nonopto_ttr
                }
            },
            'nonopto_nonopto_epoch': {
                'str': {
                    'dff': rewdFF_nonopto_eps_str,
                    'lick': licktr_nonopto_eps_str,
                    'velocity': vel_nonopto_eps_str
                },
                'ftr': {
                    'dff': rewdFF_nonopto_eps_ftr,
                    'lick': licktr_nonopto_eps_ftr,
                    'velocity': vel_nonopto_eps_ftr
                },
                'ttr': {
                    'dff': rewdFF_nonopto_eps_ttr,
                    'lick': licktr_nonopto_eps_ttr,
                    'velocity': vel_nonopto_eps_ttr
                }
            }
        }


        
        '''
        summary_path = "position_binned_summary_all_days.npz"

        # Load existing file if it exists
        if os.path.exists(summary_path):
            existing_data = dict(np.load(summary_path, allow_pickle=True))
        else:
            existing_data = {}

        # Update with current day
        existing_data[f'day_{day}'] = position_binned_data

        # Save back to npz file
        np.savez_compressed(summary_path, **existing_data)
        
        '''
        groups = ['opto', 'nonopto_opto_epoch', 'nonopto_nonopto_epoch']
        trial_types = ['str', 'ftr', 'ttr']
        signals = ['dff', 'lick']
        comparison_pairs = [
            ('opto', 'nonopto_opto_epoch'),
            ('opto', 'nonopto_nonopto_epoch'),
            ('nonopto_opto_epoch', 'nonopto_nonopto_epoch')
        ]
        


        # Bin setup
        n_bins = 30
        xticks = np.linspace(0, n_bins - 1, 10)
        xtick_labels = np.linspace(0, 270, 10, dtype=int)

        # 1. Plot full figure (heatmap + mean ± SEM) for each group and signal
        for group in groups:
            for signal in signals:
                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f'{signal.upper()} Summary - {group} (Day {day})', fontsize=16)

                gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[2, 1])

                for i, trial_type in enumerate(trial_types):
                    data = position_binned_data[group][trial_type][signal]
                    mean = np.nanmean(data, axis=1)
                    sem = np.nanstd(data, axis=1) / np.sqrt(np.sum(~np.isnan(data), axis=1))

                    # Heatmap
                    ax0 = fig.add_subplot(gs[i, 0])
                    im = ax0.imshow(data.T, aspect='auto', cmap='viridis', origin='lower')
                    ax0.set_title(f'{trial_type.upper()} Heatmap')
                    ax0.set_ylabel('Trial #')
                    ax0.set_xticks(xticks)
                    ax0.set_xticklabels(xtick_labels)
                    if i == 2:
                        ax0.set_xlabel('Position (cm)')
                    fig.colorbar(im, ax=ax0, fraction=0.015)

                    # Mean ± SEM
                    ax1 = fig.add_subplot(gs[i, 1])
                    ax1.plot(mean, label='Mean')
                    ax1.fill_between(np.arange(len(mean)), mean - sem, mean + sem, alpha=0.3)
                    ax1.set_title(f'{trial_type.upper()} Mean ± SEM')
                    ax1.set_ylabel(signal.upper())
                    ax1.set_xticks(xticks)
                    ax1.set_xticklabels(xtick_labels)
                    if i == 2:
                        ax1.set_xlabel('Position (cm)')

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{signal}_{group}_Day{day}_summary.png')
                if save_figs:
                    plt.savefig(save_path, dpi=300)
                plt.show()
                plt.close()

        # 2. Plot comparison figures with STR, FTR, TTR as subplots
        for signal in signals:
            for group_a, group_b in comparison_pairs:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
                fig.suptitle(f'{signal.upper()} Comparison: {group_a} vs {group_b} (Day {day})', fontsize=16)

                for i, trial_type in enumerate(trial_types):
                    ax = axes[i]

                    for group, color, label in zip([group_a, group_b],
                                                ['tab:blue', 'tab:orange'],
                                                [group_a, group_b]):
                        data = position_binned_data[group][trial_type][signal]
                        mean = np.nanmean(data, axis=1)
                        sem = np.nanstd(data, axis=1) / np.sqrt(np.sum(~np.isnan(data), axis=1))

                        ax.plot(mean, label=label)
                        ax.fill_between(np.arange(len(mean)), mean - sem, mean + sem, alpha=0.3)

                    ax.set_title(trial_type.upper())
                    ax.set_xlabel('Position (cm)')
                    if i == 0:
                        ax.set_ylabel(signal.upper())
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels)
                    ax.legend()
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{signal}_{group_a}_vs_{group_b}_Day{day}_comparison.png')
                if save_figs:
                    plt.savefig(save_path, dpi=300)
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