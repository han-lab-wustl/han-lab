import numpy as np

import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_lick_selectivity(ypos, trialnum, lick, rewloc, rewsize,
                fails_only = False):
    """Assume Y is the position, which is a one-dimensional vector. L is the binary licking behavior, 
    which is also a one-dimensional vector. The start position of reward 
    zone is “reward location – ½ * reward zone size – 1”, which is a scalar, P.
    Licking number in the last quarter is 
    “L[np.where((Y/P < 1) & (Y/P > 0.75))[0]].sum()”. The total 
    licking number in the pre-reward zone 
    is “L[np.where((Y/P < 1) & (Y > 3.0))[0]].sum()”, 
    where I remove all the dark time licking.
    Licks in the last quarter / total pre-reward licks is what I define of licking accuracy.
    """
    lick_selectivity_per_trial = []
    for trial in np.unique(trialnum):
        ypos_t = ypos[trialnum==trial]
        lick_t = lick[trialnum==trial]
        start_postion = rewloc-(.5*rewsize)
        last_quarter = lick_t[np.where((ypos_t/start_postion < 1) & (ypos_t/start_postion > 0.75))[0]].sum()
        pre_rew_licks = lick_t[np.where((ypos_t/start_postion < 1) & (ypos_t > 3))[0]].sum()
        total_licks = lick_t[np.where((ypos_t/start_postion < 1) & (ypos_t > 3))[0]].sum()
        pre_n_rew_licks = lick_t[np.where((ypos_t/start_postion < 1) & (ypos_t<rewloc+(.5*rewsize)+1) & (ypos_t > 3))[0]].sum()
        in_rew_zone = lick_t[np.where((ypos_t>start_postion) & (ypos_t<(rewloc+(.5*rewsize))))[0]].sum()
        # if fails_only==True:
            # print(f'Pre-reward licks: {pre_rew_licks}, in reward zone licks {in_rew_zone}')
        # lick_selectivity = last_quarter/total_licks 
        if in_rew_zone==0 or fails_only==False:# or fails_only: # done to avoid those instances when animal seems
            # to lick just before or at start of reward zone (according to vr)
            lick_selectivity = last_quarter/total_licks 
        elif pre_rew_licks>0 and in_rew_zone>0: 
            lick_selectivity = 1+last_quarter/total_licks 
        elif pre_rew_licks==0 and in_rew_zone>0: # if the mouse only licks in rew zone
            lick_selectivity = 3
        
        lick_selectivity_per_trial.append(lick_selectivity)
        
    
    return lick_selectivity_per_trial

def get_behavior_tuning_curve(ybinned, beh, bins=270):
    """
    Plot a lick tuning curve given a dataframe with position and lick columns.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - position_col: name of the column in df that contains the position data.
    - lick_col: name of the column in df that contains the lick binary variable (1 for lick, 0 for no lick).
    - bins: number of bins to divide the position data into for the curve.
    """
    df = pd.DataFrame()
    df['position'] = ybinned
    df['beh'] = beh
    # Discretize the position data into bins
    df['position_bin'] = pd.cut(df['position'], bins=bins, labels=False)
    
    # Calculate the lick probability for each bin
    grouped = df.groupby('position_bin')['beh'].agg(['mean', 'count']).reset_index()
    beh_probability = grouped['mean']  # This is the mean of the binary lick variable, which represents probability
    
    return grouped['position_bin'], beh_probability

def get_rewzones(rewlocs, gainf):
    # Initialize the reward zone numbers array with zeros
    rewzonenum = np.zeros(len(rewlocs))
    
    # Iterate over each reward location to determine its reward zone
    for kk, loc in enumerate(rewlocs):
        if loc <= 86 * gainf:
            rewzonenum[kk] = 1  # Reward zone 1
        elif 101 * gainf <= loc <= 120 * gainf:
            rewzonenum[kk] = 2  # Reward zone 2
        elif loc >= 135 * gainf:
            rewzonenum[kk] = 3  # Reward zone 3
            
    return rewzonenum
    
    return trials_before_success

def get_mean_velocity_per_ep(forwardvel):
    return np.nanmean(forwardvel)

def get_performance(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, rewsize):
    # opto ep    
    eptotest = opto_ep-1 # matlab index (+1)
    eprng = range(eps[eptotest], eps[eptotest+1])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    forwardvel_ = forwardvel[eprng]
    rewloc = np.ceil(rewlocs[eptotest]).astype(int)
    rewsize = 10 # temp
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_opto = success / total_trials
    trials_bwn_success_opto =  np.diff(np.array(strials))
    # get lick prob
    pos_bin_opto, lick_probability_opto = get_behavior_tuning_curve(ybinned_, licks_)
    # lick selectivity - only success
    lasttr = 5
    mask = np.array([xx in strials[-lasttr:] for xx in trialnum_])
    # optional - fails
    # mask = np.array([xx in ftrials for xx in trialnum_])
    lick_selectivity_per_trial_opto = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], rewloc, rewsize,
                fails_only = False)
    # split into pre, rew, and post
    lick_prob_opto = [lick_probability_opto[:int(rewloc-rewsize)], lick_probability_opto[int(rewloc-rewsize-10):int(rewloc+20)], \
                    lick_probability_opto[int(rewloc+20):]]
    vel_opto = get_mean_velocity_per_ep(forwardvel_[ybinned_<rewloc]) # pre-reward
    # previous ep
    eprng = range(eps[eptotest-1], eps[eptotest])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    forwardvel_ = forwardvel[eprng]
    rewloc = np.ceil(rewlocs[eptotest-1]).astype(int)
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_prev = success / total_trials 
    trials_bwn_success_prev =  np.diff(np.array(strials))
    # lick prob
    pos_bin_prev, lick_probability_prev = get_behavior_tuning_curve(ybinned_, licks_)
    # lick selectivity
    mask = np.array([xx in strials[-lasttr:] for xx in trialnum_])
    # optional - fails
    # mask = np.array([xx in ftrials for xx in trialnum_])
    lick_selectivity_per_trial_prev = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], rewloc, rewsize,
                fails_only = False)
    # split into pre, rew, and post
    lick_prob_prev = [lick_probability_prev[:int(rewloc-rewsize-20)], 
                    lick_probability_prev[int(rewloc-rewsize-20):int(rewloc+20)], \
                    lick_probability_prev[int(rewloc+20):]]
    # Return a dictionary or multiple dictionaries containing your results
    vel_prev = get_mean_velocity_per_ep(forwardvel_[ybinned_<rewloc])
    return rate_opto, rate_prev, lick_prob_opto, \
    lick_prob_prev, trials_bwn_success_opto, trials_bwn_success_prev, \
    vel_opto, vel_prev, lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev


def get_success_failure_trials(trialnum, reward):
    """
    Counts the number of success and failure trials.

    Parameters:
    trialnum : array-like, list of trial numbers
    reward : array-like, list indicating whether a reward was found (1) or not (0) for each trial

    Returns:
    success : int, number of successful trials
    fail : int, number of failed trials
    str : list, successful trial numbers
    ftr : list, failed trial numbers
    ttr : list, trial numbers excluding probes
    total_trials : int, total number of trials excluding probes
    """
    trialnum = np.array(trialnum)
    reward = np.array(reward)
    unique_trials = np.unique(trialnum)
    
    success = 0
    fail = 0
    str_trials = []  # success trials
    ftr_trials = []  # failure trials

    for trial in unique_trials:
        if trial >= 3:  # Exclude probe trials
            trial_indices = trialnum == trial
            if np.any(reward[trial_indices] == 1):
                success += 1
                str_trials.append(trial)
            else:
                fail += 1
                ftr_trials.append(trial)
    
    total_trials = np.sum(unique_trials >= 3)
    ttr = unique_trials[unique_trials > 2]  # trials excluding probes

    return success, fail, str_trials, ftr_trials, ttr, total_trials