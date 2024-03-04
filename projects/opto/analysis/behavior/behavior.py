import numpy as np

import numpy as np

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

def get_performance(opto_ep, epind, eps, trialnum, rewards, licks, \
    ybinned, rewlocs):
    # opto ep    
    eptotest = opto_ep-1 # matlab index (+1)
    eprng = range(eps[eptotest], eps[eptotest+1])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    rewloc = rewlocs[eptotest]    
    # Simulate the get_success_failure_trials, get_lick_selectivity, and get_com_licks functionality
    success, fail, str, ftr, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_opto = success / total_trials
    # previous ep
    eprng = range(eps[eptotest-1], eps[eptotest])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    rewloc = rewlocs[eptotest]
    success, fail, str, ftr, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_prev = success / total_trials

    # Return a dictionary or multiple dictionaries containing your results
    return rate_opto, rate_prev



def get_success_failure_trials(trialnum, reward):
    """
    Convert MATLAB function to Python.
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