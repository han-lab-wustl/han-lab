import os, shutil, glob, numpy as np, sys
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab')
from projects.DLC_behavior_classification.eye import consecutive_stretch_vralign, get_area_circumference_opto, perireward_binned_activity
from projects.DLC_behavior_classification import eye

def copyvrfl_matching_pickle(picklesrc, vrsrc):
    for fl in os.listdir(picklesrc):
        if fl[-2:]=='.p':
            picklefl = os.path.join(picklesrc,fl)
            vrfl = glob.glob(picklefl[:-15]+'*.mat')
            if len(vrfl)==0:
                vrflsearch = fl[:-15]+'*.mat'
                vrflsr = glob.glob(os.path.join(vrsrc, vrflsearch))[0]
                shutil.copy(vrflsr, picklesrc)
    print('\n ************ done copying vr files! ************')

def get_success_failure_trials(trialnum, reward):
    """
    Quantify successful and failed trials based on trial numbers and rewards.
    Also, identify:
    - failed trials after successful trials,
    - failed trials following one or many failed trials,
    - successful trials after exactly one successful trial,
    - successful trials after exactly two consecutive successful trials.

    Args:
        trialnum (numpy.ndarray): Array of trial numbers.
        reward (numpy.ndarray): Array of rewards (0 or 1) corresponding to each trial.

    Returns:
        int: Number of successful trials.
        int: Number of failed trials.
        list: List of successful trial numbers.
        list: List of failed trial numbers.
        numpy.ndarray: Array of trial numbers, excluding probe trials (trial < 3).
        int: Total number of trials, excluding probe trials.
        list: List of failed trial numbers that occur right after a successful trial.
        list: List of failed trial numbers that occur after one or more consecutive failed trials.
        list: List of successful trial numbers after exactly one successful trial.
        list: List of successful trial numbers after exactly two consecutive successful trials.
    """
    success = 0
    fail = 0
    str_trials = []
    ftr_trials = []
    fail_after_success = []
    fail_after_fail = []
    succ_after_one_succ = []
    succ_after_two_succ = []

    valid_trials = np.unique(trialnum)[np.unique(trialnum) >= 3]  # Exclude probe trials
    ttr = valid_trials.copy()
    total_trials = len(ttr)

    # Placeholder to remember the reward status of the last and second last valid trial
    last_reward = -1
    second_last_reward = -1

    for trial in valid_trials:
        current_reward = np.sum(reward[trialnum == trial] == 1) > 0

        if current_reward:
            success += 1
            str_trials.append(trial)
            # Check for success after one and two successful trials
            if last_reward == 1 and second_last_reward != 1:
                succ_after_one_succ.append(trial)
            elif last_reward == 1 and second_last_reward == 1:
                succ_after_two_succ.append(trial)
            
            second_last_reward = last_reward
            last_reward = 1
        else:
            fail += 1
            ftr_trials.append(trial)
            if last_reward == 1:
                fail_after_success.append(trial)
            elif last_reward == 0:
                fail_after_fail.append(trial)
            
            second_last_reward = last_reward
            last_reward = 0

    return success, fail, str_trials, ftr_trials, ttr, total_trials, \
            fail_after_success, fail_after_fail, succ_after_one_succ, succ_after_two_succ

def get_peri_signal_of_fail_trial_types(ftr_trials, trialnum, eps, i, rewlocs, ypos, fs, range_val,
                binsize, areas_res):
    """
    takes input of success and fail trials (can be different fail trial types)
    """
    failtr_bool = np.array([any(yy.astype(int)==xx for yy in ftr_trials) for xx in trialnum[eps[i]:eps[i+1]]])
    failed_trialnum = trialnum[eps[i]:eps[i+1]][failtr_bool]
    rews_centered = np.zeros_like(failed_trialnum)            
    rews_centered[(ypos[failtr_bool] >= rewlocs[i]-5) & (ypos[failtr_bool] <= rewlocs[i])]=1
    rews_iind = eye.consecutive_stretch(np.where(rews_centered)[0])
    min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
    rews_centered = np.zeros_like(failed_trialnum)
    rews_centered[min_iind]=1
    rewards_ep = rews_centered[ypos[failtr_bool]>2]
    # fake time var
    time_ep = np.arange(0,rewards_ep.shape[0]/fs,1/fs)
    # licks_threshold_ep = licks_threshold[eps[i]:eps[i+1]][failtr_bool]
    # velocity_ep = velocity[eps[i]:eps[i+1]][failtr_bool]
    input_peri = areas_res[eps[i]:eps[i+1]][failtr_bool][ypos[failtr_bool]>2]
    normmeanrew_t_ep, meanrew_ep, normrewall_t_ep, \
    rewall_ep = perireward_binned_activity(np.array(input_peri), \
                            rewards_ep.astype(int),
                            time_ep, range_val, binsize)
    
    return rewall_ep