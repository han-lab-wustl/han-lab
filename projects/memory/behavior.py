import numpy as np, pandas as pd, matplotlib.pyplot as plt


def get_lick_selectivity_post_reward(ypos, trialnum, lick, time, rewloc, rewsize):
    """Assume Y is the position, which is a one-dimensional vector. L is the binary licking behavior, 
    which is also a one-dimensional vector. The start position of reward 
    zone is “reward location – ½ * reward zone size – 1”, which is a scalar, P.
    Licking number in the last quarter is 
    “L[np.where((Y/P < 1) & (Y/P > 0.75))[0]].sum()”. The total 
    licking number in the pre-reward zone 
    is “L[np.where((Y/P < 1) & (Y > 3.0))[0]].sum()”, 
    where I remove all the dark time licking.
    Licks in the last quarter / total pre-reward licks is what I define of licking accuracy.
    remember that time is in seconds!
    """
    lick_selectivity_per_trial = []
    for trial in np.unique(trialnum):
        ypos_t = ypos[trialnum==trial]
        lick_t = lick[trialnum==trial]
        start_postion = rewloc-(.5*rewsize)
        time_t = time[trialnum==trial]
        time_start = time_t[ypos_t>start_postion]
        if len(time_start)>2:
            time_start = time_start[1]
            total_licks = lick_t[np.where(ypos_t > 3)[0]].sum()
            if total_licks>0:
                # get 2 s later
                # in_stim_zone = lick_t[np.where((time_t>time_start) & (time_t<(time_start+2.2)))[0]].sum()
                in_stim_zone = lick_t[np.where((time_t>time_start)&(time_t<=time_start+2))[0]].sum()
                lick_selectivity = in_stim_zone/total_licks 
            else:
                lick_selectivity = np.nan
                in_stim_zone = np.nan
                total_licks = 0
        else:
            lick_selectivity = np.nan
            in_stim_zone = np.nan
            total_licks = 0
        
        # lick_selectivity_per_trial.append(lick_selectivity)
        # test: licks/s
        lick_selectivity_per_trial.append(lick_selectivity)
    
    return lick_selectivity_per_trial

def calculate_lick_rate(recording, sampling_rate = 7.8):
    """
    Calculate the lick rate from a recording sampled at 7.8 Hz.

    Parameters:
    recording (list or array): A list or array where each element represents a sample at 7.8 Hz.
            A lick is represented by a '1' and no lick by a '0'.

    Returns:
    float: The lick rate in licks per second.
    """
    
    # Count the number of licks
    number_of_licks = sum(recording)
    
    # Calculate the duration of the recording in seconds
    duration_seconds = len(recording) / sampling_rate
    
    # Calculate the lick rate
    lick_rate = number_of_licks / duration_seconds
    
    return lick_rate

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
            lick_selectivity = 2
        
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


def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = []
    if 0 in break_point: y.append([x[break_point[0]]]) # since it does not iterate from 0
    for i in range(1, len(break_point)):
        xx = x[break_point[i - 1] + 1:break_point[i]]
        if len(xx)==0: xx = [x[break_point[i]]]
        y.append(xx)
    y.append(x[break_point[-1] + 1:])
    
    return y
    
def get_success_failure_trials(trialnum, reward):
    """
    Quantify successful and failed trials based on trial numbers and rewards.

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
    """
    success = 0
    fail = 0
    str_trials = []
    ftr_trials = []

    for trial in np.unique(trialnum):
        if trial >= 3:  # Exclude probe trials (trial < 3)
            if np.sum(reward[trialnum == trial] == 1) > 0:  # If reward was found in the trial
                success += 1
                str_trials.append(trial)
            else:
                fail += 1
                ftr_trials.append(trial)

    total_trials = np.sum(np.unique(trialnum) >= 3)
    ttr = np.unique(trialnum)[np.unique(trialnum) > 2]  # Remove probe trials

    return success, fail, str_trials, ftr_trials, ttr, total_trials
