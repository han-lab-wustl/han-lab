import numpy as np, pandas as pd

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
