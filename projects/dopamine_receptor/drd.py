"""
functions for drd cell analysis
"""
import os, sys, re
import numpy as np
import statsmodels.api as sm
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from projects.pyr_reward.rewardcell import perireward_binned_activity

import numpy as np

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

def get_moving_time_v3(velocity, thres, Fs, ftol):
    """
    Returns time points when the animal is considered moving based on the animal's change in velocity

    Parameters:
    velocity - ndarray: velocity of the animal
    thres - float: Threshold speed in cm/s
    Fs - int: Number of frames minimum to be considered stopped
    ftol - int: Frame tolerance for merging stop periods

    Returns:
    moving_middle - ndarray: Indices of points when the animal is moving
    stop - ndarray: Indices of points when the animal is stopped
    """
    vr_speed = velocity
    vr_thresh = thres

    moving = np.where(vr_speed > vr_thresh)[0]
    stop = np.where(vr_speed <= vr_thresh)[0]

    stop_time_stretch = consecutive_stretch(stop)
    stop_time_length = [len(stretch) for stretch in stop_time_stretch]
    stop_time_stretch = [stretch for stretch, length in zip(stop_time_stretch, stop_time_length) if length >= Fs]

    if len(stop_time_stretch) > 0:
        for s in range(len(stop_time_stretch)-1):
            d = 0
            while (s + d) < (len(stop_time_stretch) - 1):
                if not np.isnan(stop_time_stretch[s + d]).any():
                    while (s + d < len(stop_time_stretch) - 1) and abs(stop_time_stretch[s][-1] - stop_time_stretch[s + d][0]) <= ftol:
                        stop_time_stretch[s] = np.concatenate((stop_time_stretch[s], 
                                    np.arange(stop_time_stretch[s][-1] + 1,
                                    stop_time_stretch[s + d][0]), stop_time_stretch[s + d]))
                        stop_time_stretch[s + d] = [np.nan]
                    d += 1
                else:
                    break
                
            # re-check length after modifying the contents
            while s + d >= len(stop_time_stretch):
                d -= 1

        stop_time_stretch = [stretch for stretch in stop_time_stretch if not np.isnan(stretch).any()]
        stop = np.concatenate(stop_time_stretch)
        moving_time = np.ones(len(vr_speed))
        moving_time[stop] = 0
    else:
        moving_time = np.ones(len(vr_speed))

    moving = np.where(moving_time == 1)[0]
    moving_middle = moving
    return moving_middle, stop

# The script using this function:
def get_stops(moving_middle, stop, pre_win_framesALL, post_win_framesALL,
        forwardvelALL, reward_binned, max_reward_stop=31.25*5):
    """from gerardo

    Args:
        moving_middle (_type_): _description_
        stop (_type_): _description_
        pre_win_framesALL (_type_): _description_
        post_win_framesALL (_type_): _description_
        forwardvelALL (_type_): _description_
        reward_binned (_type_): _description_
        max_reward_stop (_type_, optional): number of seconds after reward for a stop
        to be considered a reward related stop * frame rate.

    Returns:
        _type_: _description_
    """
    mov_success_tmpts = moving_middle[np.where(np.diff(moving_middle) > 1)[0] + 1]

    idx_rm = (mov_success_tmpts - pre_win_framesALL) <= 0
    rm_idx = np.where(idx_rm)[0]
    mov_success_tmpts = np.delete(mov_success_tmpts, rm_idx)

    if len(mov_success_tmpts) > 0:
        mov_success_tmpts = np.delete(mov_success_tmpts, -1)  # Remove the last element
        
    idx_rm = (mov_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    mov_success_tmpts = np.delete(mov_success_tmpts, rm_idx)

    stop_success_tmpts = moving_middle[np.where(np.diff(moving_middle) > 1)[0]] + 1

    idx_rm = (stop_success_tmpts - pre_win_framesALL) < 0
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    if len(stop_success_tmpts) > 0:
        stop_success_tmpts = np.delete(stop_success_tmpts, -1)  # Remove the last element

    idx_rm = (stop_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    rew_idx = np.where(reward_binned)[0]

    rew_stop_success_tmpts = []
    for r in rew_idx: # iterate through all rewards
        if r in stop:
            last_stop_before_rew = stop_success_tmpts[np.where(stop_success_tmpts < r)[0]]
            if len(last_stop_before_rew) > 0:
                rew_stop_success_tmpts.append(last_stop_before_rew[-1])
            else:
                rew_stop_success_tmpts.append(np.nan)
        else:
            closest_future_stop = stop_success_tmpts[np.where((stop_success_tmpts - r >= 0) \
            & (stop_success_tmpts - r < max_reward_stop))[0]]
            if len(closest_future_stop) > 0:
                rew_stop_success_tmpts.append(closest_future_stop[0])
            else:
                rew_stop_success_tmpts.append(np.nan)

    rew_stop_success_tmpts = np.array(rew_stop_success_tmpts)
    didntstoprew = np.isnan(rew_stop_success_tmpts)
    rew_stop_success_tmpts = rew_stop_success_tmpts[~didntstoprew]
    rew_stop_success_tmpts = np.unique(rew_stop_success_tmpts)
    nonrew_stop_success_tmpts = np.setxor1d(rew_stop_success_tmpts, stop_success_tmpts)

    idx_rm = (stop_success_tmpts - pre_win_framesALL) <= 0
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    if len(stop_success_tmpts) > 0:
        stop_success_tmpts = np.delete(stop_success_tmpts, -1)  # Remove the last element

    idx_rm = (stop_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)
    
    return nonrew_stop_success_tmpts, rew_stop_success_tmpts

def get_stops_licks(moving_middle, stop, pre_win_framesALL, post_win_framesALL,
              forwardvelALL, reward_binned, lick_binned, max_reward_stop=31.25*5):
    """from gerardo

    Args:
        moving_middle (_type_): _description_
        stop (_type_): _description_
        pre_win_framesALL (_type_): _description_
        post_win_framesALL (_type_): _description_
        forwardvelALL (_type_): _description_
        reward_binned (_type_): _description_
        lick_binned (_type_): binary variable indicating licks (1 if lick occurred, 0 otherwise)
        max_reward_stop (_type_, optional): number of seconds after reward for a stop
        to be considered a reward related stop * frame rate.

    Returns:
        tuple: (non_reward_non_lick_stops, non_reward_lick_stops, reward_non_lick_stops, reward_lick_stops)
    """
    mov_success_tmpts = moving_middle[np.where(np.diff(moving_middle) > 1)[0] + 1]

    idx_rm = (mov_success_tmpts - pre_win_framesALL) <= 0
    rm_idx = np.where(idx_rm)[0]
    mov_success_tmpts = np.delete(mov_success_tmpts, rm_idx)

    if len(mov_success_tmpts) > 0:
        mov_success_tmpts = np.delete(mov_success_tmpts, -1)  # Remove the last element

    idx_rm = (mov_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    mov_success_tmpts = np.delete(mov_success_tmpts, rm_idx)

    stop_success_tmpts = moving_middle[np.where(np.diff(moving_middle) > 1)[0]] + 1

    idx_rm = (stop_success_tmpts - pre_win_framesALL) < 0
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    if len(stop_success_tmpts) > 0:
        stop_success_tmpts = np.delete(stop_success_tmpts, -1)  # Remove the last element

    idx_rm = (stop_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    rew_idx = np.where(reward_binned)[0]

    rew_stop_success_tmpts = []
    for r in rew_idx:  # iterate through all rewards
        if r in stop:
            last_stop_before_rew = stop_success_tmpts[np.where(stop_success_tmpts < r)[0]]
            if len(last_stop_before_rew) > 0:
                rew_stop_success_tmpts.append(last_stop_before_rew[-1])
            else:
                rew_stop_success_tmpts.append(np.nan)
        else:
            closest_future_stop = stop_success_tmpts[np.where((stop_success_tmpts - r >= 0) &
                        (stop_success_tmpts - r < max_reward_stop))[0]]
            if len(closest_future_stop) > 0:
                rew_stop_success_tmpts.append(closest_future_stop[0])
            else:
                rew_stop_success_tmpts.append(np.nan)

    rew_stop_success_tmpts = np.array(rew_stop_success_tmpts)
    didntstoprew = np.isnan(rew_stop_success_tmpts)
    rew_stop_success_tmpts = rew_stop_success_tmpts[~didntstoprew]
    rew_stop_success_tmpts = np.unique(rew_stop_success_tmpts)
    nonrew_stop_success_tmpts = np.setxor1d(rew_stop_success_tmpts, stop_success_tmpts)

    idx_rm = (stop_success_tmpts - pre_win_framesALL) <= 0
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)

    if len(stop_success_tmpts) > 0:
        stop_success_tmpts = np.delete(stop_success_tmpts, -1)  # Remove the last element

    idx_rm = (stop_success_tmpts + post_win_framesALL) > len(forwardvelALL) - 10
    rm_idx = np.where(idx_rm)[0]
    stop_success_tmpts = np.delete(stop_success_tmpts, rm_idx)
    
    # Determine stops with and without licks
    lick_idx = np.where(lick_binned)[0]
    
    # Function to check for licks in a given range around stops
    def no_licks_in_range(stop_frame, range_size):
        check_range = np.arange(stop_frame - range_size, stop_frame + range_size + 1)
        return np.all(np.isin(check_range, lick_idx, invert=True))

    stop_with_lick = []
    stop_without_lick = []
    range_size = 5

    for stop in stop_success_tmpts:
        if no_licks_in_range(stop, range_size):
            stop_without_lick.append(stop)
        else:
            stop_with_lick.append(stop)

    stop_without_lick = np.array(stop_without_lick)
    stop_with_lick = np.array(stop_with_lick)

    # Split reward stops into those with and without licks
    rew_stop_with_lick = np.intersect1d(rew_stop_success_tmpts, stop_with_lick)
    rew_stop_without_lick = np.intersect1d(rew_stop_success_tmpts, stop_without_lick)

    # Split non-reward stops into those with and without licks
    nonrew_stop_with_lick = np.intersect1d(nonrew_stop_success_tmpts, stop_with_lick)
    nonrew_stop_without_lick = np.intersect1d(nonrew_stop_success_tmpts, stop_without_lick)
    
    return nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, rew_stop_with_lick,\
        mov_success_tmpts

def extract_plane_number(path):
    # Search for 'plane' followed by a number
    match = re.search(r'plane(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No plane number found in the path")


# Function to load and filter dFF_iscell data
def load_and_filter_fall_data(fall_file):
    f = loadmat(fall_file)
    
    # Extract necessary variables
    dFF_iscell = f['dFF_iscell']
    stat = f['stat'][0]
    iscell = f['iscell'][:, 0].astype(bool)
    
    # Determine cells to keep, excluding merged ROIs
    statiscell = [stat[i] for i in range(len(stat)) if iscell[i]]
    garbage = []
    for st in statiscell:
        if 'imerge' in st.dtype.names and len(st['imerge'][0]) > 0:
            garb =  st['imerge'][0].flatten().tolist()
            garbage.extend(garb)
    
    arr = [x[0] for x in garbage if len(x) > 0]
    if len(arr) > 0:
        garbage = np.unique(np.concatenate(arr))
    else:
        garbage = []
    
    cllsind = np.arange(f['F'].shape[0])
    cllsindiscell = cllsind[iscell]
    keepind = ~np.isin(cllsindiscell, garbage)

    # Filter dFF_iscell
    dFF_iscell_filtered = dFF_iscell[keepind, :]
    return dFF_iscell_filtered, f

# Function to perform GLM on dFF_iscell data
def run_glm(dFF_iscell_filtered, f):
    dff_res = []
    range_val, binsize = 12, 0.2  # s
    perirew = []
    
    for cll in range(dFF_iscell_filtered.shape[0]):
        X = np.array([f['forwardvel'][0]]).T  # Predictor(s)
        X = sm.add_constant(X)  # Adds a constant term to the predictor(s)
        y = dFF_iscell_filtered[cll, :]  # Outcome

        # Fit a regression model
        model = sm.GLM(y, X, family=sm.families.Gaussian())
        result = model.fit()
        dff_res.append(result.resid_pearson)

        # Peri-reward activity
        dff = dFF_iscell_filtered[cll, :]
        normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
            dff, (f['rewards'][0] == 0.5).astype(int),
            f['timedFF'][0], f['trialnum'][0], range_val, binsize
        )
        perirew.append([meanrewdFF, rewdFF])

    dff_res = np.array(dff_res)
    return dff_res, perirew
