"""
jan 2025
generate circular statistics
"""

import scipy, numpy as np
from projects.pyr_reward.rewardcell import get_radian_position, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype,\
    consecutive_stretch,make_tuning_curves_radians_trial_by_trial
from itertools import combinations, chain

def compute_circular_stats(tuning_curve, positions, track_length):
    """
    Computes the circular mean and resultant vector length (a measure of variance)
    for a tuning curve along a circular track.

    Parameters:
    - tuning_curve: Array of firing rates at different positions.
    - positions: Array of positions along the track.
    - track_length: Total length of the circular track.

    Returns:
    - mean_angle (radians): Circular mean of the firing field.
    - resultant_vector_length (R): Measure of concentration (ranges from 0 to 1).
    """
    # Convert positions to angles (radians)
    angles = (2 * np.pi * positions) / track_length
    # Normalize tuning curve (acts as probability distribution)
    weights = tuning_curve / np.sum(tuning_curve)
    # Compute weighted circular mean
    x_mean = np.sum(weights * np.cos(angles))
    y_mean = np.sum(weights * np.sin(angles))
    mean_angle = np.arctan2(y_mean, x_mean)  # Circular mean in radians
    # Compute resultant vector length (R) as a measure of variance
    R = np.sqrt(x_mean**2 + y_mean**2)
    
    return mean_angle, R

def compute_circular_stats_rad(tuning_curve, positions):
    """
    Computes the circular mean and resultant vector length (a measure of variance)
    for a tuning curve along a circular track.

    Parameters:
    - tuning_curve: Array of firing rates at different positions.
    - positions: Array of positions along the track.
    - track_length: Total length of the circular track.

    Returns:
    - mean_angle (radians): Circular mean of the firing field.
    - resultant_vector_length (R): Measure of concentration (ranges from 0 to 1).
    """
    # Convert positions to angles (radians)
    angles = positions
    # Normalize tuning curve (acts as probability distribution)
    weights = tuning_curve / np.sum(tuning_curve)
    # Compute weighted circular mean
    x_mean = np.sum(weights * np.cos(angles))
    y_mean = np.sum(weights * np.sin(angles))
    mean_angle = np.arctan2(y_mean, x_mean)  # Circular mean in radians
    # Compute resultant vector length (R) as a measure of variance
    R = np.sqrt(x_mean**2 + y_mean**2)
    
    return mean_angle, R

def get_circular_data(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
        'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
    # get tuning curves trial by trial and get calculate radians
    trialstates, licks_trial_by_trial, tcs_trial_by_trial,\
    coms_trial_by_trial = make_tuning_curves_radians_trial_by_trial(eps,rewlocs,
        lick,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)
    # first get goal cells
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs        
    goal_cells = intersect_arrays(*com_goal)   
    all_goal_cells = np.concatenate(com_goal)
    rad_binned = np.linspace(0, 2*np.pi, bins)
    # compute circular statistics
    meanangles_all_gc = []; rvals_all_gc = []
    for ep in range(len(eps)-1):
        meanangle = []; rval = []
        for cll in all_goal_cells:
            tc = tcs_correct[ep,cll,:]
            mean_ang, r = compute_circular_stats_rad(tc, rad_binned)
            meanangle.append(mean_ang); rval.append(r)
        meanangles_all_gc.append(meanangle); rvals_all_gc.append(rval)
    # compute circular statistics for dedicated cells
    meanangles_gc = []; rvals_gc = []
    for ep in range(len(eps)-1):
        meanangle = []; rval = []
        for cll in goal_cells:
            tc = tcs_correct[ep,cll,:]
            mean_ang, r = compute_circular_stats_rad(tc, rad_binned)
            meanangle.append(mean_ang); rval.append(r)
        meanangles_gc.append(meanangle); rvals_gc.append(rval)
        
    return meanangles_gc, rvals_gc, meanangles_all_gc, rvals_all_gc, tcs_correct, coms_correct, \
        goal_cells, all_goal_cells