import numpy as np, math, scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import pearsonr, ranksums
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
from scipy.signal import gaussian
from scipy.ndimage import label
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import get_success_failure_trials

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
    
    return np.array(beh_probability)


def get_tuning_curve(ybinned, f, bins=270):
    """
    """
    df = pd.DataFrame()
    df['position'] = ybinned
    df['f'] = f
    # Discretize the position data into bins
    df['position_bin'] = pd.cut(df['position'], bins=bins, labels=False)
    
    # Calculate the lick probability for each bin
    grouped = df.groupby('position_bin')['f'].agg(['mean', 'count']).reset_index()
    f_tc = np.ones(bins)*np.nan
    f_tc[:np.array(grouped['mean'].shape[0])] = grouped['mean'] 
    
    return np.array(f_tc)


def make_tuning_curves_radians_trial_by_trial(eps,rewlocs,lick,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90):
    trialstates = []; licks = []; tcs = []; coms = []    
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        rewloc = rewlocs[ep]
        # excludes probe trials
        success, fail, strials, ftrials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        # make sure mouse did full trial & includes probes
        trials = [xx for xx in np.unique(trialnum[eprng]) if np.sum(trialnum[eprng]==xx)>100]
        trialstate = np.ones(len(trials))*-1
        # check if original trial num is correct or not
        trialstate[[xx for xx,t in enumerate(trials) if t in strials]] = 1
        trialstate[[xx for xx,t in enumerate(trials) if t in ftrials]] = 0
        trialstates.append(trialstate)
        F = Fc3[eprng,:]            
        # moving_middle,stop = get_moving_time_V3(forwardvel[eprng], 5, 5, 10)
        # simpler metric to get moving time
        moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        # overwrite velocity filter
        moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F = F[moving_middle,:]
        # cells x trial x bin        
        tcs_per_trial = np.ones((F.shape[1], len(trials), bins))*np.nan
        coms_per_trial = np.ones((F.shape[1], len(trials)))*np.nan
        licks_per_trial = np.ones((len(trials), bins))*np.nan        
        if len(ttr)>lasttr: # only if ep has more than x trials            
            for tt,trial in enumerate(trials): # iterates through the unique trials and not the
                # trial number itself
                # need to account for 0-2 probes in the beginning
                # that makes the starting trialnum 1 or 2 or 3
                mask = trialnum[eprng][moving_middle]==trial
                relpos = rad[eprng][moving_middle][mask]                
                licks_ep = lick[eprng][moving_middle][mask]                
                for celln in range(F.shape[1]):
                    f = F[mask,celln]
                    tc = get_tuning_curve(relpos, f, bins=bins)  
                    tc[np.isnan(tc)]=0 # set nans to 0
                    tcs_per_trial[celln, tt,:] = tc
                com = calc_COM_EH(tcs_per_trial[:, tt,:],bin_size)
                coms_per_trial[:, tt] = com
                lck = get_tuning_curve(relpos, licks_ep, bins=bins) 
                lck[np.isnan(lck)]=0
                licks_per_trial[tt,:] = lck
        tcs.append(tcs_per_trial)
        coms.append(coms_per_trial)
        licks.append(licks_per_trial)

    return trialstates, licks, tcs, coms


def make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90,velocity_filter=False):    
    """
    Description: This function creates tuning curves for neuronal activity aligned to reward locations and categorizes them by trial type (correct or fail). The tuning curves are generated for each epoch, and the data is filtered based on velocity if the option is enabled.
    Parameters:
    eps (numpy.ndarray): Array of epoch (trial segment) start indices.
    rewlocs (numpy.ndarray): Array of reward locations for each epoch.
    ybinned (numpy.ndarray): Array of position data (binned).
    rad (numpy.ndarray): Array of radian positions.
    Fc3 (numpy.ndarray): Fluorescence data of cells. The shape should be (time, cells).
    trialnum (numpy.ndarray): Array with trial numbers.
    rewards (numpy.ndarray): Array indicating whether a reward was received at each time point.
    forwardvel (numpy.ndarray): Array of forward velocity values at each time point.
    rewsize (float): Size of the reward zone.
    bin_size (float): Size of the bin for the tuning curve.
    lasttr (int, optional): The number of last correct trials considered for analysis (default is 8).
    bins (int, optional): The number of bins for the tuning curve (default is 90).
    velocity_filter (bool, optional): Whether to apply a velocity filter to include only times when velocity > 5 cm/s (default is False).
    Returns:
    tcs_correct (numpy.ndarray): Tuning curves for correct trials. Shape is (epochs, cells, bins).
    coms_correct (numpy.ndarray): Center of mass (COM) for correct trials. Shape is (epochs, cells).
    tcs_fail (numpy.ndarray): Tuning curves for failed trials. Shape is (epochs, cells, bins).
    coms_fail (numpy.ndarray): Center of mass (COM) for failed trials. Shape is (epochs, cells).
    """ 
    rates = []; 
    # initialize
    tcs_fail = np.ones((len(eps)-1, Fc3.shape[1], bins))*np.nan
    tcs_correct = np.ones((len(eps)-1, Fc3.shape[1], bins))*np.nan
    coms_correct = np.ones((len(eps)-1, Fc3.shape[1]))*np.nan
    coms_fail = np.ones((len(eps)-1, Fc3.shape[1]))*np.nan
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        rewloc = rewlocs[ep]
        relpos = rad[eprng]        
        success, fail, strials, ftrials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        F_all = Fc3[eprng,:]            
        # simpler metric to get moving time
        if velocity_filter==True:
            moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        else:
            moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F_all = F_all[moving_middle,:]
        relpos_all = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            # last 8 correct trials
            if len(strials)>0:
                mask = [True if xx in strials[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_correct[ep, :,:] = tc
                coms_correct[ep, :] = com
            # failed trials
            if len(ftrials)>0:
                mask = [True if xx in ftrials else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_fail[ep, :, :] = tc
                coms_fail[ep, :] = com
    
    return tcs_correct, coms_correct, tcs_fail, coms_fail


def make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90,
            velocity_filter=False):
    rates = []; tcs_fail = []; tcs_correct = []; coms_correct = []; coms_fail = []        
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        rewloc = rewlocs[ep]
        relpos = ybinned[eprng]        
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        F = Fc3[eprng,:]            
        # simpler metric to get moving time
        if velocity_filter==True:
            moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        else:
            moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F = F[moving_middle,:]
        relpos = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            # last 8 correct trials
            if len(strials)>0:
                mask = [True if xx in strials[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F[mask,:]
                relpos = relpos[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_correct.append(tc)
                coms_correct.append(com)
            # failed trials
            elif len(ftrials)>0:
                mask = [True if xx in ftrials else False for xx in trialnum[eprng][moving_middle]]
                F = F[mask,:]
                relpos = relpos[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_fail.append(tc)
                coms_fail.append(com)
    tcs_correct = np.array(tcs_correct); coms_correct = np.array(coms_correct)  
    tcs_fail = np.array(tcs_fail); coms_fail = np.array(coms_fail)  
    
    return tcs_correct, coms_correct, tcs_fail, coms_fail

def make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90,
            velocity_filter=False):
    rates = []; tcs_fail = []; tcs_correct = []; coms_correct = []; coms_fail = []        
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        rewloc = rewlocs[ep]
        relpos = ybinned[eprng]        
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        F = Fc3[eprng,:]            
        # simpler metric to get moving time
        if velocity_filter==True:
            moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        else:
            moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F = F[moving_middle,:]
        relpos = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            # last 8 trials            
            mask = [True if xx in ttr[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
            F = F[mask,:]
            relpos = relpos[mask]                
            tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
            com = calc_COM_EH(tc,bin_size)
            tcs_correct.append(tc)
            coms_correct.append(com)
    tcs_correct = np.array(tcs_correct); coms_correct = np.array(coms_correct)      
    
    return tcs_correct, coms_correct

def make_tuning_curves_relative_to_reward(eps,rewlocs,ybinned,track_length,Fc3,trialnum,
            rewards,forwardvel,rewsize,lasttr=5,bins=100):
    ypos_rel = []; tcs_early = []; tcs_late = []; coms = []    
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        rewloc = rewlocs[ep]
        relpos = [(xx-rewloc)/rewloc if xx<(rewloc-rewsize) else (xx-rewloc)/(track_length-rewloc) for xx in ybinned[eprng]]            
        ypos_rel.append(relpos)
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        F = Fc3[eprng,:]            
        # simpler metric to get moving time
        moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        F = F[moving_middle,:]
        relpos = np.array(relpos)[moving_middle]
        if len(ttr)>5:
            mask = trialnum[eprng][moving_middle]>ttr[-lasttr]
            F = F[mask,:]
            relpos = relpos[mask]                
            tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
            com = calc_COM_EH(tc,track_length/bins)
            tcs_late.append(tc)
            coms.append(com)

    return ypos_rel, tcs_late, coms

def get_place_field_widths(tuning_curves, threshold=0.5):
    """
    Calculate place field widths around peak firing fields for each cell.
    
    Parameters:
    tuning_curves (pd.DataFrame): DataFrame where each row represents a cell and each column a position.
    threshold (float): Proportion of peak firing rate to define place field boundaries (default is 0.5, i.e., 50%).
    
    Returns:
    pd.DataFrame: DataFrame with place field widths for each cell.
    """
    n_cells = tuning_curves.shape[0]
    place_field_widths = []

    for cell in range(n_cells):
        firing_rates = tuning_curves[cell, :]
        peak_rate = np.max(firing_rates)
        threshold_rate = threshold * peak_rate
        
        # Find the positions where the firing rate is above the threshold
        above_threshold = np.where(firing_rates >= threshold_rate)[0]
        
        if above_threshold.size == 0:
            place_field_widths.append(np.nan)
            continue
        
        # Calculate the width as the distance between the first and last position above the threshold
        width = above_threshold[-1] - above_threshold[0] + 1
        place_field_widths.append(width)
    
    return place_field_widths

def calculate_global_remapping(data_reward1, data_reward2, 
    num_iterations=1000):
    n_cells = data_reward1.shape[0]
    threshold=0.1 # arbitrary for now
    # Calculate real cosine similarities
    real_CS = []
    for neuron in range(data_reward1.shape[0]):
        x = data_reward1[neuron, :]
        y = data_reward2[neuron, :]
        cs = get_cosine_similarity(x, y)
        real_CS.append(cs)
    
    real_CS = np.array(real_CS)
    global_remapping = real_CS < threshold
    
    # Shuffled distribution
    shuffled_CS = []
    for _ in range(num_iterations):
        shuffled_indices = np.random.permutation(n_cells)
        shuffled_data_reward2 = data_reward2[shuffled_indices, :]
        shuffled_cs = []
        for neuron in range(data_reward1.shape[0]):
            x = data_reward1[neuron, :]
            y = shuffled_data_reward2[neuron, :]
            cs = get_cosine_similarity(x, y)
            shuffled_cs.append(cs)
        shuffled_CS.append(shuffled_cs)    
    shuffled_CS = np.array(shuffled_CS)
    
    # remove nan cell
    real_CS_ = real_CS[~np.isnan(real_CS)]
    shuffled_CS_ = shuffled_CS[:, ~np.isnan(real_CS)]
    # Calculate p-values
    p_values = []
    for ii,real_cs in enumerate(real_CS_):
        p_value = np.sum(shuffled_CS_[:,ii] > real_cs) / num_iterations
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    # Compare real vs shuffled using ranksum test
    H, P = ranksums(real_CS_, np.nanmean(shuffled_CS_,axis=0))
    
    real_distribution = real_CS_
    shuffled_distribution = shuffled_CS_
    
    return P, H, real_distribution, shuffled_distribution, p_values, global_remapping

def get_cosine_similarity(vec1, vec2):
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def perivelocitybinnedactivity(velocity, rewards, dff, timedFF, range_val, binsize, numplanes):
    """
    Compute binned peri-velocity activity around non-reward stops.

    Parameters:
    velocity (numpy.ndarray): Velocity data.
    rewards (numpy.ndarray): Reward indices.
    dff (numpy.ndarray): dF/F data.
    timedFF (numpy.ndarray): Time stamps for dF/F data.
    range_val (float): Range of time around stops (in seconds).
    binsize (float): Bin size (in seconds).
    numplanes (int): Number of planes.

    Returns:
    binnedPerivelocity (numpy.ndarray): Binned peri-velocity activity.
    allbins (numpy.ndarray): Bin centers.
    rewvel (numpy.ndarray): Peri-velocity activity for each non-reward stop.
    """
    # dff aligned to stops
    moving_middle = get_moving_time(velocity, 2, 10, 30)

    stop_idx = moving_middle[np.where(np.diff(moving_middle) > 1)[0] + 1]

    # find stops without reward
    frame_rate = 31.25 / numplanes
    max_reward_stop = 10 * frame_rate  # number of seconds after reward for a stop to be considered a reward related stop * frame rate.
    rew_idx = np.where(rewards)[0]
    rew_stop_idx = []
    frame_tol = 10  # number of frames prior to reward to check for stopping points as a tolerance for defining stopped.

    for r in rew_idx:
        stop_candidates = stop_idx[(stop_idx - r >= 0 - frame_tol) & (stop_idx - r < max_reward_stop)]
        if len(stop_candidates) > 0:
            rew_stop_idx.append(stop_candidates[0])
        else:
            rew_stop_idx.append(np.nan)

    rew_stop_idx = np.array(rew_stop_idx)
    rew_stop_idx = rew_stop_idx[~np.isnan(rew_stop_idx)].astype(int)
    non_rew_stops = np.setdiff1d(stop_idx, rew_stop_idx, assume_unique=True)

    # binsize = 0.1  # half a second bins
    # range_val = 6  # seconds back and forward in time
    rewvel = np.zeros((int(np.ceil(2 * range_val / binsize)), dff.shape[1], len(non_rew_stops)))

    for rr, non_rew_stop in enumerate(non_rew_stops):
        rewtime = timedFF[non_rew_stop]
        currentrewchecks = np.where((timedFF > rewtime - range_val) & (timedFF <= rewtime + range_val))[0]
        currentrewcheckscell = consecutive_stretch(currentrewchecks)
        currentrewardlogical = [non_rew_stop in x for x in currentrewcheckscell]

        for bin_idx in range(int(np.ceil(2 * range_val / binsize))):
            testbin = round(-range_val + bin_idx * binsize - binsize, 13)  # round to nearest 13 so 0 = 0 and not 3.576e-16
            currentidxt = np.where((timedFF > rewtime - range_val + bin_idx * binsize - binsize) &
                                   (timedFF <= rewtime - range_val + bin_idx * binsize))[0]
            checks = consecutive_stretch(currentidxt)

            if checks:
                currentidxlogical = [max(any(np.isin(x, currentrewcheckscell[i])) for x in checks) for i in currentrewardlogical]
                if sum(currentidxlogical) > 0:
                    checkidx = np.array(checks)[currentidxlogical]
                    rewvel[bin_idx, :, rr] = np.mean(dff[np.concatenate(checkidx), :], axis=0, keepdims=True)
                else:
                    rewvel[bin_idx, :, rr] = np.nan
            else:
                rewvel[bin_idx, :, rr] = np.nan

    meanrewvel = np.nanmean(rewvel, axis=2)
    binnedPerivelocity = meanrewvel.T
    allbins = np.array([round(-range_val + bin_idx * binsize - binsize, 13) for bin_idx in range(int(np.ceil(2 * range_val / binsize)))])

    return binnedPerivelocity, allbins, rewvel

def get_moving_time_V3(velocity, thres, Fs, ftol):
    """
    It returns time points when the animal is considered moving based on the animal's change in y position.
    velocity - forward velocity
    thres - Threshold speed in cm/s
    Fs - number of frames length minimum to be considered stopped.
    ftol - frame tolerance for merging close stop periods.
    """
    vr_speed = np.array(velocity)
    vr_thresh = thres
    moving = np.where(vr_speed > vr_thresh)[0]
    stop = np.where(vr_speed <= vr_thresh)[0]

    stop_time_stretch = consecutive_stretch_mov_time(stop)

    stop_time_length = [len(stretch) for stretch in stop_time_stretch]
    delete_idx = [i for i, length in enumerate(stop_time_length) if length < Fs]
    stop_time_stretch = [stretch for i, stretch in enumerate(stop_time_stretch) if i not in delete_idx]

    if len(stop_time_stretch) > 0:
        for s in range(len(stop_time_stretch) - 1):
            d = 1
            if s + d < len(stop_time_stretch):
                if not np.isnan(stop_time_stretch[s + d]).all():
                    while abs(stop_time_stretch[s][-1] - stop_time_stretch[s + d][0]) <= ftol and s + d < len(stop_time_stretch):
                        stop_time_stretch[s] = np.concatenate([
                            stop_time_stretch[s],
                            np.arange(stop_time_stretch[s][-1] + 1, stop_time_stretch[s + d][0]),
                            stop_time_stretch[s + d]
                        ])
                        stop_time_stretch[s + d] = np.array([np.nan])
                        d += 1
                        
        stop_time_stretch = [stretch for stretch in stop_time_stretch if not np.isnan(stretch).any()]
        stop = np.concatenate(stop_time_stretch).astype(int)
        moving_time = np.ones(len(vr_speed), dtype=int)
        moving_time[stop] = 0
    else:
        moving_time = np.ones(len(vr_speed), dtype=int)

    moving = np.where(moving_time == 1)[0]
    moving_middle = moving

    return moving_middle, stop

def consecutive_stretch_mov_time(arr):
    """
    This function finds consecutive stretches in an array.
    It returns a list of arrays, where each array contains the indices of a consecutive stretch.
    """
    stretched, num_features = label(np.diff(arr) == 1)
    stretches = [arr[np.where(stretched == i)[0] + 1] for i in range(1, num_features + 1)]
    return stretches


def calc_COM_EH(spatial_act, bin_width):
    """
    Calculate Center of Mass (COM) for each cell's tuning curve.

    Parameters:
    spatial_act : numpy array
        Tuning curve where rows represent cells and columns represent bins.
    bin_width : float
        Width of each bin in centimeters.

    Returns:
    com : numpy array
        Array of interpolated COM values in centimeters for each cell.
    """

    # Initialize arrays
    binn = np.zeros(spatial_act.shape[0]).astype(int)  # 1st bin above mid point
    frac = np.zeros(spatial_act.shape[0])  # Fraction for interpolated COM
    com = np.zeros(spatial_act.shape[0])  # Interpolated COM in cm

    # Get total fluorescence from tuning curve
    sum_spatial_act = np.nansum(spatial_act, axis=1)

    # Mid point of total fluorescence
    mid_sum = sum_spatial_act / 2

    # Cumulative sum of fluorescence in tuning curve
    spatial_act_cum_sum = np.nancumsum(spatial_act, axis=1)

    # Logical array of indexes above mid fluorescence
    idx_above_mid = spatial_act_cum_sum >= mid_sum[:, np.newaxis]

    for i in range(spatial_act.shape[0]):
        if not np.isnan(sum_spatial_act[i]):
            # Find index of first bin above mid fluorescence
            binn[i] = int(np.argmax(idx_above_mid[i, :]))

            # Linear interpolation
            if binn[i] == 0:  # If mid point is in the 1st bin
                frac[i] = (spatial_act_cum_sum[i, binn[i]] - mid_sum[i]) / spatial_act_cum_sum[i, binn[i]]
                com[i] = frac[i] * bin_width
            else:
                frac[i] = (spatial_act_cum_sum[i, binn[i]] - mid_sum[i]) / (spatial_act_cum_sum[i, binn[i]] - spatial_act_cum_sum[i, binn[i] - 1])
                com[i] = (binn[i] - 1 + frac[i]) * bin_width
        else:
            com[i] = np.nan

    return com

def get_spatial_info_per_cell(Fc3, fv, thres, ftol, position, Fs, nBins, track_length):
    """
    Fc3: dFF of 1 cell
    position: position of animal on track
    Fs: Frame rate of acquisition
    nBins: number of bins in which you want to divide the track into
    track_length: Length of track
    """

    time_moving, _ = get_moving_time(fv, thres, Fs, ftol)
    bin_size = track_length / nBins
    pos_moving = position[time_moving]

    time_in_bin = {i: time_moving[np.logical_and(pos_moving > (i - 1) * bin_size, pos_moving <= i * bin_size)] for i in range(1, nBins + 1)}

    cell_activity = np.array([np.mean(Fc3[time_in_bin[bin]]) for bin in range(1, nBins + 1)])
    # cell_activity = gaussian(cell_activity, 5)  # Uncomment if you want to apply Gaussian smoothing

    lambda_all = np.mean(Fc3[time_moving])
    time_fraction = np.array([len(time_in_bin[bin]) / len(time_moving) for bin in range(1, nBins + 1)])

    temp = time_fraction * cell_activity * np.log2(cell_activity / lambda_all)
    temp[np.isinf(temp)] = 0
    temp[np.isnan(temp)] = 0

    info = np.sum(temp / lambda_all)

    if np.isnan(info):
        info = 0

    return info


def convert_coordinates(coordinates, center_location, track_length=270):
    """
    Convert track coordinates from 0 to track_length (default: 270 cm) to -pi to pi radians,
    centered at a specified location.

    Args:
        coordinates (numpy.ndarray): 1D array of track coordinates in cm.
        center_location (float): Location to center the coordinates at, in cm.
        track_length (float, optional): Length of the track in cm (default: 270).

    Returns:
        numpy.ndarray: Converted coordinates in radians, centered at the specified location.
    """
    # Convert coordinates and center_location to radians
    coordinates_radians = coordinates * (2 * np.pi / track_length)
    center_radians = center_location * (2 * np.pi / track_length)

    # Center coordinates_radians around center_radians
    centered_coordinates_radians = coordinates_radians - center_radians

    # Wrap the centered_coordinates_radians to -pi to pi range
    centered_coordinates_radians = (centered_coordinates_radians + np.pi) % (2 * np.pi) - np.pi

    return centered_coordinates_radians


def intersect_arrays(*arrays):
    """
    Find the intersection between multiple NumPy arrays.

    Args:
        *arrays: Variable number of NumPy arrays.

    Returns:
        numpy.ndarray: Array containing the intersection of all input arrays.
    """
    # Convert arguments to a list of arrays
    arrays = list(arrays)

    # Base case: If there is only one array, return it
    if len(arrays) == 1:
        return arrays[0]

    # Find the intersection between the first two arrays
    intersection = np.intersect1d(arrays[0], arrays[1])

    # Find the intersection between the result and the remaining arrays
    for arr in arrays[2:]:
        intersection = np.intersect1d(intersection, arr)

    return intersection   

def convert_com_to_radians(com, reward_location, track_length):
    """
    Convert the center of mass of pyramidal cell activity from 0 to 270 cm
    to -pi to pi radians, centered at the reward location.

    Args:
        com (float): Center of mass of pyramidal cell activity in cm (0 to 270).
        reward_location (float): Reward location in cm (0 to 270).

    Returns:
        float: Center of mass in radians (-pi to pi), centered at the reward location.
    """
    # Convert com and reward_location to radians
    com_radians = com * (2 * math.pi / track_length)
    reward_radians = reward_location * (2 * math.pi / track_length)

    # Center com_radians around reward_radians
    centered_com_radians = com_radians - reward_radians

    # Wrap the centered_com_radians to -pi to pi range
    while centered_com_radians > math.pi:
        centered_com_radians -= 2 * math.pi
    while centered_com_radians < -math.pi:
        centered_com_radians += 2 * math.pi

    return centered_com_radians
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

def consecutive_stretch_time(x, tol=2):
    """note that the tol is based on approx how long
    it takes the mouse to return to rew loc
    on a 2.7m track
    i.e. the mouse cannot return to rew loc at 1.2s

    Args:
        x (_type_): _description_
        tol (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    # Calculate differences
    z = np.diff(x)
    # Find break points based on the tolerance
    break_point = np.where(z > tol)[0]

    if len(break_point) == 0:
        return [x.tolist()]  # If there are no break points, return the entire array as a single stretch

    result = []

    # Add the first stretch
    first_stretch = x[:break_point[0] + 1]
    if len(first_stretch) == 1:
        result.append(first_stretch[0])
    else:
        result.append(first_stretch.tolist())

    # Add the middle stretches
    for i in range(1, len(break_point)):
        stretch = x[break_point[i - 1] + 1:break_point[i] + 1]
        if len(stretch) == 1:
            result.append(stretch[0])
        else:
            result.append(stretch.tolist())

    # Add the last stretch
    last_stretch = x[break_point[-1] + 1:]
    if len(last_stretch) == 1:
        result.append(last_stretch[0])
    else:
        result.append(last_stretch.tolist())

    return result

def consecutive_stretch(x):
    z = np.diff(x)
    break_points = np.where(z != 1)[0]
    
    if len(break_points) == 0:
        return [x]
    
    y = []
    y.append(x[:break_points[0] + 1])
    
    for i in range(1, len(break_points)):
        y.append(x[break_points[i-1] + 1 : break_points[i] + 1])
    
    y.append(x[break_points[-1] + 1 :])
    
    return y

def find_differentially_activated_cells(tuning_curve1, tuning_curve2, threshold, binsize):
    """
    Identify cells that are differentially inactivated between two conditions.
    
    Parameters:
    tuning_curve1 (np.ndarray): Tuning curve for condition 1 (cells x bins).
    tuning_curve2 (np.ndarray): Tuning curve for condition 2 (cells x bins).
    threshold (float): The threshold for considering a cell differentially inactivated.
    
    Returns:
    np.ndarray: Indices of cells considered differentially inactivated.
    """
    # Calculate the AUC across bins for each cell in each condition
    auc_tc1 = []; auc_tc2 = []
    for cll in range(tuning_curve1.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve1[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc1.append(np.nanquantile([np.trapz(tuning_curve1[cll,tr],dx=binsize) for tr in transients],0.75))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanquantile([np.trapz(tuning_curve2[cll,tr],dx=binsize) for tr in transients],0.75))
    
    mean_activity1 = np.array(auc_tc1)
    mean_activity2 = np.array(auc_tc2)
    
    # Find the difference in mean activity between conditions
    activity_diff = mean_activity1 - mean_activity2
    
    differentially_activated_cells = np.where(activity_diff < -threshold)[0]
    
    return differentially_activated_cells

def find_differentially_inactivated_cells(tuning_curve1, tuning_curve2, threshold, binsize):
    """
    Identify cells that are differentially inactivated between two conditions.
    
    Parameters:
    tuning_curve1 (np.ndarray): Tuning curve for condition 1 (cells x bins).
    tuning_curve2 (np.ndarray): Tuning curve for condition 2 (cells x bins).
    threshold (float): The threshold for considering a cell differentially inactivated.
    
    Returns:
    np.ndarray: Indices of cells considered differentially inactivated.
    """
    # Calculate the AUC across bins for each cell in each condition
    auc_tc1 = []; auc_tc2 = []
    for cll in range(tuning_curve1.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve1[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc1.append(np.nanmean([np.trapz(tuning_curve1[cll,tr],dx=binsize) for tr in transients]))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanmean([np.trapz(tuning_curve2[cll,tr],dx=binsize) for tr in transients]))
    
    mean_activity1 = np.array(auc_tc1)
    mean_activity2 = np.array(auc_tc2)
    
    # Find the difference in mean activity between conditions
    activity_diff = mean_activity1 - mean_activity2
    
    # Identify cells with a decrease in activity beyond the threshold
    differentially_inactivated_cells = np.where(activity_diff > threshold)[0]
    
    return differentially_inactivated_cells

        
def calculate_difference(tuning_curve1, tuning_curve2):
    """
    Calculate the difference between two normalized tuning curves.
    
    Parameters:
    tuning_curve1, tuning_curve2 (numpy.ndarray): The two tuning curves.
    
    Returns:
    numpy.ndarray: The difference between the tuning curves.
    """
    diff = tuning_curve1 - tuning_curve2
    return diff

def get_pyr_metrics_opto(conddf, dd, day, threshold=5, pc = False):
    track_length = 270
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    if not pc:
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
            'tuning_curves_late_trials', 'coms_early_trials', 'trialnum'])
        trialnum = fall['trialnum'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
    else:
        fall = scipy.io.loadmat(params_pth, variable_names=['coms_pc_late_trials', 'changeRewLoc', 'tuning_curves_pc_early_trials',\
            'tuning_curves_pc_late_trials', 'coms_pc_early_trials', 'trialnum'])
        trialnum = fall['trialnum'][0]
        coms = fall['coms_pc_late_trials'][0]
        coms_early = fall['coms_pc_early_trials'][0]
        tcs_early = fall['tuning_curves_pc_early_trials'][0]
        tcs_late = fall['tuning_curves_pc_late_trials'][0]
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    eps = np.append(eps, len(changeRewLoc)) 
    # exclude last ep if too little trials
    lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
    if lastrials<8:
        eps = eps[:-1]
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    bin_size = 3    
    tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0]]]))
    tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1]]]))
    tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0]]]))
    tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1]]]))    
    # replace nan coms
    # peak = np.nanmax(tc1_late,axis=1)
    # coms1_max = np.array([np.where(tc1_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
    # peak = np.nanmax(tc2_late,axis=1)
    # coms2_max = np.array([np.where(tc2_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])    
    coms1 = np.hstack(coms[comp[0]])
    coms2 = np.hstack(coms[comp[1]])
    coms1_early = np.hstack(coms_early[comp[0]])
    coms2_early = np.hstack(coms_early[comp[1]])
    
    # take fc3 in area around com
    difftc1 = tc1_late-tc1_early
    coms1_bin = np.floor(coms1/bin_size).astype(int)
    difftc1 = np.array([np.nanmean(difftc1[ii,com-2:com+2]) for ii,com in enumerate(coms1_bin)])
    difftc2 = tc2_late-tc2_early
    coms2_bin = np.floor(coms2/bin_size).astype(int)
    difftc2 = np.array([np.nanmean(difftc2[ii,com-2:com+2]) for ii,com in enumerate(coms2_bin)])

    # Find differentially inactivated cells
    differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[1]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    # differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[1]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late, tc2_late, threshold, bin_size)
    # differentially_activated_cells = find_differentially_activated_cells(tc1_late, tc2_late, threshold, bin_size)
    # tc1_pc_width = evaluate_place_field_width(tc1_late, bin_centers, threshold=0.5)
    rewloc_shift = rewlocs[comp[1]]-rewlocs[comp[0]]
    com_shift = [np.nanmean(coms[comp[1]][differentially_inactivated_cells]-coms[comp[0]][differentially_inactivated_cells]), \
                np.nanmean(coms[comp[1]][differentially_activated_cells]-coms[comp[0]][differentially_activated_cells]), \
                    np.nanmean(coms[comp[1]]-coms[comp[0]])]
    # circular alignment
    rel_coms1 = [convert_com_to_radians(com, rewlocs[comp[0]], track_length) for com in coms1]
    rel_coms2 = [convert_com_to_radians(com, rewlocs[comp[1]], track_length) for com in coms2]
    # rel_coms2 = np.hstack([(coms2[coms2<=rewlocs[comp[1]]]-rewlocs[comp[1]])/rewlocs[comp[1]],(coms2[coms2>rewlocs[comp[1]]]-rewlocs[comp[1]])/(track_length-rewlocs[comp[1]])])
    # rel_coms2 = (coms2-rewlocs[comp[1]])/rewlocs[comp[1]]
    dct['comp'] = comp
    dct['rel_coms1'] = np.array(rel_coms1)
    dct['rel_coms2'] = np.array(rel_coms2)
    dct['learning_tc1'] = [tc1_early, tc1_late]
    dct['learning_tc2'] = [tc2_early, tc2_late]
    dct['difftc1'] = difftc1
    dct['difftc2'] = difftc2
    dct['rewzones_comp'] = rewzones[comp]
    dct['coms1'] = coms1
    dct['coms2'] = coms2
    # dct['frac_place_cells_tc1'] = sum((coms1>(rewlocs[comp[0]]-5-(track_length*.2))) & (coms1<(rewlocs[comp[0]])+5+(track_length*.2)))/len(coms1[(coms1>bin_size) & (coms1<=(track_length/bin_size))])
    # dct['frac_place_cells_tc2'] = sum((coms2>(rewlocs[comp[1]]-5-(track_length*.2))) & (coms2<(rewlocs[comp[1]])+5+(track_length*.2)))/len(coms2[(coms2>bin_size) & (coms2<=(track_length/bin_size))])
    dct['frac_place_cells_tc1_late_trials'] = sum((coms1>(rewlocs[comp[0]]-5-(track_length*.2))) & (coms1<(rewlocs[comp[0]])+5+(track_length*.2)))/len(coms1[(coms1>=bin_size)])
    dct['frac_place_cells_tc2_late_trials'] = sum((coms2>(rewlocs[comp[1]]-5-(track_length*.2))) & (coms2<(rewlocs[comp[1]])+5+(track_length*.2)))/len(coms2[(coms2>=bin_size)])
    dct['frac_place_cells_tc1_early_trials'] = sum((coms1_early>(rewlocs[comp[0]]-5-(track_length*.2))) & (coms1_early<(rewlocs[comp[0]])+5+(track_length*.2)))/len(coms1_early[(coms1_early>=bin_size)])
    dct['frac_place_cells_tc2_early_trials'] = sum((coms2_early>(rewlocs[comp[1]]-5-(track_length*.2))) & (coms2_early<(rewlocs[comp[1]])+5+(track_length*.2)))/len(coms2_early[(coms2_early>=bin_size)])
    dct['rewloc_shift'] = rewloc_shift
    dct['com_shift'] = com_shift
    dct['inactive'] = differentially_inactivated_cells
    dct['active'] = differentially_activated_cells
    dct['rewlocs_comp'] = rewlocs[comp]
    return dct

def get_dff_opto(conddf, dd, day):
    track_length = 270
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms_pc_late_trials', 'changeRewLoc', 'dFF',
                    'ybinned'])
    coms = fall['coms_pc_late_trials'][0]
    dFF = fall['dFF']
    ybinned = fall['ybinned'][0]*1.5
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    eps = np.append(eps, len(changeRewLoc))  
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    dff_prev = np.nanmean(dFF[eps[comp[0]]:eps[comp[1]],:])#[ybinned[eps[comp[0]]:eps[comp[1]]]<rewlocs[comp[0]],:])
    dff_opto = np.nanmean(dFF[eps[comp[1]]:eps[comp[1]+1],:])#[ybinned[eps[comp[1]]:eps[comp[1]+1]]<rewlocs[comp[1]],:])
    return dff_opto, dff_prev
