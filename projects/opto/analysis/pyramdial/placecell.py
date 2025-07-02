import numpy as np, math, scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import pearsonr, ranksums
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
from scipy.signal.windows import gaussian
from scipy.ndimage import label
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, \
        intersect_arrays, make_tuning_curves_radians_by_trialtype, make_tuning_curves_radians_by_trialtype_early,\
            make_tuning_curves_by_trialtype_w_darktime_early
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
from itertools import combinations

def get_inactivated_cells_hist(dd, day, conddf):
    dct = {}
    threshold = 5
    track_length = 270
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials', 'trialnum', 'rewards', 'VR', 'ybinned','iscell',
        'licks', 'forwardvel', 'bordercells'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    rewards = fall['rewards'][0]; lick = fall['licks'][0]
    ybinned = fall['ybinned'][0]; forwardvel=fall['forwardvel'][0]
    trialnum = fall['trialnum'][0]
    coms = fall['coms'][0]
    coms_early = fall['coms_early_trials'][0]
    tcs_early = fall['tuning_curves_early_trials'][0]
    tcs_late = fall['tuning_curves_late_trials'][0]
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~fall['bordercells'][0].astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~fall['bordercells'][0].astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    eps = np.append(eps, len(changeRewLoc)) 
    # # exclude last ep if too little trials
    # lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
    # if lastrials<8:
    #     eps = eps[:-1]
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    bin_size = 3    
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates
    tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0],skew>2]]))
    tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1],skew>2]]))
    tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0],skew>2]]))
    tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1],skew>2]]))    
    # replace nan coms
    # peak = np.nanmax(tc1_late,axis=1)
    # coms1_max = np.array([np.where(tc1_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
    # peak = np.nanmax(tc2_late,axis=1)
    # coms2_max = np.array([np.where(tc2_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])    
    coms1 = np.hstack(coms[comp[0]])
    coms2 = np.hstack(coms[comp[1]])
    coms1_early = np.hstack(coms_early[comp[0]])
    coms2_early = np.hstack(coms_early[comp[1]])
    tuning_curve1=tc1_late[:, :int(rewlocs[comp[0]]/bin_size)]
    tuning_curve2=tc2_late[:, :int(rewlocs[comp[1]]/bin_size)]
    # Calculate the AUC across bins for each cell in each condition
    auc_tc1 = []; auc_tc2 = []
    for cll in range(tuning_curve1.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve1[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc1.append(np.nanmean([np.trapz(tuning_curve1[cll,tr],dx=bin_size) for tr in transients]))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanmean([np.trapz(tuning_curve2[cll,tr],dx=bin_size) for tr in transients]))
    mean_activity1 = np.array(auc_tc1)
    mean_activity2 = np.array(auc_tc2)
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          

    # Find the difference in mean activity between conditions
    activity_diff = mean_activity1 - mean_activity2
    dct['comp'] = comp
    dct['learning_tc1'] = [tc1_early, tc1_late]
    dct['learning_tc2'] = [tc2_early, tc2_late]
    dct['tcs_radian_alignment'] = tcs_correct
    dct['coms_radian_alignment'] = coms_correct
    dct['rewzones'] = rewzones
    dct['coms1'] = coms1
    dct['coms2'] = coms2
    dct['frac_place_cells_tc1_late_trials'] = sum((coms1>(rewlocs[comp[0]]-5-(track_length*.2))) & (coms1<(rewlocs[comp[0]])+5+(track_length*.2)))/len(coms1[(coms1>=bin_size)])
    dct['frac_place_cells_tc2_late_trials'] = sum((coms2>(rewlocs[comp[1]]-5-(track_length*.2))) & (coms2<(rewlocs[comp[1]])+5+(track_length*.2)))/len(coms2[(coms2>=bin_size)])
    dct['frac_place_cells_tc1_early_trials'] = sum((coms1_early>(rewlocs[comp[0]]-5-(track_length*.2))) & (coms1_early<(rewlocs[comp[0]])+5+(track_length*.2)))/len(coms1_early[(coms1_early>=bin_size)])
    dct['frac_place_cells_tc2_early_trials'] = sum((coms2_early>(rewlocs[comp[1]]-5-(track_length*.2))) & (coms2_early<(rewlocs[comp[1]])+5+(track_length*.2)))/len(coms2_early[(coms2_early>=bin_size)])
    dct['rewlocs'] = rewlocs
    dct['activity_diff']=activity_diff
    dct['skew']=skew # if want to apply skew
    return dct

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


def make_tuning_curves_radians(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90):
    rates = []; tcs_early = []; tcs_late = []; coms = []    
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        rewloc = rewlocs[ep]
        relpos = rad[eprng]        
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        F = Fc3[eprng,:]            
        moving_middle,stop = get_moving_time(forwardvel[eprng], 2, 31.25, 10)
        F = F[moving_middle,:]
        relpos = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            mask = trialnum[eprng][moving_middle]>ttr[-lasttr]
            F = F[mask,:]
            relpos = relpos[mask]                
            tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
            com = calc_COM_EH(tc,bin_size)
            tcs_late.append(tc)
            coms.append(com)

    return rates,tcs_late, coms

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
        moving_middle,stop = get_moving_time(forwardvel[eprng], 2, 31.25, 10)
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
    
def get_moving_time(velocity, thres, Fs, ftol):
    """
    It returns time points when the animal is considered moving based on animal's change in y position.
    velocity - forward velocity
    thres - Threshold speed in cm/s
    Fs - number of frames length minimum to be considered stopped.
    ftol - 10 frames
    """
    vr_speed = np.array(velocity)
    vr_thresh = thres
    moving = np.where(vr_speed > vr_thresh)[0]
    stop = np.where(vr_speed <= vr_thresh)[0]

    stop_time_stretch, num_features = label(np.diff(stop) == 1)
    stop_time_stretch = [np.where(stop_time_stretch == i)[0] for i in range(1, num_features + 1)]

    stop_time_length = [len(stretch) for stretch in stop_time_stretch]
    delete_idx = [i for i, length in enumerate(stop_time_length) if length < Fs]
    stop_time_stretch = [stretch for i, stretch in enumerate(stop_time_stretch) if i not in delete_idx]

    if len(stop_time_stretch) > 0:
        for s in range(len(stop_time_stretch) - 1):
            d = 1
            while s + d < len(stop_time_stretch):
                if not np.isnan(stop_time_stretch[s + d]).all():
                    if abs(stop_time_stretch[s][-1] - stop_time_stretch[s + d][0]) <= ftol:
                        stop_time_stretch[s] = np.concatenate([stop_time_stretch[s], np.arange(stop_time_stretch[s][-1] + 1, stop_time_stretch[s + d][0]), stop_time_stretch[s + d]])
                        stop_time_stretch[s + d] = np.array([np.nan])
                        d += 1
                    else:
                        break
                else:
                    break
        
        stop_time_stretch = [stretch for stretch in stop_time_stretch if not np.isnan(stretch).all()]
        stop = np.concatenate(stop_time_stretch).astype(int)
        moving_time = np.ones(len(vr_speed), dtype=int)
        moving_time[stop] = 0
    else:
        moving_time = np.ones(len(vr_speed), dtype=int)

    moving = np.where(moving_time == 1)[0]
    moving_middle = moving

    return moving_middle, stop

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
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = [x[:break_point[0]]]
    for i in range(1, len(break_point)):
        xx = x[break_point[i - 1] + 1:break_point[i]]
        y.append(xx)
    y.append(x[break_point[-1] + 1:])
    
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
    com_shift = [np.nanmean(coms2[differentially_inactivated_cells]-coms1[differentially_inactivated_cells]), \
                np.nanmean(coms2[differentially_activated_cells]-coms1[differentially_activated_cells]), \
                    np.nanmean(coms2-coms1)]
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

def get_dff_opto(conddf, dd, pc=True):
    """
    get pre-reward dff on opto vs. ctrl epochs
    """    
    animal = conddf.animals.values[dd]
    day = conddf.days.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'Fc3','VR','ybinned', 'iscell', 'bordercells', 'putative_pcs'])
    dFF = fall['Fc3']
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    pcs = np.array([np.squeeze(xx) for xx in fall['putative_pcs'][0]])
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # if animal!='z14' and animal!='e200' and animal!='e189':                    
    # place cells only
    if pc: dFF = dFF[:, (np.sum(pcs,axis=0)>0)]
    skew=skew[(np.sum(pcs,axis=0)>0)]
    dFF = dFF[:, skew>2] 
    # else: dFF = dFF[:, ~(np.sum(pcs,axis=0)>0)]
    ybinned = fall['ybinned'][0]/scalingf
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]/scalingf
    rewzones = get_rewzones(rewlocs, 1/scalingf)
    eps = np.append(eps, len(changeRewLoc))  
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare, python indexing   
    # dff_prev = np.nanmean(dFF[eps[comp[0]]:eps[comp[1]],:][ybinned[eps[comp[0]]:eps[comp[1]]]<rewlocs[comp[0]]-rewsize/2,:])
    # dff_opto = np.nanmean(dFF[eps[comp[1]]:eps[comp[1]+1],:][ybinned[eps[comp[1]]:eps[comp[1]+1]]<rewlocs[comp[1]]-rewsize/2,:])
    # get overall activity
    # per cell
    dff_prev = dFF[eps[comp[0]]:eps[comp[1]],:]
    dff_opto = dFF[eps[comp[1]]:eps[comp[1]+1],:]

    return dff_opto, dff_prev

def get_rew_cells_opto_dff(params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
    radian_alignment, cm_window=20):   
    if animal=='e145': pln=2  
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
    print(params_pth)

    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
    'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
    time = fall['timedFF'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        time=time[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    # only test opto vs. ctrl
    eptest = conddf.optoep.values[ii]
    if conddf.optoep.values[ii]<2: 
            eptest = random.randint(2,3)   
            if len(eps)<4: eptest = 2 # if no 3 epochs 
    eptest=int(eptest)   
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins

    if sum([f'{animal}_{day:03d}' in xx for xx in list(radian_alignment_saved.keys())])>0:
        k = [xx for xx in radian_alignment_saved.keys() if f'{animal}_{day:03d}' in xx][0]
        print(k)
        tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early = radian_alignment_saved[k]            
    else:# remake tuning curves relative to reward        
    # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3_org = fall_fc3['Fc3']
        dFF_org = fall_fc3['dFF']
        Fc3_org = Fc3_org[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF_org = dFF_org[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF_org, nan_policy='omit', axis=0)
        dFF=dFF_org[:, skew>2]
        Fc3=Fc3_org[:, skew>2]
        # low cells
        if animal=='e217' or animal=='z17' or animal=='z14' or animal=='e200':
            dFF=dFF_org[:, skew>1.2]
            Fc3=Fc3_org[:, skew>1.2]
        # tc w/ dark time
        print('making tuning curves...\n')
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,dFF,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt,lasttr=8) 
        # early tc
        tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,dFF,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)        
    goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
    rz = get_rewzones(rewlocs,1/scalingf) 
    #     return {
    #     'goal_cell_prop': goal_cell_p,
    #     'epoch_perm': perm,
    #     'goal_cell_shuf_ps': goal_cell_shuf_ps
    # }
    results_pre_early = process_goal_cell_proportions(eptest, 
    cell_type='pre',
    coms_correct=coms_correct_early,
    tcs_correct=tcs_correct_early,
    rewlocs=rewlocs,
    animal=animal,
    day=day,
    pdf=pdf,
    rz=rz,
    scalingf=scalingf,
    bins=bins,
    goal_window=goal_window
    )

    results_post_early = process_goal_cell_proportions(eptest, 
        cell_type='post',
        coms_correct=coms_correct_early,
        tcs_correct=tcs_correct_early,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins,
        goal_window=goal_window
    )
    results_pre = process_goal_cell_proportions(eptest, 
        cell_type='pre',
        coms_correct=coms_correct,
        tcs_correct=tcs_correct,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins,
        goal_window=goal_window
    )
    results_post = process_goal_cell_proportions(eptest, 
        cell_type='post',
        coms_correct=coms_correct,
        tcs_correct=tcs_correct,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins,
        goal_window=goal_window
    )
    # save 
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early]
    return radian_alignment, results_pre, results_post, results_pre_early, results_post_early

def get_rew_cells_opto(params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
    radian_alignment, cm_window=20):   
    if animal=='e145': pln=2  
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
    print(params_pth)

    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
    'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
    time = fall['timedFF'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        time=time[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    # only test opto vs. ctrl
    eptest = conddf.optoep.values[ii]
    if conddf.optoep.values[ii]<2: 
            eptest = random.randint(2,3)   
            if len(eps)<4: eptest = 2 # if no 3 epochs 
    eptest=int(eptest)   
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length

    if sum([f'{animal}_{day:03d}' in xx for xx in list(radian_alignment_saved.keys())])>0:
        k = [xx for xx in radian_alignment_saved.keys() if f'{animal}_{day:03d}' in xx][0]
        print(k)
        tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early = radian_alignment_saved[k]            
    else:# remake tuning curves relative to reward        
    # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # if animal!='z14' and animal!='e200' and animal!='e189':                
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
        skew_thres_range=np.arange(0,1.6,0.1)[::-1]
        iii=0
        while Fc3.shape[1]==0:      
            iii+=1
            print('************************0 cells skew > 2************************')
            Fc3 = fall_fc3['Fc3']                        
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
            Fc3 = Fc3[:, skew>skew_thres_range[iii]]
        # 9/19/24
        # find correct trials within each epoch!!!!
        # tc w/ dark time
        print('making tuning curves...\n')
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt,lasttr=8) 
        # early tc
        tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)        
    goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
    rz = get_rewzones(rewlocs,1/scalingf) 
    #     return {
    #     'goal_cell_prop': goal_cell_p,
    #     'epoch_perm': perm,
    #     'goal_cell_shuf_ps': goal_cell_shuf_ps
    # }
    results_pre_early = process_goal_cell_proportions(eptest, 
    cell_type='pre',
    coms_correct=coms_correct_early,
    tcs_correct=tcs_correct_early,
    rewlocs=rewlocs,
    animal=animal,
    day=day,
    pdf=pdf,
    rz=rz,
    scalingf=scalingf,
    bins=bins_dt,
    goal_window=goal_window
    )

    results_post_early = process_goal_cell_proportions(eptest, 
        cell_type='post',
        coms_correct=coms_correct_early,
        tcs_correct=tcs_correct_early,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins_dt,
        goal_window=goal_window
    )
    results_pre = process_goal_cell_proportions(eptest, 
        cell_type='pre',
        coms_correct=coms_correct,
        tcs_correct=tcs_correct,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins_dt,
        goal_window=goal_window
    )
    results_post = process_goal_cell_proportions(eptest, 
        cell_type='post',
        coms_correct=coms_correct,
        tcs_correct=tcs_correct,
        rewlocs=rewlocs,
        animal=animal,
        day=day,
        pdf=pdf,
        rz=rz,
        scalingf=scalingf,
        bins=bins_dt,
        goal_window=goal_window
    )
    # save 
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early]
    return radian_alignment, results_pre, results_post, results_pre_early, results_post_early
    
def process_goal_cell_proportions(
    eptest, cell_type, coms_correct, tcs_correct, rewlocs,
    animal, day, pdf, rz, scalingf, bins, goal_window, epsilon=0.7,
    num_iterations=1000,bound=np.pi/2,allep=False
):
    """
    near pre and all post
    Parameters:
        eptest: tuple of two epoch indices (ctrl_ep, opto_ep)
        cell_type: 'pre' or 'post'
        coms_correct: array of shape (n_epochs, n_cells)
        tcs_correct: tuning curves (n_epochs, n_cells, bins)
        rewlocs: reward location per epoch
        goal_window: window for goal definition (radians)
        epsilon: wraparound window around π
        pdf: PDF object for saving figures (optional)

    Returns:
        dict with:
            - goal_cell_prop: real goal cell proportion
            - goal_cell_shuf_ps: list of shuffled proportions
    """
    from numpy import abs, pi
    from itertools import combinations

    # only get opto vs. ctrl epoch comparisons
    # change to relative value 
    coms_correct=coms_correct[[eptest-2,eptest-1],:]
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    print(perm)
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
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

    # Select goal COMs based on cell_type
    if cell_type == 'pre':
        com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:, xx], axis=0) <= 0)&(np.nanmedian(coms_rewrel[:, xx], axis=0)> -bound))] if len(com) > 0 else [] for com in com_goal]
    elif cell_type == 'post':
        com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:, xx], axis=0) > 0)] if len(com) > 0 else [] for com in com_goal]
    else:
        raise ValueError("cell_type must be 'pre' or 'post'")
    # Get goal cells across epochs
    com_goal_postrew=[xx for xx in com_goal_postrew if len(xx) > 0]
    goal_cells = intersect_arrays(*com_goal_postrew) if len(com_goal_postrew) > 0 else []
    goal_cell_p = len(goal_cells) / len(coms_correct[0])
    perm = list(combinations(range(len(coms_correct)), 2))

    # Optional plot of tuning curves
    if len(goal_cells) > 0:
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30, 30), sharex=True)
        axes = axes.flatten() if len(goal_cells) > 1 else [axes]
        for i, gc in enumerate(goal_cells):
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep, gc, :], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins / 2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('$\Theta$')
        ax.set_ylabel('Fc3')
        fig.suptitle(f'{animal}, day {day}')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return {
        'goal_cell_prop': goal_cell_p,
        'epoch_perm': perm,
        'goal_id': goal_cells
    }

def get_main_rew_cells_opto(params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, radian_alignment, cm_window=20):   
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
    'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
    time = fall['timedFF'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        time=time[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    # only test opto vs. ctrl
    eptest = conddf.optoep.values[ii]
    if conddf.optoep.values[ii]<2: 
            eptest = random.randint(2,3)   
            if len(eps)<4: eptest = 2 # if no 3 epochs 
    eptest=int(eptest)   
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins

    if sum([f'{animal}_{day:03d}' in xx for xx in list(radian_alignment_saved.keys())])>0:
        k = [xx for xx in radian_alignment_saved.keys() if f'{animal}_{day:03d}' in xx][0]
        print(k)
        tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early = radian_alignment_saved[k]            
    else:# remake tuning curves relative to reward        
    # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # if animal!='z14' and animal!='e200' and animal!='e189':                
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
        skew_thres_range=np.arange(0,1.6,0.1)[::-1]
        iii=0
        while Fc3.shape[1]==0:      
            iii+=1
            print('************************0 cells skew > 2************************')
            Fc3 = fall_fc3['Fc3']                        
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
            Fc3 = Fc3[:, skew>skew_thres_range[iii]]
        # 9/19/24
        # find correct trials within each epoch!!!!
        # tc w/ dark time
        print('making tuning curves...\n')
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt,lasttr=8) 
        # early tc
        tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)        
    goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
    # only get opto vs. ctrl epoch comparisons
    # change to relative value 
    # save 
    goal_cells_early = get_goal_cell_ind(coms_correct_early,eptest,goal_window)
    goal_cells = get_goal_cell_ind(coms_correct,eptest,goal_window)
    # average pre-reward activity
    rng_prev=np.arange(eps[eptest-2],eps[eptest-1])
    rng_opto=np.arange(eps[eptest-1],eps[eptest])
    dff_p=np.nanmean(Fc3[rng_prev[ybinned[rng_prev]<rewlocs[eptest-2]-rewsize/2],:][:,goal_cells],axis=0)
    dff_o=np.nanmean(Fc3[rng_opto[ybinned[rng_opto]<rewlocs[eptest-1]-rewsize/2],:][:,goal_cells],axis=0)
    dff_p_e=np.nanmean(Fc3[rng_prev[ybinned[rng_prev]<rewlocs[eptest-2]-rewsize/2],:][:,goal_cells_early],axis=0)
    dff_o_e=np.nanmean(Fc3[rng_opto[ybinned[rng_opto]<rewlocs[eptest-1]-rewsize/2],:][:,goal_cells_early],axis=0)

    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, goal_cells_early, goal_cells,dff_p,dff_o,dff_p_e,dff_o_e]
    return radian_alignment

def get_goal_cell_ind(coms_correct,eptest,goal_window,bound=np.pi/4):
    coms_correct=coms_correct[[eptest-2,eptest-1],:]
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    print(perm)
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

    # Select goal COMs based on cell_type
    # near pre and post
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:, xx], axis=0)> -bound))] if len(com) > 0 else [] for com in com_goal]
    # Get goal cells across epochs
    com_goal_postrew=[xx for xx in com_goal_postrew if len(xx) > 0]
    goal_cells = intersect_arrays(*com_goal_postrew) if len(com_goal_postrew) > 0 else []    

    return goal_cells