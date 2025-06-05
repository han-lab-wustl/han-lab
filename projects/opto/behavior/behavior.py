import numpy as np
import scipy, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def smooth_lick_rate(licks, dt, sigma_sec=0.05):
    # Convert sigma from seconds to samples
    sigma_samples = sigma_sec / dt
    # Smooth the binary lick vector
    lick_rate = gaussian_filter1d(licks.astype(float), sigma=sigma_samples)
    # Units: licks/sec
    lick_rate = lick_rate / dt
    return lick_rate

def get_lick_tuning_curves_per_trial(params_pth, conddf, dd, bin_size = 2, probes=False):    
    fall = scipy.io.loadmat(params_pth, variable_names=['VR'])
    VR = fall['VR'][0][0][()]
    eps = np.where(np.hstack(VR['changeRewLoc']>0))[0]
    eps = np.append(eps, len(np.hstack(VR['changeRewLoc'])))
    scalingf = VR['scalingFACTOR'][0][0]
    track_length = 180/scalingf
    nbins = int(track_length/bin_size)
    ybinned = np.hstack(VR['ypos']/scalingf)
    rewlocs = np.ceil(np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf).astype(int)
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
    trialnum = np.hstack(VR['trialNum'])
    rewards = np.hstack(VR['reward'])
    forwardvel = np.hstack(VR['ROE']); time =np.hstack(VR['time'])
    forwardvel=-0.013*forwardvel[1:]/np.diff(time) # make same size
    forwardvel = np.append(forwardvel, np.interp(len(forwardvel)+1, np.arange(len(forwardvel)),forwardvel))
    licks = np.hstack(VR['lickVoltage'])
    licks = licks<=-0.065 # remake boolean
    eptest = conddf.optoep.values[dd]    
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)   
        if len(eps)<4: eptest = 2 # if no 3 epochs    
    lickdf = pd.DataFrame({'lick': licks})
    lick_smooth = np.hstack(lickdf.rolling(20).mean().values)
    comp = [eptest-1] # just comparison epoch
    # get licks per trial
    # just 1 ep for now
    trialnum_eps = [trialnum[eps[ep]:eps[ep+1]] for ep in comp]
    lick_smooth_eps = [lick_smooth[eps[ep]:eps[ep+1]] for ep in comp] 
    ybinned_eps = [ybinned[eps[ep]:eps[ep+1]] for ep in comp] 
    rewards_eps = [rewards[eps[ep]:eps[ep+1]] for ep in comp] 
    # get all reward eligible trials in one list
    if not probes: # everything but probes
        licks_per_trial_per_ep = [lick_smooth_eps[ii][trialnum_ep == xx] for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx > 2]
        ybinned_per_trial_per_ep = [ybinned_eps[ii][trialnum_ep == xx] for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx > 2]
        trialstate_per_ep = [sum(rewards_eps[ii][trialnum_ep == xx]==1)>0 for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx > 2]         
        lick_tuning_curves_per_trial_per_ep = []
        for ii,lick in enumerate(licks_per_trial_per_ep):
            _, beh_prob = get_behavior_tuning_curve(ybinned_per_trial_per_ep[ii], lick, bins=nbins)
            lick_tuning_curves_per_trial_per_ep.append(beh_prob.values)
        # exclude incomplete trials
        trialstate_per_ep = [xx for ii,xx in enumerate(trialstate_per_ep) if len(lick_tuning_curves_per_trial_per_ep[ii])>70]
        lick_tuning_curves_per_trial_per_ep = [xx for xx in lick_tuning_curves_per_trial_per_ep if len(xx)>70]

        lick_tuning_curves_per_trial_per_ep_padded = np.ones((len(lick_tuning_curves_per_trial_per_ep),nbins))*np.nan
        for trial in range(len(lick_tuning_curves_per_trial_per_ep_padded)):
            lick_tuning_curves_per_trial_per_ep_padded[trial,:len(lick_tuning_curves_per_trial_per_ep[trial])] = lick_tuning_curves_per_trial_per_ep[trial]
        rewzones = get_rewzones(rewlocs,1/scalingf)
        rewzone = rewzones[comp]
        rewzone_prev = rewzones[comp[0]-1]
    else:  # get probes starting from previous epoch
        if (eptest+1)<len(eps):
            probecomp = [eptest]
            trialnum_eps = [trialnum[eps[ep]:eps[ep+1]] for ep in probecomp]
            lick_smooth_eps = [lick_smooth[eps[ep]:eps[ep+1]] for ep in probecomp] 
            ybinned_eps = [ybinned[eps[ep]:eps[ep+1]] for ep in probecomp] 
            rewards_eps = [rewards[eps[ep]:eps[ep+1]] for ep in probecomp] 
            # only first probe
            licks_per_trial_per_ep = [lick_smooth_eps[ii][trialnum_ep == xx] for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx<=2]
            ybinned_per_trial_per_ep = [ybinned_eps[ii][trialnum_ep == xx] for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx<=2]
            trialstate_per_ep = [sum(rewards_eps[ii][trialnum_ep == xx]==1)>0 for ii,trialnum_ep in enumerate(trialnum_eps) for xx in np.unique(trialnum_ep) if xx<=2]
            
            lick_tuning_curves_per_trial_per_ep = []
            for ii,lick in enumerate(licks_per_trial_per_ep):
                _, beh_prob = get_behavior_tuning_curve(ybinned_per_trial_per_ep[ii], lick, bins=nbins)
                lick_tuning_curves_per_trial_per_ep.append(beh_prob.values)
            # exclude incomplete trials
            trialstate_per_ep = [xx for ii,xx in enumerate(trialstate_per_ep) if len(lick_tuning_curves_per_trial_per_ep[ii])>70]
            lick_tuning_curves_per_trial_per_ep = [xx for xx in lick_tuning_curves_per_trial_per_ep if len(xx)>70]

            lick_tuning_curves_per_trial_per_ep_padded = np.ones((len(lick_tuning_curves_per_trial_per_ep),nbins))*np.nan
            for trial in range(len(lick_tuning_curves_per_trial_per_ep_padded)):
                lick_tuning_curves_per_trial_per_ep_padded[trial,:len(lick_tuning_curves_per_trial_per_ep[trial])] = lick_tuning_curves_per_trial_per_ep[trial]
            rewzones = get_rewzones(rewlocs,1/scalingf)
            rewzone = rewzones[comp]
            rewzone_prev = rewzones[comp[0]-1]
        else:
            lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep,rewzone, rewzone_prev = np.nan, np.nan, 0,0
                    
    return lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev
    
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
    x = np.array(x)
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x.tolist()]

    y = []
    y.append(x[:break_point[0] + 1].tolist())
    
    for i in range(1, len(break_point)):
        y.append(x[break_point[i - 1] + 1 : break_point[i] + 1].tolist())
    
    y.append(x[break_point[-1] + 1 :].tolist())

    return y

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
    
def get_mean_velocity_per_ep(forwardvel):
    return np.nanmean(forwardvel)

def lick_selectivity_probes(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, rewsize):
    # opto ep    
    eptotest = opto_ep # matlab index (+1), but remember here we are testing the next ep
    if len(eps)>eptotest+1:
        eprng = range(eps[eptotest], eps[eptotest+1])
        trialnum_ = trialnum[eprng]
        reward_ = rewards[eprng]
        licks_ = licks[eprng]
        ybinned_ = ybinned[eprng]
        forwardvel_ = forwardvel[eprng]
        rewloc = np.ceil(rewlocs[eptotest-1]).astype(int)
        rewsize = 10 # temp
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
        rate_opto = success / total_trials
        # probes
        mask = trialnum_<3
        # optional - fails
        # mask = np.array([xx in ftrials for xx in trialnum_])
        lick_selectivity_per_trial_opto = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                    licks_[mask], rewloc, rewsize,
                    fails_only = False)
        # previous ep
        eprng = range(eps[eptotest-1], eps[eptotest])
        trialnum_ = trialnum[eprng]
        reward_ = rewards[eprng]
        licks_ = licks[eprng]
        ybinned_ = ybinned[eprng]
        forwardvel_ = forwardvel[eprng]
        rewloc = np.ceil(rewlocs[eptotest-2]).astype(int)
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
        rate_prev = success / total_trials 
        trials_bwn_success_prev =  np.diff(np.array(strials))
        # probes
        mask = trialnum_<3
        # optional - fails
        # mask = np.array([xx in ftrials for xx in trialnum_])
        lick_selectivity_per_trial_prev = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                    licks_[mask], rewloc, rewsize,
                    fails_only = False)
        return lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev
    else:
        return np.nan, np.nan
    
def lick_selectivity_fails(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, rewsize):
    # opto ep    
    eptotest = opto_ep-1 # matlab index (+1)
    if len(eps)>eptotest+1:
        eprng = range(eps[eptotest], eps[eptotest+1])
        trialnum_ = trialnum[eprng]
        reward_ = rewards[eprng]
        licks_ = licks[eprng]
        ybinned_ = ybinned[eprng]
        forwardvel_ = forwardvel[eprng]
        rewloc = np.ceil(rewlocs[eptotest-1]).astype(int)
        rewsize = 10 # temp
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
        rate_opto = success / total_trials
        # probes
        mask = [xx in ftrials for xx in trialnum_]
        # optional - fails
        # mask = np.array([xx in ftrials for xx in trialnum_])
        lick_selectivity_per_trial_opto = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                    licks_[mask], rewloc, rewsize,
                    fails_only = False)
        # previous ep
        eprng = range(eps[eptotest-1], eps[eptotest])
        trialnum_ = trialnum[eprng]
        reward_ = rewards[eprng]
        licks_ = licks[eprng]
        ybinned_ = ybinned[eprng]
        forwardvel_ = forwardvel[eprng]
        rewloc = np.ceil(rewlocs[eptotest-2]).astype(int)
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
        rate_prev = success / total_trials 
        trials_bwn_success_prev =  np.diff(np.array(strials))
        # probes
        mask = [xx in ftrials for xx in trialnum_]
        # optional - fails
        # mask = np.array([xx in ftrials for xx in trialnum_])
        lick_selectivity_per_trial_prev = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                    licks_[mask], rewloc, rewsize,
                    fails_only = False)
        return lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev
    else:
        return np.nan, np.nan

def lick_selectivity_current_and_prev_reward(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, rewsize,fs=31.25):
    eptotest = opto_ep-1 # matlab index (+1)
    eprng = range(eps[eptotest], eps[eptotest+1])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    forwardvel_ = forwardvel[eprng]
    rewloc = np.ceil(rewlocs[eptotest]).astype(int)
    prevrewloc = np.ceil(rewlocs[eptotest-1]).astype(int)
    # last 5 trials?
    lasttr = 5
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_opto = success / total_trials
    mask = np.array([xx in ttr[-lasttr:] for xx in trialnum_])
    # optional - fails
    # mask = np.array([xx in ftrials for xx in trialnum_])
    lick_selectivity_per_trial_opto = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], rewloc, rewsize,
                fails_only = False)
    lick_selectivity_per_trial_opto_prevrew = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], prevrewloc, rewsize,
                fails_only = False)
    return lick_selectivity_per_trial_opto,lick_selectivity_per_trial_opto_prevrew 
    

def get_performance(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, time, rewsize,fs=31.25,lasttr = 8,firsttr = 3):
    # opto ep    
    eptotest = opto_ep-1 # matlab index (+1)
    eprng = range(eps[eptotest], eps[eptotest+1])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    licks_[np.isnan(licks_)]=0
    ybinned_ = ybinned[eprng]
    forwardvel_ = forwardvel[eprng]
    time_ = time[eprng]
    rewloc = np.ceil(rewlocs[eptotest]).astype(int)
    rewsize = 10 # temp
    # already excludes probe trials
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_opto = success / total_trials
    trials_bwn_success_opto =  np.diff(np.array(strials))
    # get lick prob
    pos_bin_opto, lick_probability_opto = get_behavior_tuning_curve(ybinned_, licks_)
    # get lick rate / trial
    # only in the first 5 trials
    mask = np.array([xx in ttr[:firsttr] for xx in trialnum_])    
    t = time_[mask][(ybinned_<rewloc)[mask]]
    dt = np.nanmedian(np.diff(t))
    lick_rate_opto = smooth_lick_rate(licks_[mask][(ybinned_<rewloc)[mask]], dt)
    # lick selectivity - only success 
    mask = np.array([xx in strials[-lasttr:] for xx in trialnum_])
    # optional - fails
    # mask = np.array([xx in ftrials for xx in trialnum_])
    lick_selectivity_per_trial_opto = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], rewloc, rewsize,
                fails_only = False)
    # late lick rate
    lick_rate_opto_late = smooth_lick_rate(licks_[mask][(ybinned_<rewloc)[mask]], dt)
    # split into pre, rew, and post
    lick_prob_opto = [lick_probability_opto[:int(rewloc-rewsize)], lick_probability_opto[int(rewloc-rewsize-10):int(rewloc+20)], \
                    lick_probability_opto[int(rewloc+20):]]
    vel_opto = get_mean_velocity_per_ep(forwardvel_[ybinned_<rewloc]) # pre-reward
    com_opto = np.nanmean(ybinned_[mask][licks_[mask].astype(bool)])-(rewloc-(rewsize/2))
    # previous ep
    eprng = range(eps[eptotest-1], eps[eptotest])
    trialnum_ = trialnum[eprng]
    reward_ = rewards[eprng]
    licks_ = licks[eprng]
    ybinned_ = ybinned[eprng]
    forwardvel_ = forwardvel[eprng]
    time_ = time[eprng]
    rewloc = np.ceil(rewlocs[eptotest-1]).astype(int)
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum_, reward_)
    rate_prev = success / total_trials 
    trials_bwn_success_prev =  np.diff(np.array(strials))
    # lick prob
    pos_bin_prev, lick_probability_prev = get_behavior_tuning_curve(ybinned_, licks_)
    # get lick rate / trial
    # only in the first 5 trials
    mask = np.array([xx in ttr[:firsttr] for xx in trialnum_])
    t = time_[mask][(ybinned_<rewloc)[mask]]
    dt = np.nanmedian(np.diff(t))
    lick_rate_prev = smooth_lick_rate(licks_[mask][(ybinned_<rewloc)[mask]], dt)
    # lick selectivity
    mask = np.array([xx in strials[-lasttr:] for xx in trialnum_])
    # optional - fails
    # mask = np.array([xx in ftrials for xx in trialnum_])
    lick_selectivity_per_trial_prev = get_lick_selectivity(ybinned_[mask], trialnum_[mask], 
                licks_[mask], rewloc, rewsize,
                fails_only = False)
    # late lick rate
    lick_rate_prev_late = smooth_lick_rate(licks_[mask][(ybinned_<rewloc)[mask]], dt)
    # split into pre, rew, and post
    lick_prob_prev = [lick_probability_prev[:int(rewloc-rewsize-20)], 
                    lick_probability_prev[int(rewloc-rewsize-20):int(rewloc+20)], \
                    lick_probability_prev[int(rewloc+20):]]
    com_prev = np.nanmean(ybinned_[mask][licks_[mask].astype(bool)])-(rewloc-(rewsize/2))
    # Return a dictionary or multiple dictionaries containing your results
    vel_prev = get_mean_velocity_per_ep(forwardvel_[ybinned_<rewloc])
    
    return rate_opto, rate_prev, lick_prob_opto, \
    lick_prob_prev, trials_bwn_success_opto, trials_bwn_success_prev, \
    vel_opto, vel_prev, lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev, \
    lick_rate_opto, lick_rate_prev, com_opto, com_prev, lick_rate_opto_late, lick_rate_prev_late


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