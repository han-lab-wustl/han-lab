

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
def circular_arc_length(start_idx, end_idx, nbins):
    """
    Return the number of bins in the forward arc from start_idx to end_idx
    on a circle of length nbins.  Always ≥1 and ≤nbins.
    """
    # if end comes after start, simple
    if end_idx >= start_idx:
        return end_idx - start_idx + 1
    # if end wrapped around past zero
    else:
        return (nbins - start_idx) + (end_idx + 1)
import numpy as np

def circular_fwhm(tc, bin_size):
    """
    Compute full‐width at half‐maximum on a circular tuning curve tc.
    Returns (width, left_cross, right_cross) where width is in same units
    as bin_size.
    """
    nbins = len(tc)
    peak = np.argmax(tc)
    half = tc[peak]*.2 # 20%

    # mask of bins ≥ half-max
    mask = tc >= half
    if not mask.any():
        return 0.0, None, None
    if mask.all():
        # everywhere above half-max
        return nbins * bin_size, 0, nbins - 1

    # double for wrap detection
    mm = np.concatenate([mask, mask])
    d = np.diff(mm.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0]

    # pick the run that contains the peak (in either copy)
    candidate = None
    for s, e in zip(starts, ends):
        if (s <= peak < e) or (s <= peak + nbins < e):
            candidate = (s, e)
            break
    if candidate is None:
        # fallback to longest run
        lengths = [e - s + 1 for s, e in zip(starts, ends)]
        idx = np.argmax(lengths)
        candidate = (starts[idx], ends[idx])

    s, e = candidate
    # map back into [0, nbins)
    left  = s  % nbins
    right = e  % nbins

    # compute forward arc length
    length_bins = circular_arc_length(left, right, nbins)
    return length_bins * bin_size, left, right
import numpy as np

def eq_rectangular_width(tc, bin_size):
    """
    Area under tc ÷ peak height → width in same units as bin_size.
    """
    peak = np.nanmax(tc)
    if peak <= 0:
        return 0.0
    area = np.trapz(tc, dx=bin_size)
    return area / peak

def get_pre_post_field_widths(params_pth,animal,day,ii,goal_window_cm=20,bins=90):
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
    lick=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # tc w/ dark time added to the end of track
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
        rewsize,ybinned,time,lick,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)  
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
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
    # all cells before 0
    com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
        xx], axis=0)<0)] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)        
    # example:
    plt.close('all')
    # rectangular method
    alldf=[]
    widths_per_ep = []
    peak_per_ep = []
    # adjusted binsize to cm?
    bin_size = track_length_dt/bins_dt
    for ep in range(len(tcs_correct)):
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        # for large fields?
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        # convert to cm
        # w = np.array(w)
        widths_per_ep.append(w)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,goal_all]])
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([goal_all]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['animal'] = [animal]*len(df)
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Pre']*len(df)
    try: # eg., suppress after validation
        ii=0
        plt.figure()
        plt.plot(tcs_correct[:,goal_all[ii],:].T)
        plt.title(f"{df.loc[df.cellid==goal_all[ii], 'width_cm'].values}")
        plt.show()
    except Exception as e:
        print(e)
    alldf.append(df)
    # ppost reward
    com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
        xx], axis=0)>0)] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)        
    widths_per_ep = []
    peak_per_ep = []
    for ep in range(len(tcs_correct)):
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        widths_per_ep.append(w)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,goal_all]])
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([goal_all]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['animal'] = [animal]*len(df)
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Post']*len(df)
    # try:
    #     ii=0
    #     plt.figure()
    #     plt.plot(tcs_correct[:,goal_all[ii],:].T)
    #     plt.title(f"{df.loc[df.cellid==goal_all[ii], 'width_cm'].values}")
    # except Exception as e:
    #     print(e)
    # plt.show()
    # add add pre and post dfs
    alldf.append(df)
    return pd.concat(alldf)