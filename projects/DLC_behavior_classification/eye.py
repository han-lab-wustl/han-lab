# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas as pd, scipy, sys, h5py
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom your clone
import statsmodels.api as sm 
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter 
from projects.DLC_behavior_classification.preprocessing import consecutive_stretch
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
################################FUNCTION DEFINITIONS################################

def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
                - y, 1d numpy array with possible NaNs
        Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
        Example:
                # linear interpolation of NaNs
                nans, x= nan_helper(y)
                y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

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

def get_unrewarded_stops(vralign):
    stops = vralign['forwardvel']==0 # 0 velocity
    stop_ind = consecutive_stretch(stops)
    
def get_area_circumference_from_vralign(pdst, range_val, binsize):
    
    # example on how to open the pickle file
    # pdst = r"Y:\DLC\dlc_mixedmodel2\E201_25_Mar_2023_vr_dlc_align.p"
    with open(pdst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
    # edit name of eye points
    eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']

    # for e in eye: #optional: threshold points
    #     vralign[e+'_x'][vralign[e+'_likelihood'].astype('float32')<0.9]=0
    #     vralign[e+'_y'][vralign[e+'_likelihood'].astype('float32')<0.9]=0

    #eye centroids, area, perimeter
    centroids_x = []; centroids_y = []
    areas = []; circumferences = []
    for i in range(len(vralign['EyeNorthWest_y'])):
        eye_x = np.array([vralign[xx+"_x"][i] for xx in eye])
        eye_y = np.array([vralign[xx+"_y"][i] for xx in eye])
        eye_coords = np.array([eye_x, eye_y]).astype(float)
        centroid_x, centroid_y = centeroidnp(eye_coords)
        area, circumference = get_eye_features([(vralign[xx+"_x"][i], 
                                    vralign[xx+"_y"][i]) for xx in eye])
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        areas.append(area); circumferences.append(circumference)
    areas = np.array(areas); circumferences = np.array(circumferences)
    areas = scipy.signal.savgol_filter(areas,5, 2)
    centroids_x = scipy.signal.savgol_filter(centroids_x,5, 2)
    centroids_y = scipy.signal.savgol_filter(centroids_y,5, 2)
    licks_threshold = vralign['lickVoltage']<=-0.065 # manually threshold licks
    licks = scipy.signal.savgol_filter(licks_threshold,10, 2) 
    rewards = vralign['rewards']
    velocity = vralign['forwardvel']
    nans, x = nan_helper(velocity)
    velocity[nans]= np.interp(x(nans), x(~nans), velocity[~nans])
    velocity = scipy.signal.savgol_filter(velocity,10, 2)
    # calculate acceleration
    acc_ = np.diff(velocity)/np.diff(np.hstack(vralign['timedFF']))
    # pad nans
    acc=np.zeros_like(velocity)
    acc[:-1]=acc_
    acc = scipy.signal.savgol_filter(acc,10, 2)
    vralign['EyeNorthEast_y'][vralign['EyeNorthEast_likelihood']<0.75]=0 # filter out low prob
    eyelid = vralign['EyeNorthEast_y']
    eyelid = scipy.signal.savgol_filter(eyelid,10, 2)

    X = np.array([velocity, acc, licks, eyelid]).T # Predictor(s)
    X = sm.add_constant(X) # Adds a constant term to the predictor(s)
    y = areas # Outcome
    # Fit a regression model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result = model.fit()
    areas_res = result.resid_pearson
    y = centroids_x # Outcome
    # Fit a regression model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result = model.fit()
    centroids_x_res = result.resid_pearson
    y = centroids_y # Outcome
    # Fit a regression model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result = model.fit()
    centroids_y_res = result.resid_pearson
    ############## GLM ##############
    # run peri reward time & plot
    input_peri = areas_res
    rewards = vralign["rewards"]
    normmeanrew_t, meanrew, normrewall_t, \
    rewall = perireward_binned_activity(np.array(input_peri), \
                            rewards.astype(int), 
                            vralign['timedFF'], range_val, binsize)

    normmeanlicks_t, meanlicks, normlickall_t, \
    lickall = perireward_binned_activity(licks_threshold, \
                    rewards.astype(int), 
                    vralign['timedFF'], range_val, binsize)
    normmeanvel_t, meanvel, normvelall_t, \
    velall = perireward_binned_activity(velocity, \
                    rewards.astype(int), 
                    vralign['timedFF'], range_val, binsize)

    
    return areas, areas_res, circumferences, centroids_x_res, centroids_y_res, \
    meanrew, rewall, meanlicks, meanvel

def get_pose_tuning_curve(pth, vralign, pose, gainf, rewsize, \
    pose_name, savedst, success=True):
    # get ep 1 all trials averaged (successes/fails?)
    # essentially make a tuning curve    
    try:
        licks = np.hstack(vralign['licks'])
    except:
        licks = vralign['licks']
    vel = np.hstack(vralign['forwardvel'])
    eps_ = np.where(vralign['changeRewLoc']>0)[0]
    eps = np.zeros(len(eps_)+1); eps[:-1] = eps_
    eps[len(eps_)] = len(vralign['changeRewLoc'])    
    color_traces = ['royalblue', 'navy', 
                    'indigo', 'mediumorchid', 'k']
    for i in range(len(eps)-1):
        rangeep = np.arange(eps[i], eps[i+1]).astype(int)
        trialnum = vralign['trialnum'][rangeep]
        rewards = vralign['rewards'][rangeep]
        # types of trials, success vs. fail
        s_tr = []; f_tr = []
        for tr in np.unique(trialnum[trialnum>=3]):
            if sum(rewards[trialnum==tr])>0:
                s_tr.append(tr)
            else:
                f_tr.append(tr)
        trm_f = np.isin(trialnum,f_tr)    
        trm = np.isin(trialnum,s_tr)   
        rng_s = rangeep[np.hstack(trm)] 
        rng_f = rangeep[np.hstack(trm_f)] 
        if success:
            pose_ = pose[rng_s.astype(int)]
            licks_ = licks[rng_s]
            try:
                ypos = np.hstack(vralign['ybinned'])[rng_s]
            except:
                ypos = vralign['ybinned'][rng_s]
            vel_ = vel[rng_s]
        else:
            pose_ = pose[rng_f.astype(int)]
            ypos = vralign['ybinned'][rng_f]
            licks_ = np.hstack(licks[rng_f])
            vel_ = vel[rng_f]
        ypos[ypos<3]=0
        ypos=(ypos*(gainf)).astype(int)
        df = pd.DataFrame(np.array([pose_, licks_, vel_, ypos]).T)
        # takes median and normalizes
        df_ = df.groupby(3).mean()
        circ_ep = np.hstack(df_[0].values)
        licks_ep = np.hstack(df_[1].values)
        vel_ep = np.hstack(df_[2].values)
        rewloc = np.hstack(vralign['changeRewLoc'])[int(eps[i])]*(gainf)    
        # Create a Rectangle patch
        normcirc_ep = gaussian_filter((circ_ep-np.min(circ_ep))/(np.max(circ_ep)-np.min(circ_ep)),1)        
        fig, ax = plt.subplots()
        ax.plot(normcirc_ep, color=color_traces[i])
        rect = patches.Rectangle((rewloc-rewsize/2, 0), rewsize, 1, linewidth=1, edgecolor='k', 
                    facecolor=color_traces[i], alpha=0.3)
        ax.add_patch(rect)
        ax.plot(licks_ep, color='r', linestyle='dotted', label='Licks')        
        vel_ep = gaussian_filter((vel_ep-np.min(vel_ep))/(np.max(vel_ep)-np.min(vel_ep)),1)        
        ax.plot(vel_ep, color='slategray', linestyle='dashed', label = 'Velocity')        
        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(0,max(ypos)+5,90))
        ax.set_xticklabels(np.arange(0,max(ypos)+5,90))
        ax.set_xlabel('Track Position (cm)')
        ax.set_ylabel(f'Normalized {pose_name}')
        ax.set_title(f'Mouse: {pth}, epoch: {i+1}')
        ax.legend()
        plt.show()
        fig.savefig(os.path.join(savedst, f'{pth[:-2]}_pose_tuning_ep{i+1}.svg'),
            bbox_inches = 'tight', transparent = True)

    return os.path.join(savedst, f'{pth[:-2]}_pose_tuning_ep{i+1}.svg')

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_eye_features(eye_coords, eyelbl = False):
    # eye coords format = list of (x,y) tuples
    img = Image.new('L', (600, 422), 0) # L is imagetype, 600, 422 is image dim

    ImageDraw.Draw(img).polygon(eye_coords, outline=1, fill=1)
    mask = np.array(img)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)  # Area of first contour
    perimeter = cv2.arcLength(cnt, True)  # Perimeter of first contour 

    return area, perimeter

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


def perireward_binned_activity(dFF, rewards, timedFF, range_val, binsize):
    """adaptation of gerardo's code to align IN BOTH TIME AND POSITION, dff or pose data to 
    rewards within a certain window

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    Rewindx = np.where(rewards)[0]
    rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(Rewindx)))*np.nan

    for rr in range(0,len(Rewindx)):
        rewtime = timedFF[Rewindx[rr]]
        currentrewchecks = np.where((timedFF > rewtime - range_val) & (timedFF <= rewtime + range_val))[0]
        currentrewcheckscell = consecutive_stretch(currentrewchecks) # gets consecutive stretch of reward ind
        # check for missing vals
        currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx)>0]
        currentrewcheckscell = np.array(currentrewcheckscell) # reformat for py
        currentrewardlogical = np.array([sum(Rewindx[rr]==x).astype(bool) for x in currentrewcheckscell])
        val = 0
        for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
            val = bin_val+1
            currentidxt = np.where((timedFF>(rewtime - range_val + (val * binsize) - binsize)) & (timedFF <= rewtime - range_val + val * binsize))[0]
            checks = consecutive_stretch(currentidxt)
            checks = [list(xx) for xx in checks]
            if len(checks[0]) > 0:
                currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                for x in checks])
                for i,cidx in enumerate(currentidxlogical):
                    cidx = [bool(xx) for xx in cidx]
                    if sum(cidx)>0:
                        checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                        rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

    meanrewdFF = np.nanmean(rewdFF, axis=1)    
    # allbins = np.array([round(-range_val + bin_val * binsize - binsize, 13) for bin_val in range(int(np.ceil(range_val * 2 / binsize)))])
    normmeanrewdFF = (meanrewdFF-np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    normrewdFF = np.array([(xx-np.min(xx))/((np.max(xx)-np.min(xx))) for xx in rewdFF.T])
    return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF

def get_area_circumference_from_vralign_with_fails(pdst, vrfl,
                    range_val, binsize,fs=62.5):

    with open(pdst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
    f = h5py.File(vrfl,'r')  #need to save vrfile with -v7.3 tag for this to work
    VR = f['VR']
    # edit name of eye points
    eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest',
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']
    # for e in eye: #optional: threshold points
    #     vralign[e+'_x'][vralign[e+'_likelihood'].astype('float32')<0.9]=0
    #     vralign[e+'_y'][vralign[e+'_likelihood'].astype('float32')<0.9]=0
    #eye centroids, area, perimeter
    centroids_x = []; centroids_y = []
    areas = []; circumferences = []
    for i in range(len(vralign['EyeNorthWest_y'])):
        eye_x = np.array([vralign[xx+"_x"][i] for xx in eye])
        eye_y = np.array([vralign[xx+"_y"][i] for xx in eye])
        eye_coords = np.array([eye_x, eye_y]).astype(float)
        centroid_x, centroid_y = centeroidnp(eye_coords)
        area, circumference = get_eye_features([(vralign[xx+"_x"][i],
                                    vralign[xx+"_y"][i]) for xx in eye])
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        areas.append(area); circumferences.append(circumference)
    areas = np.array(areas); circumferences = np.array(circumferences)
    areas = scipy.signal.savgol_filter(areas,5, 2)
    centroids_x = scipy.signal.savgol_filter(centroids_x,5, 2)
    centroids_y = scipy.signal.savgol_filter(centroids_y,5, 2)
    licks_threshold = vralign['lickVoltage']<=-0.065 # manually threshold licks
    licks = scipy.signal.savgol_filter(licks_threshold,10, 2)
    rewards = vralign['rewards']
    velocity = vralign['forwardvel']
    nans, x = nan_helper(velocity)
    velocity[nans]= np.interp(x(nans), x(~nans), velocity[~nans])
    velocity = scipy.signal.savgol_filter(velocity,10, 2)
    # calculate acceleration
    acc_ = np.diff(velocity)/np.diff(np.hstack(vralign['timedFF']))
    # pad nans
    acc=np.zeros_like(velocity)
    acc[:-1]=acc_
    acc = scipy.signal.savgol_filter(acc,10, 2)
    vralign['EyeNorthEast_y'][vralign['EyeNorthEast_likelihood']<0.75]=0 # filter out low prob
    eyelid = vralign['EyeNorthEast_y']
    eyelid = scipy.signal.savgol_filter(eyelid,10, 2)
    X = np.array([velocity, acc, licks, eyelid]).T # Predictor(s)
    X = sm.add_constant(X) # Adds a constant term to the predictor(s)
    y = areas # Outcome
    ############## GLM ##############
    # Fit a regression model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result = model.fit()
    areas_res = result.resid_pearson

    # run peri reward time & plot  
    # successful trials
    rewards = vralign["rewards"]
    input_peri = areas_res
    time = vralign['timedFF']
    normmeanrew_t, meanrew, normrewall_t, \
    rewall = perireward_binned_activity(np.array(input_peri), \
                            rewards.astype(int),
                            time, range_val, binsize)
    # failed
    trialnum = vralign['trialnum']
    ybinned = vralign['ybinned']
    rewlocs = VR['changeRewLoc'][:][VR['changeRewLoc'][:]>0]
    changeRewLoc = vralign["changeRewLoc"]
    crl = consecutive_stretch_vralign(np.where(changeRewLoc>0)[0])
    crl = np.array([min(xx) for xx in crl])
    eps = np.array([xx for ii,xx in enumerate(crl[1:]) if np.diff(np.array([crl[ii],xx]))[0]>5000])
    eps = np.append(eps, 0)
    eps = np.append(eps, len(changeRewLoc))
    eps = np.sort(eps)
    rewallfail = []; meanrewfail = []
    for ep in range(len(eps)-1):
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eps[ep]:eps[ep+1]], rewards[eps[ep]:eps[ep+1]])
        failtr_bool = np.array([any(yy==xx for yy in ftr_trials) for xx in trialnum[eps[ep]:eps[ep+1]]])
        failed_trialnum = trialnum[eps[ep]:eps[ep+1]][failtr_bool]
        rews_centered = np.zeros_like(failed_trialnum)
        ypos = ybinned[eps[ep]:eps[ep+1]]
        rews_centered[(ypos[failtr_bool] >= rewlocs[ep]-5) & (ypos[failtr_bool] <= rewlocs[ep]+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[min_iind]=1
        rewards_ep = rews_centered
        # fake time var
        time_ep = np.arange(0,rewards_ep.shape[0]/fs,1/fs)
        licks_threshold_ep = licks_threshold[eps[ep]:eps[ep+1]][failtr_bool]
        velocity_ep = velocity[eps[ep]:eps[ep+1]][failtr_bool]
        input_peri = areas_res[eps[ep]:eps[ep+1]][failtr_bool]
        normmeanrew_t, meanrew_ep, normrewall_t, \
        rewall_ep = perireward_binned_activity(np.array(input_peri), \
                                rewards_ep.astype(int),
                                time_ep, range_val, binsize)
        rewallfail.append(rewall_ep.T)
        meanrewfail.append(meanrew_ep)
    rewallfail = np.concatenate(rewallfail)
    meanrewfail = np.mean(np.array(meanrewfail),axis=0)

    return areas, areas_res, circumferences, meanrew, rewall, meanrewfail, rewallfail


def get_area_circumference_opto(pdst, range_val, binsize):
    # example on how to open the pickle file
    # pdst = r"Y:\DLC\dlc_mixedmodel2\E201_25_Mar_2023_vr_dlc_align.p"
    with open(pdst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
    # edit name of eye points
    eye_pnts = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']

    #eye centroids, area, perimeter
    centroids_x = []; centroids_y = []
    areas = []; circumferences = []
    for i in range(len(vralign['EyeNorthWest_y'])):
        eye_x = np.array([vralign[xx+"_x"][i] for xx in eye_pnts])
        eye_y = np.array([vralign[xx+"_y"][i] for xx in eye_pnts])
        eye_coords = np.array([eye_x, eye_y]).astype(float)
        centroid_x, centroid_y = centeroidnp(eye_coords)
        area, circumference = get_eye_features([(vralign[xx+"_x"][i], 
                                    vralign[xx+"_y"][i]) for xx in eye_pnts])
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        areas.append(area); circumferences.append(circumference)
    areas = np.array(areas); circumferences = np.array(circumferences)
    areas = scipy.signal.savgol_filter(areas,5, 2)
    centroids_x = scipy.signal.savgol_filter(centroids_x,5, 2)
    centroids_y = scipy.signal.savgol_filter(centroids_y,5, 2)
    licks_threshold = vralign['lickVoltage']<=-0.065 # manually threshold licks
    licks = scipy.signal.savgol_filter(licks_threshold,10, 2) 
    rewards = vralign['rewards']
    velocity = vralign['forwardvel']
    nans, x = nan_helper(velocity)
    velocity[nans]= np.interp(x(nans), x(~nans), velocity[~nans])
    velocity = scipy.signal.savgol_filter(velocity,10, 2)
    # calculate acceleration
    acc_ = np.diff(velocity)/np.diff(np.hstack(vralign['timedFF']))
    # pad nans
    acc=np.zeros_like(velocity)
    acc[:-1]=acc_
    acc = scipy.signal.savgol_filter(acc,10, 2)
    vralign['EyeNorthEast_y'][vralign['EyeNorthEast_likelihood']<0.75]=0 # filter out low prob
    eyelid = vralign['EyeNorthEast_y']
    eyelid = scipy.signal.savgol_filter(eyelid,10, 2)

    X = np.array([velocity, licks, eyelid]).T # Predictor(s)
    X = sm.add_constant(X) # Adds a constant term to the predictor(s)
    y = areas # Outcome
    # Fit a regression model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    result = model.fit()
    areas_res = result.resid_pearson
    
    return areas, areas_res

def consecutive_stretch_vralign(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]
    y = []
    start = 0
    
    for i in range(len(break_point) + 1):
        if i == len(break_point):
            end = len(x)
        else:
            end = break_point[i] + 1
        
        stretch = x[start:end]
        y.append(stretch)
        start = end
    
    return y

################################FUNCTION DEFINITIONS################################