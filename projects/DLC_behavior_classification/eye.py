# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas as pd
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter 
################################FUNCTION DEFINITIONS################################

def get_unrewarded_stops(vralign):
    stops = vralign['forwardvel']==0 # 0 velocity
    stop_ind = consecutive_stretch(stops)
    
def get_area_circumference_from_vralign(pdst, gainf, rewsize):
    
    # example on how to open the pickle file
    # pdst = r"Y:\DLC\dlc_mixedmodel2\E201_25_Mar_2023_vr_dlc_align.p"
    with open(pdst, "rb") as fp: #unpickle
            vralign = pickle.load(fp)
    # edit name of eye points
    eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']

    for e in eye:
        vralign[e+'_x'][vralign[e+'_likelihood'].astype('float32')<0.9]=0
        vralign[e+'_y'][vralign[e+'_likelihood'].astype('float32')<0.9]=0

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
    # run peri reward time
    range_val = 10 #s
    binsize = 0.1 #s
    normmeanrew_t, meanrew, normrewall_t, \
        rewall = perireward_binned_activity(np.array(circumferences), \
                            (vralign['rewards']==0.5).astype(int), 
                            vralign['timedFF'], range_val, binsize)
    normmeanlicks_t, meanlicks, normlickall_t, \
    lickall = perireward_binned_activity(vralign['licks'], \
                        (vralign['rewards']==0.5).astype(int), 
                        vralign['timedFF'], range_val, binsize)
    normmeanvel_t, meanvel, normvelall_t, \
    velall = perireward_binned_activity(vralign['forwardvel'], \
                        (vralign['rewards']==0.5).astype(int), 
                        vralign['timedFF'], range_val, binsize)
    # TODO: add to pickle structure
    return areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall

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
        y.append(x[break_point[i - 1] + 1:break_point[i]])
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
################################FUNCTION DEFINITIONS################################