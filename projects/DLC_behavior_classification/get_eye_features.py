# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas as pd
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter 
mpl.use('TkAgg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
################################FUNCTION DEFINITIONS################################
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
        eye_x = np.array([vralign[xx+"_x"] for xx in eye])
        eye_y = np.array([vralign[xx+"_y"] for xx in eye])
        eye_coords = np.array([eye_x, eye_y]).astype(float)
        centroid_x, centroid_y = centeroidnp(eye_coords)
        area, circumference = get_eye_features([(vralign[xx+"_x"][i], 
                                    vralign[xx+"_y"][i]) for xx in eye])
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        areas.append(area); circumferences.append(circumference)
    areas = np.array(areas); circumferences = np.array(circumferences)
    # run peri reward time
    range_val = 5 #s
    binsize = 0.1 #s
    normmeanrew_t, meanrew, normrewall_t, \
        rewall = perireward_binned_activity(np.array(circumferences), \
                            (vralign['rewards']==0.5).astype(int), 
                            vralign['timedFF'], range_val, binsize)

    return areas, circumferences, normmeanrew_t, \
            normrewall_t

def get_pose_tuning_curve(vralign, pose, gainf, rewsize, \
                          pose_name):
     # get ep 1 all trials averaged (successes/fails?)
    # essentially make a tuning curve    
    eps_ = np.where(vralign['changeRewLoc']>0)[0]
    eps = np.zeros(len(eps_)+1); eps[:-1] = eps_
    eps[len(eps_)] = len(vralign['changeRewLoc'])    
    color_traces = ['royalblue', 'navy', 
                    'indigo', 'mediumorchid', 'k']
    for i in range(len(eps)-1):
        rangeep = np.arange(eps[i], eps[i+1]).astype(int)
        pose_ = pose[rangeep.astype(int)]
        ypos = vralign['ybinned'][rangeep]
        ypos[ypos<3]=0
        ypos=np.hstack((ypos*(gainf)).astype(int))  
        df = pd.DataFrame(np.array([pose_, ypos]).T)
        # takes median and normalizes
        circ_ep = np.hstack(df.groupby(1).median().values)
        rewloc = np.hstack(vralign['changeRewLoc'])[int(eps[i])]*(gainf)    
        # Create a Rectangle patch
        normcirc_ep = (circ_ep-np.min(circ_ep))/(np.max(circ_ep)-np.min(circ_ep))        
        fig, ax = plt.subplots()
        ax.plot(normcirc_ep, color=color_traces[i])
        rect = patches.Rectangle((rewloc-rewsize/2, 0), rewsize, 1, linewidth=1, edgecolor='k', 
                    facecolor=color_traces[i], alpha=0.3)
        ax.add_patch(rect)
        ax.set_ylim([0,1])
        ax.set_xticks(np.arange(0,max(ypos)+5,90))
        ax.set_xticklabels(np.arange(0,max(ypos)+5,90))
        ax.set_xlabel('Track Position (cm)')
        ax.set_ylabel(f'Normalized {pose_name}')

    return

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
    rewdFF = np.zeros((int(np.ceil(range_val * 2 / binsize)), len(Rewindx)))

    for rr in range(len(Rewindx)):
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
            currentidxt = np.where((timedFF>(rewtime - range_val + (val * binsize) - binsize)) & \
                                   (timedFF <= rewtime - range_val + val * binsize))[0]
            checks = np.array(consecutive_stretch(currentidxt))
            if len(checks[0]) > 0:
                currentidxlogical = np.array([[np.isin(x, [xx[yy] for yy in currentrewardlogical for xx in currentrewcheckscell]) for x in check] for check in checks])
                for i,cidx in enumerate(currentidxlogical):
                    if any(cidx):
                        checkidx = checks[i][cidx]
                        rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])
                    else:
                        rewdFF[bin_val, rr] = np.nan
            else:
                rewdFF[bin_val, rr] = np.nan

    meanrewdFF = np.nanmean(rewdFF, axis=1)    
    # allbins = np.array([round(-range_val + bin_val * binsize - binsize, 13) for bin_val in range(int(np.ceil(range_val * 2 / binsize)))])
    normmeanrewdFF = (meanrewdFF-np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    normrewdFF = np.array([(xx-np.min(xx))/((np.max(xx)-np.min(xx))) for xx in rewdFF.T])
    return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF
################################FUNCTION DEFINITIONS################################
if __name__ == "__main__":    

    # mpl.use('TkAgg')
    %matplotlib inline
    fig,axes = plt.subplots(3,1)    
    rng = np.arange(0,vralign['ybinned'].shape[0]) # restrict number of frames
    ypos =vralign['ybinned'][rng]*(3/2) 
    cf = np.array(circumferences)[rng]
    # filter out low vals for display
    cf[cf<130]=np.nan
    normcf = (cf-np.nanmin(cf))/(np.nanmax(cf)-np.nanmin(cf))
    toplot = gaussian_filter(normcf,
                             sigma=5)
    ax = axes[0]
    ax.plot(toplot, color='k')
    ax.set_ylabel('Normalized \n Pupil \n Circumference')
    ax.set_xticks(np.arange(0,len(rng),2000),\
            [np.ceil(xx[0]/60).astype(int) for xx in \
             vralign['timedFF'][rng][::2000]], \
                fontsize=8)
    ax = axes[1]
    ax.plot(vralign['forwardvel'][rng], color='dimgrey')
    ax.set_ylabel('Velocity (cm/s)')
    ax.set_xticks(np.arange(0,len(rng),2000),\
            [np.ceil(xx[0]/60).astype(int) \
             for xx in vralign['timedFF'][rng][::2000]], \
                fontsize=8)
    ax = axes[2]
    ax.scatter(np.arange(0, len(ypos)), ypos, color='slategray', \
                s=0.5)
    
    ax.scatter(np.argwhere(vralign['licks'][rng]>0).T[0], \
                ypos[vralign['licks'][rng]>0], color='r', marker='.',
                s=3, alpha=0.5)
    ax.scatter(np.argwhere(vralign['rewards'][rng]==0.5).T[0], \
                ypos[vralign['rewards'][rng]==0.5], color='b', marker='.')
    ax.set_ylim([0,270])
    ax.set_yticks(np.arange(0,270+1,90),np.arange(0,270+1,90))
    ax.set_xticks(np.arange(0,len(rng),2000),\
               [np.ceil(xx[0]/60).astype(int) for xx in vralign['timedFF'][rng][::2000]], \
                fontsize=8)
    ax.set_xlabel('Time in Session (minutes)')
    ax.set_ylabel('Position (cm)')
    plt.savefig(r"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\pupil_in_session.svg", \
                    bbox_inches='tight', transparent=True)

    # plot perireward pupil
    # plot perireward pupil
    fig, ax = plt.subplots()
    for rew in normrewall:
        # plot individual trials in grey
        ax.plot(rew, color='slategray', alpha=0.2)
    ax.axvline(np.median(np.arange(0,len(normmeanrew))), color='b', linestyle='--')
    ax.plot(np.nanmean(normrewall,axis=0),color='k')

    ax.set_ylabel('Normalized Pupil Circumference')
    ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
    ax.set_xticklabels(np.arange(-range_val,range_val+1))
    ax.set_xlabel('Time from reward (s)')
    plt.savefig(r"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\perirew_pupil.svg", \
                bbox_inches='tight',transparent=True)
#%%
        
# run peri reward LOCATION????????
range_val = 10 #cm
binsize = 1 #cm
normmeanrew, meanrew, rewall = perireward_binned_activity(np.array(circumferences), (vralign['rewards']==0.5).astype(int), 
                        vralign['ybinned'], range_val, binsize)

            