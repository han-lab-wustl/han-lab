# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl

################################FUNCTION DEFINITIONS################################
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
    normmeanrewdFF = (meanrewdFF - np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    
    return normmeanrewdFF, meanrewdFF, rewdFF
################################FUNCTION DEFINITIONS################################
if __name__ == "__main__":

    # example on how to open the pickle file
    pdst = r"Y:\DLC\dlc_mixedmodel2\E201_25_Mar_2023_vr_dlc_align.p"
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
        
    # mpl.use('TkAgg')
    plt.figure()
    rng = np.arange(25000,30000) # restrict number of frames
    ypos =vralign['ybinned'][rng]
    plt.plot(np.array(circumferences)[rng])
    plt.plot(ypos)
    plt.plot(vralign['rewards'][rng]*50)
    plt.scatter(np.argwhere(vralign['licks'][rng]>0).T[0], \
                ypos[vralign['licks'][rng]>0], color='r', marker='.')
    plt.ylim([0,200])
    plt.xlabel('frames')
    plt.ylabel('pupil circumference')

    # run peri reward
    range_val = 8 #s
    binsize = 0.1 #s
    normmeanrew, meanrew, rewall = perireward_binned_activity(np.array(circumferences), (vralign['rewards']==0.5).astype(int), 
                            vralign['timedFF'], range_val, binsize)
    # plot perireward pupil
    fig, ax = plt.subplots()
    for rew in rewall.T:
        # plot individual trials in grey
        normrew = (rew - np.nanmin(rew)) / (np.nanmax(rew) - np.nanmin(rew))
        ax.plot(normrew, color='slategray', alpha=0.2)
    ax.axvline(np.median(np.arange(0,len(normmeanrew))), color='b', linestyle='--')
    ax.plot(normmeanrew,color='k')
    
    ax.set_ylabel('Normalized Pupil Circumference')
    ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
    ax.set_xticklabels(np.arange(-range_val,range_val+1))
    ax.set_xlabel('Time from reward (s)')
                    