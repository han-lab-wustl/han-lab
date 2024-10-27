"""functions for dopamine 1 rewloc analysis
july 2024
"""
import os, scipy, numpy as np, pandas as pd, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from projects.memory.behavior import consecutive_stretch

def extract_vars(path, stims, rewloc, newrewloc, planes=4):
    planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
    params = scipy.io.loadmat(path)
    VR = params['VR'][0][0]; gainf = VR[14][0][0]             
    rewsize = VR[18][0][0][4][0][0]/gainf
    planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
    pln = int(planenum[-1])
    layer = planelut[pln]
    params_keys = params.keys()
    keys = params['params'].dtype
    # dff is in row 7 - roibasemean3/average
    dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
    # fig, ax = plt.subplots()
    # ax.plot(dff)
    # nan out stims every plane
    dff[stims[pln::planes].astype(bool)] = np.nan
    dffdf = pd.DataFrame({'dff': dff})
    dff = np.hstack(dffdf.rolling(3).mean().values)
    rewards = np.hstack(params['solenoid2'])
    trialnum = np.hstack(params['trialnum'])
    ybinned = np.hstack(params['ybinned'])/gainf
    licks = np.hstack(params['licks'])
    timedFF = np.hstack(params['timedFF'])
    # mask out dark time
    # dff = dff[ybinned>3]
    # rewards = rewards[ybinned>3]
    # trialnum = trialnum[ybinned>3]
    # licks = licks[ybinned>3]
    # timedFF = timedFF[ybinned>3]
    # ybinned = ybinned[ybinned>3]
    # plot pre-first reward dop activity    
    firstrew = np.where(rewards==1)[0][0]
    rews_centered = np.zeros_like(ybinned[:firstrew])
    rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
    rews_iind = consecutive_stretch(np.where(rews_centered)[0])
    min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
    rews_centered = np.zeros_like(ybinned[:firstrew])
    rews_centered[min_iind]=1
    trialnumvr = VR[8][0]
    catchtrialsnum = trialnumvr[VR[16][0].astype(bool)]

    return dff, rewards, trialnum, ybinned, licks, timedFF, rews_centered, layer, firstrew, catchtrialsnum, gainf, rewsize

def get_rewzones(rewlocs, gainf):
    # note that gainf is multiplied here
    # gainf = 1/scalingf
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
