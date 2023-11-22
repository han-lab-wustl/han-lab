"""
@author: zahra
"""

import os, sys, pickle, pandas as pd, numpy as np, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from clustering.kmeans import collect_clustering_vars, run_pca, run_kmeans
from preprocessing import fixcsvcols
#analyze videos and copy vr files before this step
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import ceil 
import datetime
# test
################################
vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
dlcfls = r'Y:\DLC\dlc_mixedmodel2'#\for_analysis'
with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp) 

row = mouse_df.iloc[196]
################################
def get_long_grooms_per_ep(dlcfls,row,hrz_summary = False,
savedst = r'Y:\DLC\dlc_mixedmodel2\figures',gainf=3/2):
    """collect variables for locations of grooming

    Args:
        dlcfls (_type_): _description_
        row (_type_): _description_
        hrz_summary (bool, optional): _description_. Defaults to False.
        gainf = length of track relative to 180cm (default in virmen)
    Returns:
        groom: whether the animal grooms or not
        counts: grooms in diff defined categories
        cat: categories 
        yposgrs: ypos of grooms        
    """
    matfl = os.path.join(dlcfls,row["VR"][:16]+"_vr_dlc_align.p")       
    with open(matfl,'rb') as fp: #unpickle
        mat = pickle.load(fp)
    eps = np.where(mat['changeRewLoc']>0)[0]    
    eps = np.hstack([list(eps), len(mat['changeRewLoc'])])    
    # default return
    groom, starts, stops, tr_s, tr_f, \
          yposgrs_s, yposgrs_f, yposgrs_p, len_grooms = [np.nan]*9
    # at least 2 epochs, rewarded hrz
    if 'HRZ' in mat['experiment'] and sum(mat['rewards']>0) and len(eps)>2: # only for hrz
        dfpth = os.path.join(dlcfls, row['DLC']) #.values[0]
        df = pd.read_csv(dfpth)
        if 'bodyparts' not in df.columns:  # format csv
            df = fixcsvcols(dfpth)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns = ["Unnamed: 0"])
        # downsample to neural data
        idx = len(df) - 1 if len(df) % 2 else len(df)
        df = df[:idx].groupby(df.index[:idx] // 2).mean()
        #paw
        paw_x, paw_y = filter_paw(df)
        licks = mat['lickVoltage']<-0.07 # HARD CODED, MIGHT BE DIFF
        ybin_paw = mat['ybinned'][:-1]*gainf# remove 1 for diff
        rewz = mat['changeRewLoc'][mat['changeRewLoc']>0]*gainf
        # if there is any grooming        
        if sum(paw_y.values)>0:
            starts, stops_ = get_starts_stops_grooming(paw_x,paw_y,
                            frame_thres=75)
            len_grooms = stops_-starts # in frames
            if len(starts)>0:                
                # successful trials, gets ypos and total number of trials
                yposgrs_s, yposgrs_p, tr_s = categorize_grooming(eps, mat,starts,rewz, success=True)
                # failed trials
                yposgrs_f, yposgrs_p, tr_f = categorize_grooming(eps, mat,starts,rewz, success=False)

                if hrz_summary:    
                    make_hrz_summary_fig(mat, licks, matfl, ybin_paw, starts,
                    save=savedst)
                groom = 1           

        else:
            if hrz_summary:    
                    # plots hrz behavior
                    plt.figure()
                    ypos = mat['ybinned']*gainf
                    plt.plot(ypos, color='slategray',
                            linewidth=0.5)
                    plt.scatter(np.argwhere(mat['rewards']==0.5).T[0], ypos[mat['rewards']==0.5], color='b', marker='o')
                    plt.scatter(np.argwhere(licks).T[0], ypos[licks], color='r', marker='o',
                                s = 2**2)                    
                    plt.title(os.path.basename(matfl))
                    plt.savefig(os.path.join(savedst, f'{os.path.basename(matfl)}_beh.pdf'))
            groom = 0
    
    return groom, starts, stops, \
        yposgrs_s, yposgrs_f, yposgrs_p, tr_s, tr_f, len_grooms
                            

def categorize_grooming(eps, mat, starts, rewz, success=True, gainf=3/2):
    yposgrs = []; yposgrs_p = []
    for ep in range(len(eps)-1):
        rng = np.arange(eps[ep],eps[ep+1])
        trialnumep = np.hstack(mat['trialnum'][rng])
        # rng = rng[trialnumep] # > 3 no probes
        # trialnumep = trialnumep[trialnumep]
        rewep = (mat['rewards']==1)[rng]
        # types of trials, success vs. fail
        s_tr = []; f_tr = []
        for tr in np.unique(trialnumep[trialnumep>=3]):
            if sum(rewep[trialnumep==tr])>0:
                s_tr.append(tr)
            else:
                f_tr.append(tr)
        trm_f = np.isin(trialnumep,f_tr)    
        trm = np.isin(trialnumep,s_tr)   
        rng_s = rng[trm] 
        rng_f = rng[trm_f] 
        if not ep==0:
            rng_p = rng[trialnumep<3]
        else:
            rng_p = [] # dont count start probes
        # categorize by ypos
        if success:
            gr_ = [xx in rng_s for xx in starts]
            tr_ = len(s_tr)
        else:
            gr_ = [xx in rng_f for xx in starts]
            tr_ = len(f_tr)
        # grooms during probes
        gr_p = [xx in rng_p for xx in starts]; gr_p = starts[gr_p]
        gr_ = starts[gr_]        
        yend = 180*gainf
        # ypos rel to reward
        # yposgr_rel_rew = (yposgr-rewz[ep])/rewz[ep]
        # gm addition
        yrew = rewz[ep]-5 # approximate start of rew zone
        yrew_p = rewz[ep]-5
        # relative to no. of trials/tr_ 
        yposgr_p = [((ceil(xx)*gainf)-yrew_p) for xx in mat['ybinned'][gr_p]]
        yposgr = [((ceil(xx)*gainf)-yrew) for xx in mat['ybinned'][gr_]]
        # condition = yrew-yposgr>=0        
        # yposgr_rel_rew = (((yend*condition)-yposgr-((condition-0.5*2)*yrew))/((yend*condition)-(condition-0.5*2)*yrew))-(condition-1)
        if len(yposgr)>0:
            yposgrs.append(yposgr)
        if len(yposgr_p)>0:
            yposgrs_p.append(yposgr_p)

    yposgrs = convert_to_hstack(yposgrs)
    yposgrs_p = convert_to_hstack(yposgrs_p)

    return yposgrs, yposgrs_p, tr_

def filter_paw(df, threshold=0.99):
    """
    filters dlc poses with low likelihood
    """
    df['PawTop_x'][df['PawTop_likelihood'].astype('float32') < threshold] = 0
    df['PawTop_y'][df['PawTop_likelihood'].astype('float32') < threshold] = 0
    df['PawMiddle_x'][df['PawMiddle_likelihood'].astype('float32') < threshold] = 0
    df['PawMiddle_y'][df['PawMiddle_likelihood'].astype('float32') < threshold] = 0
    df['PawBottom_x'][df['PawBottom_likelihood'].astype('float32') < threshold] = 0
    df['PawBottom_y'][df['PawBottom_likelihood'].astype('float32') < threshold] = 0
    paw_y = df[['PawTop_y','PawBottom_y','PawMiddle_y']].astype('float32').mean(axis=1)
    paw_x = df[['PawTop_x','PawBottom_x','PawMiddle_x']].astype('float32').mean(axis=1)

    return paw_x, paw_y

def get_starts_stops_grooming(paw_x,paw_y,frame_thres=75):
    """get the start and stop of a grooming bout

    Args:
        paw_x (_type_): _description_
        paw_y (_type_): _description_
        frame_thres (int, optional): _description_. Defaults to 75.

    Returns:
        _type_: _description_
    """
    paw_gf = scipy.ndimage.gaussian_filter(paw_y.values,3)
    paw_gf_x = scipy.ndimage.gaussian_filter(paw_x.values,3)
    diffs = np.diff((paw_gf>0).astype(int),axis=0)
    starts = np.argwhere(diffs == 1).T[0]    
    stops = np.argwhere(diffs == -1).T[0]                   
    
    if len(stops)<len(starts):
        stops_ = np.zeros(len(stops)+1)
        stops_[:-1] = stops
        stops_[-1] = len(paw_gf)
    else:
        stops_ = stops
    start_stop = stops_-starts
    # filter by long grooms
    starts = starts[start_stop>frame_thres]
    stops_ = stops_[start_stop>frame_thres]

    return starts, stops_

def convert_to_hstack(arr):
    if len(arr)>0: 
        arr = np.hstack(arr)
    return arr
    
def make_hrz_summary_fig(mat, licks, matfl, ybin_paw, starts, 
    save=False,gainf=3/2):
    # plots hrz behavior with grooms
    plt.figure()
    ypos = mat['ybinned']*gainf
    plt.plot(ypos, color='slategray',
            linewidth=0.5)
    plt.scatter(np.argwhere(mat['rewards']==0.5).T[0], ypos[mat['rewards']==0.5], color='b', marker='o')
    plt.scatter(np.argwhere(licks).T[0], ypos[licks], color='r', marker='o',
                s = 2**2)
    plt.scatter(starts, ybin_paw[starts], color='y', marker='*',
                s = 20**2)
    plt.title(os.path.basename(matfl))
    if save:
        plt.savefig(os.path.join(save, f'{os.path.basename(matfl)}_grooming_beh.pdf'))
    
    plt.close('all')

    return