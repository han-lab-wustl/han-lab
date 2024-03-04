
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import get_success_failure_trials, get_performance, get_rewzones
#%%

days_cnt_an1 = 10; days_cnt_an2=9; days_cnt_an3=24; days_cnt_an4=11; days_cnt_an5=8; days_cnt_an6=9
animals = np.hstack([['e218']*(days_cnt_an1), ['e216']*(days_cnt_an2), \
                    ['e201']*(days_cnt_an3), ['e186']*(days_cnt_an4), ['e190']*(days_cnt_an5), ['e189']*(days_cnt_an6)])
in_type = np.hstack([['vip']*(days_cnt_an1), ['vip']*(days_cnt_an2), \
                    ['sst']*(days_cnt_an3), ['pv']*(days_cnt_an4), ['ctrl']*(days_cnt_an5), ['ctrl']*(days_cnt_an6)])
days = np.array([20,21,22,23, 35, 38, 41, 44, 47,50,7,8,9,37, 41, 48, \
                50, 54,57,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,\
                2,3,4,5,31,32,33,34,36,37,40,33,34,35,40,41,42,43,45,35,36,37,38,39,40,41,42,44])#[20,21,22,23]#
optoep = np.array([-1,-1,-1,-1, 3, 2, 3, 2,3, 2,-1,-1,-1,2, 3, 3, 2, 3,2, \
                -1,-1,-1,2,3,0,2,3,0,2,3,0,2,3,0,2,3,0,2,3,0,2,3,3,-1,-1,-1,-1,2,3,2,3,2,3,2,-1, \
        -1,-1,3,0,1,2,3,-1,-1,-1,-1,2,3, 2, 0, 2])#[2,3,2,3]
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing
#%%
bin_size = 3
# com shift analysis
dcts = []
for dd,day in enumerate(days):
    dct = {}
    animal = animals[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['VR'])
    VR = fall['VR'][0][0][()]
    eps = np.where(np.hstack(VR['changeRewLoc']>0))[0]
    eps = np.hstack(np.ravel([eps, len(np.hstack(VR['changeRewLoc']))]))
    scalingf = VR['scalingFACTOR'][0][0]
    track_length = 180/scalingf
    nbins = track_length/bin_size
    ybinned = np.hstack(VR['ypos']/scalingf)
    rewlocs = np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
    trialnum = np.hstack(VR['trialNum'])
    rewards = np.hstack(VR['reward'])
    licks = np.hstack(VR['lick']).astype(bool)
    eptest = optoep[dd]
    if optoep[dd]<2: eptest = random.randint(2,3)   
    if len(eps)<4: eptest = 2 # if no 3 epochs    
    rates_opto, rates_prev = get_performance(eptest, dd, eps, trialnum, rewards, licks, ybinned, rewlocs)
    rewzones = get_rewzones(rewlocs, 1.5)

    dct['rates'] = [rates_prev, rates_opto]    
    dct['rewlocs'] = [rewlocs[eptest-2], rewlocs[eptest-1]]
    dct['rewzones'] = [rewzones[eptest-2], rewzones[eptest-1]]
    dcts.append(dct)