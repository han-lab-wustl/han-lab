# Zahra
# extract dff and cluster
import numpy as np, os
from scipy.io import loadmat
import h5py
import seaborn as sns

src = r'Y:\sstcre_analysis\celltrack\e201_week4789\Results'
mat = os.path.join(src,'dff_per_day.mat')
f = h5py.File(mat)
# extract from h5py
dff = []
for i in range(len(f['dff'][:])):
    dff.append(f[f['dff'][i][0]][:])
commoncells = os.path.join(src,'commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat')
cc = loadmat(commoncells)['cellmap2dayacrossweeks'].astype(int)
cc=cc-1 # subtract from matlab ind
# load fall example
# day 41
daypth = r'Z:\sstcre_imaging\e201\41\230413_ZD_000_000'
fallpth = os.path.join(daypth,'suite2p', 'plane0', 'Fall.mat')
fall = loadmat(fallpth)
epoch = np.where(fall['changeRewLoc']>1)
epoch_ind = epoch[1][1]
trials = max(max(fall['trialnum'][:, :epoch_ind])) # total number of trials
dff_av = []
mask=cc[:,17]
dff_day = dff[17].T[mask][mask>-1]
for trial in range(trials-10,trials): #only first epoch
    print(trial)
    if trial > 0:
        trialind = np.where(fall['trialnum']==trial)[1]
        trialind = trialind[trialind < epoch_ind] # first trial of first epoch?
        # dff structure = each item of list is a day
        # 18 days 
        # days=[14,15,16,17,18,27, 28, 29, 30, 31, 32, 33,36,38,39,40,41]
        # cluster dff 1 day            
        dff_av.append(dff_day.T[trialind]) #mean across all frames

dff_av = np.mean(np.array(dff_av),axis=0)        

sns.clustermap(dff_av,row_cluster=False,cmap='viridis',
                        col_cluster=True)