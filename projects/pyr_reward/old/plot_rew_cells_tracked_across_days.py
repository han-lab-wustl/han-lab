
"""
zahra
june 2024
visualize reward-relative cells across days
idea 1: find all the reward relative cells per day and see if they map onto the same 
subset of cells
idea 2: find reward relative cells on the last day (or per week, or per 5 days)
and see what their activity was like on previous days

"""
#%%

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, matplotlib.backends.backend_pdf
import pickle, seaborn as sns, random, math, os, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_days_from_cellreg_log_file, find_log_file, get_radian_position, \
    get_tracked_lut, get_tracking_vars, get_shuffled_goal_cell_indices, get_reward_cells_that_are_tracked

from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
animals = ['e218','e216','e201',
        'e186','e189','e145', 'z8', 'z9']

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r'Z:\\saved_datasets\\radian_tuning_curves_reward_cell_bytrialtype_nopto_window030.p'
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_rew_cells.p"
with open(rew_cells_tracked_dct, "rb") as fp:   #Pickling
    trackeddct = pickle.load(fp)['rew_cells_coms_tracked'][0]
#
#%%
# plot dff / tuning curves of cells tracked > 1
an = 'e218'
pln=0
trackedcellarr = np.array(trackeddct[an][0]).astype(int)
trackedcellind = np.where(trackedcellarr>1)[0]
tracked_lut, days= get_tracked_lut(celltrackpth,an,0)
tracked_lut_multiday_rew_cells = tracked_lut.iloc[trackedcellind]
days_rec = conddf.loc[(conddf.animals==an) & (conddf.optoep.values==-1), 'days'].values
tracked_lut_multiday_rew_cells = tracked_lut_multiday_rew_cells[days_rec]
num_days_tracked = trackedcellarr[np.where(trackedcellarr>1)[0]]

for day in days_rec:    
    params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['dFF'])
    # to remove skew cells
    dFF = fall['dFF']
    cells2plt = tracked_lut_multiday_rew_cells[day].values
    fig,ax=plt.subplots()
    ax.plot(dFF[10000:15000, cells2plt])
    
#%% 
# sanity check plot cell per day

from projects.DLC_behavior_classification.eye import perireward_binned_activity
range_val=10; binsize=0.1
ind2plt = tracked_lut.iloc[2].values
fig,ax=plt.subplots()
an='e218'
for day in tracked_lut.columns.values[:10]:  
    print(day)  
    params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'rewards', 'timedFF'])
    dayidx = np.where(tracked_lut.columns==day)[0][0]
    dFF = fall['dFF']
    rewards = fall['rewards'][0]==.5
    timedFF = fall['timedFF'][0]
    if ind2plt[dayidx]>-1:
        normmeanrewdFF, meanrewdFF, normrewdFF, \
        rewdFF = perireward_binned_activity(dFF[:, ind2plt[dayidx]], rewards, timedFF, \
            range_val, binsize)    
        ax.plot(meanrewdFF,label=day)
ax.legend()