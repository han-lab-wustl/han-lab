
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

def phase_shifted_correlation(acceleration, neural_activity, max_shift):
    """
    Calculate phase-shifted correlation between acceleration and neural activity.
    
    Parameters:
        acceleration (np.array): The acceleration data.
        neural_activity (np.array): The neural activity data.
        max_shift (int): The maximum shift (in samples) to apply for phase-shifting.
        
    Returns:
        shifts (np.array): Array of shift values.
        correlations (np.array): Correlation values for each shift.
    """
    
    # Ensure the signals have the same length
    assert len(acceleration) == len(neural_activity), "Signals must have the same length"
    
    shifts = np.arange(-max_shift, max_shift + 1, 10)
    correlations = np.zeros(len(shifts))
    
    for i, shift in enumerate(shifts):
        if shift < 0:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[shift:] = 0
        else:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[:shift] = 0
        
        # Calculate the correlation for this shift
        correlation, _ = scipy.stats.pearsonr(acceleration, shifted_neural_activity)
        correlations[i] = correlation
    
    return correlations
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
an = 'e216'
pln=0
day=32
trackedcellarr = np.array(trackeddct[an][0]).astype(int)
trackedcellind = np.where(trackedcellarr>1)[0]
tracked_lut, days= get_tracked_lut(celltrackpth,an,0)
tracked_lut_multiday_rew_cells = tracked_lut.iloc[trackedcellind]
days_rec = conddf.loc[(conddf.animals==an) & (conddf.optoep.values==-1), 'days'].values
tracked_lut_multiday_rew_cells = tracked_lut_multiday_rew_cells[days_rec]
num_days_tracked = trackedcellarr[np.where(trackedcellarr>1)[0]]
# sanity check plot cell per day

from projects.DLC_behavior_classification.eye import perireward_binned_activity
range_val=20; binsize=0.1
params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel','rewards', 'timedFF'])
dayidx = np.where(tracked_lut.columns==day)[0][0]
dFF = fall['dFF']
rewards = fall['rewards'][0]==.5
timedFF = fall['timedFF'][0]

acc = np.diff(fall['forwardvel'][0])/np.diff(timedFF)
accdf = pd.DataFrame({'acc': acc})
acc = np.hstack(accdf.rolling(100).mean().fillna(0).values)
# cells correlated with acc
# Calculate phase-shifted correlation
max_shift = 100  # You can adjust the max shift based on your data
# ~ 3 s phase shifts
rshift = [phase_shifted_correlation(acc, dFF[1:,i], max_shift) 
        for i in range(dFF.shape[1])]
rshiftmax = [np.max(xx) for xx in rshift]
plt.hist(rshiftmax)

acccells = np.where(np.array(rshiftmax)>0.15)[0]
pos1,pos2 = 15000,23000
plt.plot(dFF[pos1:pos2,acccells])
plt.plot(acc[pos1:pos2]/50,'k')
meandf = []
for acccell in acccells:
    normmeanrewdFF, meanrewdFF, normrewdFF, \
    rewdFF = perireward_binned_activity(dFF[:, acccell], rewards, timedFF, \
        range_val, binsize)    
    meandf.append(meanrewdFF)
__, meanrewacc, _, ___ = perireward_binned_activity(acc, rewards[1:], timedFF[1:], \
        range_val, binsize)

plt.plot(np.array(meandf).T)
plt.plot(meanrewacc/20,'k')
#%%

fig,ax=plt.subplots()
for day in tracked_lut.columns.values[:10]:  
    print(day)  
    params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel','rewards', 'timedFF'])
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