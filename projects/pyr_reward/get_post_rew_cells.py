
"""
zahra
aug. 2024
get post rew cells based on acc
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
from projects.DLC_behavior_classification.eye import perireward_binned_activity

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
    
    shifts = np.arange(-max_shift, max_shift + 1, 5)
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
an = 'e186'
pln=0
day=2
trackedcellarr = np.array(trackeddct[an][0]).astype(int)
trackedcellind = np.where(trackedcellarr>1)[0] # more that 1 day tracjed
tracked_lut, days= get_tracked_lut(celltrackpth,an,pln)
tracked_lut_multiday_rew_cells = tracked_lut.iloc[trackedcellind]
days_rec = conddf.loc[(conddf.animals==an) & (conddf.optoep.values==-1), 'days'].values
tracked_lut_multiday_rew_cells = tracked_lut_multiday_rew_cells[days_rec] # only for days recorded
num_days_tracked = trackedcellarr[np.where(trackedcellarr>1)[0]]
#%%
# sanity check plot cell per day
range_val=15; binsize=0.1
params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel','rewards', 'timedFF'])
dayidx = np.where(tracked_lut.columns==day)[0][0]
dFF = fall['dFF']
if an=='e145':
    rewards = (fall['rewards'][0]==1).astype(int)
else:
    rewards = (fall['rewards'][0]==.5).astype(int)
timedFF = fall['timedFF'][0]

acc = np.diff(fall['forwardvel'][0])/np.diff(timedFF)
accdf = pd.DataFrame({'acc': acc})
acc = np.hstack(accdf.rolling(100).mean().fillna(0).values)
# cells correlated with acc
# Calculate phase-shifted correlation
max_shift = int(np.ceil(31.25/(pln+1)))  # You can adjust the max shift based on your data
# ~ 1 s phase shifts
rshiftmax = []
for i in range(dFF.shape[1]):
    dff = dFF[1:,i]
    dff[np.isnan(dff)]=0 # nan to 0
    r=phase_shifted_correlation(acc, dff, max_shift)
    rshiftmax.append(np.max(r))

plt.hist(rshiftmax)
# only plot top 5% for now
acccells = np.where(np.array(rshiftmax)>np.quantile(rshiftmax,.95))[0]
# remove INs
skew = scipy.stats.skew(dFF[:,acccells], nan_policy='omit', axis=0)
acccells = acccells[skew>2]

meandf = []
for acccell in acccells:
    normmeanrewdFF, meanrewdFF, normrewdFF, \
    rewdFF = perireward_binned_activity(dFF[:, acccell], rewards, timedFF, \
        range_val, binsize)    
    meandf.append(meanrewdFF)
    
__, meanrewacc, _, ___ = perireward_binned_activity(acc, rewards[1:], timedFF[1:], \
        range_val, binsize)
velocity = fall['forwardvel'][0]
velocitydf = pd.DataFrame({'velocity': velocity})
__, meanrewvel, _, ___ = perireward_binned_activity(velocity, rewards, timedFF, \
        range_val, binsize)
#%%
fig,ax=plt.subplots()
ax.plot(np.array(meandf).T)
ax.plot(meanrewacc/20,'k')
ax.plot(meanrewvel/100,'k--')
# ax.set_ylim([-.5,2])

#%%
# see if a couple of these cells are consistent over days
# get last day
dy=conddf.loc[(conddf.animals==an) & (conddf.optoep<0), 'days'].values[-1]
dys = conddf.loc[(conddf.animals==an) & (conddf.optoep<0), 'days'].values
trackedidxacc = [np.where(tracked_lut[dy].values==acc)[0] for acc in acccells]
# test with 1 i know is stop and lick cell
# but does not pass acc criteria?
acctrs = [xx[0] for xx in trackedidxacc if len(xx)>0]  
acctr=acctrs[1]
fig,ax=plt.subplots()
for day in dys:  
    print(day)  
    params_pth = rf"Y:\analysis\fmats\{an}\days\{an}_day{day:03d}_plane{pln}_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 
        'forwardvel','rewards', 'timedFF'])
    dFF = fall['dFF']
    dayidx = np.where(tracked_lut.columns==day)[0][0]
    if an=='e145':
        rewards = (fall['rewards'][0]==1).astype(int)
    else:
        rewards = (fall['rewards'][0]==.5).astype(int)
    timedFF = fall['timedFF'][0]        
    if tracked_lut.iloc[acctr][day]>-1:
        normmeanrewdFF, meanrewdFF, normrewdFF, \
        rewdFF = perireward_binned_activity(dFF[:, tracked_lut.iloc[acctr][day]], \
            rewards, timedFF, range_val, binsize)     
        ax.plot(meanrewdFF,label=f'Day {day}')
        ax.set_title(f'Acceleration-correlated tracked cell # {acctr}')
ax.legend()