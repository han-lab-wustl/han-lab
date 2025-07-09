"""
zahra
nov 2024
quantify reward-relative cells post reward
# TODO: take all the cells and quantify their transient in rewarded vs. unrewarded stops
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position,extract_data_nearrew,perireward_binned_activity
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_window_cm=40 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#%%
# test
# from projects.pyr_reward.rewardcell import perireward_binned_activity
iind='e186_017_index015'
radian_alignment=radian_alignment_saved
tcs_correct, coms_correct, tcs_fail, coms_fail,\
        com_goal, goal_cell_shuf_ps_per_comp_av,\
                goal_cell_shuf_ps_av,pdist=radian_alignment[iind]
track_length=270
goal_window = goal_window_cm*(2*np.pi/track_length) 
# change to relative value 
coms_rewrel = np.array([com-np.pi for com in coms_correct])
# only get cells near reward        
perm = list(combinations(range(len(coms_correct)), 2))     
com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])                
# tuning curves that are close to each other across epochs
com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
# in addition, com near but after goal
upperbound=np.pi/4
com_goal = [com for com in com_goal if len(com)>0]
# get goal cells across all epochs        
goal_cells = intersect_arrays(*com_goal)
goal_cells=np.unique(np.concatenate(com_goal))
#%%
goal_cell_iind = goal_cells
tc = tcs_correct
for gc in goal_cell_iind:
        plt.figure()
        for ep in range(len(tc)):
                plt.plot(tcs_correct[ep,gc,:])
                plt.title(gc)
animal,day,pln = iind[:4], int(iind[5:8]),0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"        
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'licks',
'pyr_tc_s2p_cellind', 'timedFF','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
        'stat'])
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
# skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
# skew_mask = skew_filter>2
Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
dFF = dFF[:, skew>2]

scalingf=2/3
ybinned = fall['ybinned'][0]/scalingf
track_length=180/scalingf    
forwardvel = fall['forwardvel'][0]    
changeRewLoc = np.hstack(fall['changeRewLoc'])
trialnum=fall['trialnum'][0]
rewards = fall['rewards'][0]
if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]        # set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))

velocity = fall['forwardvel'][0]
veldf = pd.DataFrame({'velocity': velocity})

from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
velocity = np.hstack(veldf.rolling(5).mean().values)
# velocity - ndarray: velocity of the animal
# thres - float: Threshold speed in cm/s
# Fs - int: Number of frames minimum to be considered stopped
# ftol - int: Frame tolerance for merging stop periods
moving_middle,stop = get_moving_time_v3(velocity,2,40,20)
pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, rew_stop_with_lick,\
        mov_success_tmpts=get_stops_licks(moving_middle, stop, pre_win_framesALL, post_win_framesALL,\
        velocity, (fall['rewards'][0]==.5).astype(int), fall['licks'][0], 
        max_reward_stop=31.25*5)
# nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
#         post_win_framesALL,velocity, fall['rewards'][0])
nonrew_stop_without_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
nonrew_stop_without_lick_per_plane[nonrew_stop_without_lick.astype(int)] = 1
nonrew_stop_with_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
nonrew_stop_with_lick_per_plane[nonrew_stop_with_lick.astype(int)] = 1
rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
rew_per_plane[rew_stop_with_lick.astype(int)] = 1
#%%
plt.rc('font', size=16)
range_val,binsize=10, .1
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')
goal_cell_iind=np.array(goal_cell_iind)
colors=sns.color_palette('Dark2')
fig, axes = plt.subplots(nrows=3,sharex=True,figsize=(3,4),height_ratios=[2,1,1])
# pre and post eg cell
for iii,gc in enumerate(goal_cell_iind[[28,2]]):
    # TODO: make condensed
    _, meannstops, __, rewnstops = perireward_binned_activity(Fc3[:,gc], nonrew_stop_without_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanvelnonrew, __, velnonrew = perireward_binned_activity(velocity, nonrew_stop_without_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanlicknonrew, __, licknonrew = perireward_binned_activity(fall['licks'][0], nonrew_stop_without_lick_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanrewnonrew, __, rewnonrew = perireward_binned_activity(fall['rewards'][0], nonrew_stop_without_lick_per_plane, 
    fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)

    _, meannlstops, __, rewnlstops = perireward_binned_activity(Fc3[:,gc], nonrew_stop_with_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanvelnonrewl, __, velnonrewl = perireward_binned_activity(velocity, nonrew_stop_with_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanlicknonrewwl, __, licknonrewwl = perireward_binned_activity(fall['licks'][0], nonrew_stop_with_lick_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanrewnonrewl, __, rewnonrewl = perireward_binned_activity(fall['rewards'][0], nonrew_stop_with_lick_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)

    _, meanrstops, __, rewrstops = perireward_binned_activity(Fc3[:,gc], rew_per_plane, 
    fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanvelrew, __, velrew = perireward_binned_activity(velocity, rew_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanlickrew, __, lickrew = perireward_binned_activity(fall['licks'][0], rew_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    _, meanrewrewstops, __, rewrewstops = perireward_binned_activity(fall['rewards'][0], rew_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
    if iii == 0:
        ax = axes[0]  # base axis for first cell
        ax1 = ax
    else:
        ax1 = ax.twinx()  # second y-axis for second cell
    ax1.plot(meanrstops, color=colors[iii], label=f'Cell {gc}')
    ax1.fill_between(
        range(0, int(range_val / binsize) * 2),
        meanrstops - scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
        meanrstops + scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
        alpha=0.5, color=colors[iii]
    )
    ax1.spines[['top']].set_visible(False)
    if iii == 0:
        ax1.set_ylabel('$\Delta F/F$ (Pre)', color=colors[iii])
    else:
        ax1.set_ylabel('$\Delta F/F$ (Post)', color=colors[iii])
    ax1.axvline(int(range_val / binsize), color='k', linestyle='--')

excolor='dimgrey'
ax=axes[2]
ax.plot(meanvelrew,color=excolor, label='rewarded stops')
ax.fill_between(
range(0, int(range_val / binsize) * 2),
meanvelrew - scipy.stats.sem(velrew, axis=1, nan_policy='omit'),
meanvelrew + scipy.stats.sem(velrew, axis=1, nan_policy='omit'),
alpha=0.5, color=excolor
)               
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Velocity (cm/s)')
ax.set_xlabel('Time from stop (s)')

ax=axes[1]
# meanlickrew, __, lickrew
lickrew = np.array([moving_average(xx,3) for xx in lickrew.T]).T
lickrew=lickrew/np.nanmax(lickrew,axis=0)
meanlickrew=np.nanmean(lickrew,axis=1)
ax.plot(moving_average(meanlickrew,3),alpha=0.5,color=excolor, label='rewarded stops')
ax.fill_between(
range(0, int(range_val / binsize) * 2),
meanlickrew - scipy.stats.sem(lickrew, axis=1, nan_policy='omit'),
meanlickrew + scipy.stats.sem(lickrew, axis=1, nan_policy='omit'),
alpha=0.5, color=excolor
)                 
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)    
ax.set_ylabel('Norm. Licks')
ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1,20))
ax.set_xticklabels(np.arange(-range_val, range_val + 1, 2))
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Norm. licks')
ax.legend().set_visible(False)

plt.savefig(os.path.join(savedst, f'postrew_traces_cell{gc}.svg'))
