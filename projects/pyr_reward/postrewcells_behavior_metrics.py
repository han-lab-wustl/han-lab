

"""
zahra
nov 2024
quantify reward-relative cells post reward
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
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,extract_data_nearrew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'near_rew.pdf')
#%%
goal_window_cm=40 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
# with open(saveddataset, "rb") as fp: #unpickle
#     radian_alignment_saved = pickle.load(fp)
radian_alignment_saved = {} # overwrite
goal_cell_iinds = []
goal_cell_props = []
goal_cell_nulls = []
num_epochs = []
pvals = []
rates_all = []
total_cells_all = []
epoch_perm = []
radian_alignment = {}
lasttr=8 # last trials
bins=90
saveto = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
# iterate through all animals
ii=1
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if animal!='e217' and conddf.optoep.values[ii]<1:
    pln=0
    if animal=='e145': pln=2
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
    radian_alignment,rate,p_value,total_cells,\
    goal_cell_iind,perm,goal_cell_prop,num_epoch,goal_cell_null=extract_data_nearrew(ii,
            params_pth,animal, day,
            bins,radian_alignment,radian_alignment_saved,goal_window_cm,pdf)
    # save
    rates_all.append(rate); pvals.append(p_value); total_cells_all.append(total_cells)
    epoch_perm.append(perm); goal_cell_iinds.append(goal_cell_iind); 
    goal_cell_props.append(goal_cell_prop); num_epochs.append(num_epoch)
    goal_cell_nulls.append(goal_cell_null)
pdf.close()

# save pickle of dcts
# with open(saveto, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp)
#%%
# test
# from projects.pyr_reward.rewardcell import perireward_binned_activity

tc=radian_alignment['e218_021_index007'][0]
for gc in goal_cell_iind:
    plt.figure()
    for ep in range(len(tc)):
        plt.plot(tc[ep,gc,:])
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
#%%
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
velocity = np.hstack(veldf.rolling(5).mean().values)
# velocity - ndarray: velocity of the animal
# thres - float: Threshold speed in cm/s
# Fs - int: Number of frames minimum to be considered stopped
# ftol - int: Frame tolerance for merging stop periods
moving_middle,stop = get_moving_time_v3(velocity,2,40,10)
pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, rew_stop_with_lick,=get_stops_licks(moving_middle, stop, pre_win_framesALL, post_win_framesALL,\
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
range_val,binsize=10, .2
#%%
for gc in goal_cell_iind:
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


    fig, axes = plt.subplots(nrows=4,sharex=True,figsize=(3,6))
    ax=axes[0]
    ax.plot(meanrstops,color='darkgoldenrod', label='rewarded stops')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanrstops - scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
    meanrstops + scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
    alpha=0.5, color='darkgoldenrod'
    )             
    ax.plot(meannstops,color='slategray', label='unrewarded stops w/o licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meannstops - scipy.stats.sem(rewnstops, axis=1, nan_policy='omit'),
    meannstops + scipy.stats.sem(rewnstops, axis=1, nan_policy='omit'),
    alpha=0.5, color='slategray'
    )         
    ax.plot(meannlstops,color='yellowgreen', label='unrewarded stops w/ licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meannlstops - scipy.stats.sem(rewnlstops, axis=1, nan_policy='omit'),
    meannlstops + scipy.stats.sem(rewnlstops, axis=1, nan_policy='omit'),
    alpha=0.5, color='yellowgreen'
    )                 
    ax.legend()
    ax.set_ylabel('$\Delta$ F/F')
    ax=axes[1]
    ax.plot(meanvelrew,color='navy', label='rewarded stops')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanvelrew - scipy.stats.sem(velrew, axis=1, nan_policy='omit'),
    meanvelrew + scipy.stats.sem(velrew, axis=1, nan_policy='omit'),
    alpha=0.5, color='navy'
    )             
    ax.plot(meanvelnonrewl,color='teal', label='unrewarded stops w licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanvelnonrewl - scipy.stats.sem(velnonrewl, axis=1, nan_policy='omit'),
    meanvelnonrewl + scipy.stats.sem(velnonrewl, axis=1, nan_policy='omit'),
    alpha=0.5, color='teal'
    )                    
    ax.plot(meanvelnonrew,color='k', label='unrewarded stops w/o licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanvelnonrew - scipy.stats.sem(velnonrew, axis=1, nan_policy='omit'),
    meanvelnonrew + scipy.stats.sem(velnonrew, axis=1, nan_policy='omit'),
    alpha=0.5, color='k'
    )                    
    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Velocity (cm/s)')
    ax=axes[2]
    # meanlickrew, __, lickrew
    ax.plot(meanlickrew,color='navy', label='rewarded stops')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanlickrew - scipy.stats.sem(lickrew, axis=1, nan_policy='omit'),
    meanlickrew + scipy.stats.sem(lickrew, axis=1, nan_policy='omit'),
    alpha=0.5, color='navy'
    )             
    # meanlicknonrewwl, __, licknonrewwl
    ax.plot(meanlicknonrewwl,color='teal', label='unrewarded stops w licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanlicknonrewwl - scipy.stats.sem(licknonrewwl, axis=1, nan_policy='omit'),
    meanlicknonrewwl + scipy.stats.sem(licknonrewwl, axis=1, nan_policy='omit'),
    alpha=0.5, color='teal'
    )                 
    # meanlicknonrew, __, licknonrew   
    ax.plot(meanlicknonrew,color='k', label='unrewarded stops w/o licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanlicknonrew - scipy.stats.sem(licknonrew, axis=1, nan_policy='omit'),
    meanlicknonrew + scipy.stats.sem(licknonrew, axis=1, nan_policy='omit'),
    alpha=0.5, color='k'
    )                    
    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    ax.set_ylabel('Licks')
    
    ax=axes[3]
    # meanlickrew, __, lickrew
    ax.plot(meanrewrewstops,color='navy', label='rewarded stops')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanrewrewstops - scipy.stats.sem(rewrewstops, axis=1, nan_policy='omit'),
    meanrewrewstops + scipy.stats.sem(rewrewstops, axis=1, nan_policy='omit'),
    alpha=0.5, color='navy'
    )             
    # meanlicknonrewwl, __, licknonrewwl
    ax.plot(meanrewnonrewl,color='teal', label='unrewarded stops w licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanrewnonrewl - scipy.stats.sem(rewnonrewl, axis=1, nan_policy='omit'),
    meanrewnonrewl + scipy.stats.sem(rewnonrewl, axis=1, nan_policy='omit'),
    alpha=0.5, color='teal'
    )                 
    # meanlicknonrew, __, licknonrew   
    ax.plot(meanrewnonrew,color='k', label='unrewarded stops w/o licks')
    ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    meanrewnonrew - scipy.stats.sem(rewnonrew, axis=1, nan_policy='omit'),
    meanrewnonrew + scipy.stats.sem(rewnonrew, axis=1, nan_policy='omit'),
    alpha=0.5, color='k'
    )                    
    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
    ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 10))
    ax.set_xticklabels(np.arange(-range_val, range_val + 1, 2))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Time from stop (s)')
    ax.set_ylabel('Rewards binned')
    
    fig.suptitle(f'cell # {gc}')
