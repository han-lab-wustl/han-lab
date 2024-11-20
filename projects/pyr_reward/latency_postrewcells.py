

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
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays, consecutive_stretch
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,extract_data_nearrew,perireward_binned_activity,\
    calculate_pre_latencies,calculate_post_latencies,compare_latencies
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'

goal_window_cm=40 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#%%
plt.close('all')
dfs=[]
for k,v in radian_alignment_saved.items():
    radian_alignment=radian_alignment_saved
    tcs_correct, coms_correct, tcs_fail, coms_fail,\
            com_goal, goal_cell_shuf_ps_per_comp_av,\
                    goal_cell_shuf_ps_av,pdist=radian_alignment[k]
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
    com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0))] for com in com_goal if len(com)>0]
    # get goal cells across all epochs        
    goal_cells = intersect_arrays(*com_goal)

    goal_cell_iind = goal_cells
    tc = tcs_correct
    animal,day,trash = k.split('_'); day=int(day)
    if animal=='e145': pln=2
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"        
    print(params_pth)
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
    timedFF=fall['timedFF'][0]
    if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]        # set vars
            timedFF=timedFF[:-1]
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    velocity = fall['forwardvel'][0]
    veldf = pd.DataFrame({'velocity': velocity})
    velocity = np.hstack(veldf.rolling(5).mean().values)
    # velocity - ndarray: velocity of the animal
    # thres - float: Threshold speed in cm/s
    # Fs - int: Number of frames minimum to be considered stopped
    # ftol - int: Frame tolerance for merging stop periods
    moving_middle,stop = get_moving_time_v3(velocity,2,40,20)
    pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
    nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, \
    rew_stop_with_lick,mov_success_tmpts=get_stops_licks(moving_middle, stop, 
                pre_win_framesALL, post_win_framesALL,\
            velocity, (fall['rewards'][0]==.5).astype(int), fall['licks'][0], 
            max_reward_stop=31.25*5)    
    # nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
    #         post_win_framesALL,velocity, fall['rewards'][0])
    nonrew_stop_without_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
    nonrew_stop_without_lick_per_plane[nonrew_stop_without_lick.astype(int)] = 1
    nonrew_stop_with_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
    nonrew_stop_with_lick_per_plane[nonrew_stop_with_lick.astype(int)] = 1
    movement_starts=mov_success_tmpts.astype(int)
    rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
    rew_per_plane[rew_stop_with_lick.astype(int)] = 1
    
    gc_latencies_rew=[]; gc_latencies_mov=[]
    for gc in goal_cell_iind:
        fd = pd.DataFrame({'f': Fc3[:,gc]})
        # smooth and set threshold so not detecting random spikes
        f = np.hstack(fd.rolling(70).mean().values)
        transients = consecutive_stretch(np.where(f>0.05)[0])
        transients = np.array([min(xx) for xx in transients])
        # plt.figure() test
        # for t in transients:
        #     plt.figure()
        #     plt.plot(Fc3[t-100:t+100,gc])
        # latency from start of transient to movement
        latencies_to_movement, latencies_to_rewards=compare_latencies(transients,
                movement_starts,(rewards==0.5).astype(int),timedFF)
        gc_latencies_rew.append(latencies_to_rewards)
        gc_latencies_mov.append(latencies_to_movement)
        # fig,ax=plt.subplots()
        # ax.scatter(range(len(latencies_to_movement)),latencies_to_movement,color='k')
        # ax2 = ax.twinx()
        # ax2.scatter(range(len(latencies_to_rewards)),latencies_to_rewards,color='orchid')
    # concat by cell
    df=pd.DataFrame()
    df['latency (s)']=np.concatenate([np.concatenate(gc_latencies_rew),np.concatenate(gc_latencies_mov)])/31.25
    df['behavior']=np.concatenate([['CS']*len(np.concatenate(gc_latencies_rew)),['Movement Start']*len(np.concatenate(gc_latencies_mov))])
    df['animal']=[animal]*len(df)
    df['day']=[day]*len(df)
    dfs.append(df)
#%%
#plot all cells
df=pd.concat(dfs)
df = df.reset_index()
# df=df[df.animal=='e201']
# df=dfs[5]
fig,ax=plt.subplots(figsize=(2.5,5))
# sns.stripplot(x='behavior',y='latency (s)',data=df,color='k',s=8,alpha=0.3)
sns.boxplot(x='behavior',y='latency (s)',data=df,fill=False,showfliers=False,whis=0)
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)

