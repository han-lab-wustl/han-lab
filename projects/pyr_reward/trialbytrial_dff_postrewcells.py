

"""
zahra
april 2025
trial by trial dff
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
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays, consecutive_stretch, \
    make_velocity_tuning_curves
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position, extract_data_nearrew, perireward_binned_activity
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
from projects.memory.behavior import get_behavior_tuning_curve
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'

goal_window_cm=20 # to search for rew cells
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
                    goal_cell_shuf_ps_av=radian_alignment[k]
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
    com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=upperbound) & (np.nanmedian(coms_rewrel[:,
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
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10

    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
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
    dfs_ep = []
    if len(goal_cells)>0:
        for ep in range(len(eps)-1):
            try:
                eprng = np.arange(eps[ep],eps[ep+1])
                # eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                breaks = np.where(np.diff(str_trials) != 1)[0] + 1
                runs = np.split(str_trials, breaks)
                # 2) keep only runs of length >= 3
                valid_runs = [run for run in runs if len(run) >= 3]
                # 3) flatten back into one sorted array of “good” indices
                good_idxs = np.sort(np.concatenate(valid_runs))
                str_trials = good_idxs
                trials = trialnum[eprng]
                f = Fc3[eprng,:]
                f = f[:,goal_cells]
                # make trial structure
                f__ = [f[trials==tr] for tr in str_trials]
                maxtime = np.nanmax(np.array([len(xx) for xx in f__]))
                dff = np.ones((len(str_trials), f.shape[1], maxtime))*np.nan
                unqtrials = str_trials
                for tr_idx, tr in enumerate(unqtrials):
                    f_ = f[trials == tr].T  # shape: (n_cells, trial_length)
                    trial_length = f_.shape[1]
                    dff[tr_idx, :, :trial_length] = f_

                dff_by_trial_cell = np.nanmean(dff, axis=2)/np.nanmax(dff, axis=2)  

                df=pd.DataFrame()
                df['dff']=np.concatenate(dff_by_trial_cell)    
                df['animal']=[animal]*len(df)
                df['day']=[day]*len(df)
                df['cellid']=np.concatenate([goal_cells]*dff_by_trial_cell.shape[0])
                df['trial']=np.repeat(np.arange(dff_by_trial_cell.shape[0]),len(goal_cells))
                df['epoch'] = ep        
                dfs_ep.append(df)
            except Exception as e:
                print(e)
        df = pd.concat(dfs_ep)
        dfs.append(df)

#%%
#plot all cells
df=pd.concat(dfs)
df = df.reset_index()
df=df[(df.animal!='e139') & (df.animal!='e145')]
sns.lineplot(y='dff',x='trial',data=df[(df.epoch==0)],hue='animal')