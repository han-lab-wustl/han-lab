
#%%
"""
zahra
feb 2025
activity of cells per trial
"""

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=8
mpl.rcParams["ytick.major.size"]=8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,per_trial_dff
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_cm_window=20 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#################### PRE-REWARD ####################
trial_dffs_pre = []
trialstates_all_pre = []
com_goal_all_pre = []
com_goal_subset_all_pre = []
goal_cells_all_pre = []
epoch_perm_all_pre = []
lasttr=8 #  last trials
bins=90
reward_cell_type='pre'
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        trial_dff,trialstates,com_goal,com_goal_subset,goal_cells,\
            epoch_perm=per_trial_dff(reward_cell_type,ii,params_pth,radian_alignment_saved,
                animal,day,bins,goal_cm_window=20)
        trial_dffs_pre.append(trial_dff)
        trialstates_all_pre.append(trialstates)
        com_goal_all_pre.append(com_goal)
        com_goal_subset_all_pre.append(com_goal_subset)
        goal_cells_all_pre.append(goal_cells)
        epoch_perm_all_pre.append(epoch_perm)
#%%
#################### POST-REWARD ####################
trial_dffs_post = []
trialstates_all_post = []
com_goal_all_post = []
com_goal_subset_all_post = []
goal_cells_all_post = []
epoch_perm_all_post = []
lasttr=8 #  last trials
bins=90
reward_cell_type='post'
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        trial_dff,trialstates,com_goal,com_goal_subset,goal_cells,\
            epoch_perm=per_trial_dff(reward_cell_type,ii,params_pth,radian_alignment_saved,
                animal,day,bins,goal_cm_window=20)
        trial_dffs_post.append(trial_dff)
        trialstates_all_post.append(trialstates)
        com_goal_all_post.append(com_goal)
        com_goal_subset_all_post.append(com_goal_subset)
        goal_cells_all_post.append(goal_cells)
        epoch_perm_all_post.append(epoch_perm)
#%%
# post-reward
dfc = conddf.copy()
dfc = dfc[((dfc.animals!='e217')) & (dfc.optoep<2)]
dfs = []
for ii in range(len(dfc)):    
    cells = [f'cell{gc:04d}' for gc in goal_cells_all_post[ii]]
    if len(cells)>0:
        df=pd.DataFrame()
        prerew_cells_dff_ep1=np.concatenate([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep1' in k] for cll in cells])
        trialsep1 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep1' in k] for cll in cells][0])
        numcllsep1 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep1' in k] for cll in cells])
        prerew_cells_dff_ep2=np.concatenate([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep2' in k] for cll in cells])
        trialsep2 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep2' in k] for cll in cells][0])
        numcllsep2 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep2' in k] for cll in cells])
        prerew_cells_dff_ep3=np.concatenate([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep3' in k] for cll in cells])
        trialsep3 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep3' in k] for cll in cells][0])
        numcllsep3 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep3' in k] for cll in cells])
        prerew_cells_dff_ep4=np.concatenate([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep4' in k] for cll in cells])
        trialsep4 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep4' in k] for cll in cells][0])
        numcllsep4 = len([[v for k,v in trial_dffs_post[ii].items() if cll in k and 'ep4' in k] for cll in cells])
        nanpad = np.ones(np.max([len(prerew_cells_dff_ep1),len(prerew_cells_dff_ep2),
                        len(prerew_cells_dff_ep3),len(prerew_cells_dff_ep4)]))*np.nan
        df['postrew_cells_dff_ep1']=nanpad;df['postrew_cells_trialnm_ep1']=nanpad
        df['trialstates_ep1']=nanpad;df['trialstates_ep2']=nanpad;df['trialstates_ep3']=nanpad;df['trialstates_ep4']=nanpad
        df['postrew_cells_dff_ep1'][:len(prerew_cells_dff_ep1)]=prerew_cells_dff_ep1
        df['postrew_cells_trialnm_ep1'][:len(prerew_cells_dff_ep1)]=np.concatenate([np.arange(trialsep1)]*numcllsep1)
        df['trialstates_ep1'][:len(prerew_cells_dff_ep1)]=np.concatenate([trialstates_all_post[ii]['ep1']]*numcllsep1)
        df['postrew_cells_dff_ep2']=nanpad;df['postrew_cells_trialnm_ep2']=nanpad
        df['postrew_cells_dff_ep2'][:len(prerew_cells_dff_ep2)]=prerew_cells_dff_ep2
        df['postrew_cells_trialnm_ep2'][:len(prerew_cells_dff_ep2)]=np.concatenate([np.arange(trialsep2)]*numcllsep2)
        df['trialstates_ep2'][:len(prerew_cells_dff_ep2)]=np.concatenate([trialstates_all_post[ii]['ep2']]*numcllsep2)
        df['postrew_cells_dff_ep3']=nanpad;df['postrew_cells_trialnm_ep3']=nanpad
        df['postrew_cells_dff_ep3'][:len(prerew_cells_dff_ep3)]=prerew_cells_dff_ep3
        df['postrew_cells_trialnm_ep3'][:len(prerew_cells_dff_ep3)]=np.concatenate([np.arange(trialsep3)]*numcllsep3)
        try:
            df['trialstates_ep3'][:len(prerew_cells_dff_ep3)]=np.concatenate([trialstates_all_post[ii]['ep3']]*numcllsep3)
        except Exception as e:
            print(e)
        df['postrew_cells_dff_ep4']=nanpad;df['postrew_cells_trialnm_ep4']=nanpad
        df['postrew_cells_dff_ep4'][:len(prerew_cells_dff_ep4)]=prerew_cells_dff_ep4
        df['postrew_cells_trialnm_ep4'][:len(prerew_cells_dff_ep4)]=np.concatenate([np.arange(trialsep4)]*numcllsep4)
        try:
            df['trialstates_ep4'][:len(prerew_cells_dff_ep4)]=np.concatenate([trialstates_all_post[ii]['ep4']]*numcllsep4)
        except Exception as e:
            print(e)

        df['animal'] = [dfc.iloc[ii]['animals']]*np.max([len(df['postrew_cells_dff_ep1']),
                len(df['postrew_cells_dff_ep2']),len(df['postrew_cells_dff_ep3']),
                len(df['postrew_cells_dff_ep4'])])
        df['day'] = [dfc.iloc[ii]['days']]*np.max([len(df['postrew_cells_dff_ep1']),
                len(df['postrew_cells_dff_ep2']),len(df['postrew_cells_dff_ep3']),
                len(df['postrew_cells_dff_ep4'])])
        dfs.append(df)

bigdf = pd.concat(dfs)
bigdf=bigdf.reset_index()
#%%
# plot
#
# bigdf['prerew_cells_trialnm_ep1']=bigdf['prerew_cells_trialnm_ep1']+1
sns.scatterplot(x='postrew_cells_trialnm_ep1',y='postrew_cells_dff_ep1',hue='trialstates_ep1',
                data=bigdf,alpha=0.4)
sns.lineplot(x='postrew_cells_trialnm_ep1',y='postrew_cells_dff_ep1',hue='trialstates_ep1',
                data=bigdf)

#%%
for an in bigdf.animal.unique():
    dfplt = bigdf[bigdf.animal==an]
    fig,ax=plt.subplots(figsize=(6,5))
    sns.scatterplot( x='postrew_cells_trialnm_ep1',y='postrew_cells_dff_ep1',
        hue='trialstates_ep1',
        data=dfplt,alpha=.4)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    fig,ax=plt.subplots(figsize=(6,5))
    sns.lineplot( x='postrew_cells_trialnm_ep1',y='postrew_cells_dff_ep1',
        hue='trialstates_ep1',
        data=dfplt,alpha=.4)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
#%%
# pre-reward
# concatenate across animals
dfc = conddf.copy()
dfc = dfc[((dfc.animals!='e217')) & (dfc.optoep<2)]
dfs = []
for ii in range(len(dfc)):    
    cells = [f'cell{gc:04d}' for gc in goal_cells_all_pre[ii]]
    if len(cells)>0:
        df=pd.DataFrame()
        prerew_cells_dff_ep1=np.concatenate([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep1' in k] for cll in cells])
        trialsep1 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep1' in k] for cll in cells][0])
        numcllsep1 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep1' in k] for cll in cells])
        prerew_cells_dff_ep2=np.concatenate([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep2' in k] for cll in cells])
        trialsep2 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep2' in k] for cll in cells][0])
        numcllsep2 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep2' in k] for cll in cells])
        prerew_cells_dff_ep3=np.concatenate([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep3' in k] for cll in cells])
        trialsep3 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep3' in k] for cll in cells][0])
        numcllsep3 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep3' in k] for cll in cells])
        prerew_cells_dff_ep4=np.concatenate([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep4' in k] for cll in cells])
        trialsep4 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep4' in k] for cll in cells][0])
        numcllsep4 = len([[v for k,v in trial_dffs_pre[ii].items() if cll in k and 'ep4' in k] for cll in cells])
        nanpad = np.ones(np.max([len(prerew_cells_dff_ep1),len(prerew_cells_dff_ep2),
                        len(prerew_cells_dff_ep3),len(prerew_cells_dff_ep4)]))*np.nan
        df['prerew_cells_dff_ep1']=nanpad;df['prerew_cells_trialnm_ep1']=nanpad
        df['trialstates_ep1']=nanpad;df['trialstates_ep2']=nanpad;df['trialstates_ep3']=nanpad;df['trialstates_ep4']=nanpad
        df['prerew_cells_dff_ep1'][:len(prerew_cells_dff_ep1)]=prerew_cells_dff_ep1
        df['prerew_cells_trialnm_ep1'][:len(prerew_cells_dff_ep1)]=np.concatenate([np.arange(trialsep1)]*numcllsep1)
        df['trialstates_ep1'][:len(prerew_cells_dff_ep1)]=np.concatenate([trialstates_all_pre[ii]['ep1']]*numcllsep1)
        df['prerew_cells_dff_ep2']=nanpad;df['prerew_cells_trialnm_ep2']=nanpad
        df['prerew_cells_dff_ep2'][:len(prerew_cells_dff_ep2)]=prerew_cells_dff_ep2
        df['prerew_cells_trialnm_ep2'][:len(prerew_cells_dff_ep2)]=np.concatenate([np.arange(trialsep2)]*numcllsep2)
        df['trialstates_ep2'][:len(prerew_cells_dff_ep2)]=np.concatenate([trialstates_all_pre[ii]['ep2']]*numcllsep2)
        df['prerew_cells_dff_ep3']=nanpad;df['prerew_cells_trialnm_ep3']=nanpad
        df['prerew_cells_dff_ep3'][:len(prerew_cells_dff_ep3)]=prerew_cells_dff_ep3
        df['prerew_cells_trialnm_ep3'][:len(prerew_cells_dff_ep3)]=np.concatenate([np.arange(trialsep3)]*numcllsep3)
        try:
            df['trialstates_ep3'][:len(prerew_cells_dff_ep3)]=np.concatenate([trialstates_all_pre[ii]['ep3']]*numcllsep3)
        except Exception as e:
            print(e)
        df['prerew_cells_dff_ep4']=nanpad;df['prerew_cells_trialnm_ep4']=nanpad
        df['prerew_cells_dff_ep4'][:len(prerew_cells_dff_ep4)]=prerew_cells_dff_ep4
        df['prerew_cells_trialnm_ep4'][:len(prerew_cells_dff_ep4)]=np.concatenate([np.arange(trialsep4)]*numcllsep4)
        try:
            df['trialstates_ep4'][:len(prerew_cells_dff_ep4)]=np.concatenate([trialstates_all_pre[ii]['ep4']]*numcllsep4)
        except Exception as e:
            print(e)

        df['animal'] = [dfc.iloc[ii]['animals']]*np.max([len(df['prerew_cells_dff_ep1']),
                len(df['prerew_cells_dff_ep2']),len(df['prerew_cells_dff_ep3']),
                len(df['prerew_cells_dff_ep4'])])
        df['day'] = [dfc.iloc[ii]['days']]*np.max([len(df['prerew_cells_dff_ep1']),
                len(df['prerew_cells_dff_ep2']),len(df['prerew_cells_dff_ep3']),
                len(df['prerew_cells_dff_ep4'])])
        dfs.append(df)

bigdf = pd.concat(dfs)
bigdf=bigdf.reset_index()
#%%
# plot
#
# bigdf['prerew_cells_trialnm_ep1']=bigdf['prerew_cells_trialnm_ep1']+1
sns.scatterplot(x='prerew_cells_trialnm_ep1',y='prerew_cells_dff_ep1',hue='trialstates_ep1',
                data=bigdf)
#%%
for an in bigdf.animal.unique():
    dfplt = bigdf[bigdf.animal==an]
    fig,ax=plt.subplots(figsize=(6,5))
    sns.scatterplot( x='prerew_cells_trialnm_ep1',y='prerew_cells_dff_ep1',
        hue='trialstates_ep1',
        data=dfplt,alpha=.4)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    fig,ax=plt.subplots(figsize=(6,5))
    sns.lineplot( x='prerew_cells_trialnm_ep1',y='prerew_cells_dff_ep1',
        hue='trialstates_ep1',
        data=dfplt,alpha=.4)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)