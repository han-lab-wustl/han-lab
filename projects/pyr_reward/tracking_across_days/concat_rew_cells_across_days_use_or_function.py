
"""
zahra
march 2025
find proportion of cells that are considered reward cells for 
multiple epochs and days
1) get day 1 reward cells
2) get the next 2 days of reward cells
3) get proportion of cells that are reward cells across all the epochs
3/10/25
split up day 1 into epochs
use or function (if a cell is a reward cell for 3/5 epochs for eg., keep it)
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
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_days_from_cellreg_log_file, find_log_file, get_radian_position, \
    get_tracked_lut, get_tracking_vars, get_shuffled_goal_cell_indices, get_reward_cells_that_are_tracked
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
animals = ['e218','e216','e217','e201','e186',
        'e190', 'e145', 'z8', 'z9']
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r'Z:\\saved_datasets\\radian_tuning_curves_rewardcentric_all.p'
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)

tracked_rew_cell_inds_all = {}
trackeddct = {}
#%%
per_day_goal_cells_all = []
# defined vars
maxep = 5
shuffles = 1000
# redo across days analysis but init array per animal
for animal in animals:
    # all rec days
    dys = conddf.loc[conddf.animals==animal, 'days'].values
    # index compared to org df
    dds = list(conddf[conddf.animals==animal].index)
    # init 
    iind_goal_cells_all_per_day=[]
    perm_per_day = []
    for ii, day in enumerate(dys[:4]): # iterate per day
        if animal!='e217' and conddf.optoep.values[dds[ii]]==-1:
            if animal=='e145': pln=2
            else: pln=0
            # get lut
            tracked_lut, days= get_tracked_lut(celltrackpth,animal,pln)
            if ii==0:
                # init with min 4 epochs
                # ep x cells x days
                # instead of filling w/ coms, fill w/ binary
                tracked = np.zeros((tracked_lut.shape[0]))
                tracked_shuf =np.zeros((shuffles, tracked_lut.shape[0]))
            # get vars
            params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
            dFF, suite2pind_remain, VR, scalingf, rewsize, ybinned, forwardvel, changeRewLoc,\
                rewards, eps, rewlocs, track_length = get_tracking_vars(params_pth)
            goal_window = 20*(2*np.pi/track_length) # cm converted to rad, consistent with quantified window sweep
            # find key
            k = [k for k,v in radian_alignment_saved.items() if f'{animal}_{day:03d}' in k][0]
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
                com_goal, goal_cell_shuf_ps_per_comp_av,\
                goal_cell_shuf_ps_av = radian_alignment_saved[k]     
            perm = list(combinations(range(len(coms_correct)), 2))       
            assert suite2pind_remain.shape[0]==tcs_correct.shape[1]
            # indices per epo
            iind_goal_cells_all=[suite2pind_remain[xx] for xx in com_goal]
            iind_goal_cells_all_per_day.append(iind_goal_cells_all)
            perm_per_day.append(perm)
    # per day cells
    per_day_goal_cells = [intersect_arrays(*xx) for xx in iind_goal_cells_all_per_day]
    # split per ep
    # test per day + epoch 1 from next day, n so on...
    per_day_nextday_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        per_day_nextday_ep1.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0]])])))
    per_day_nextday_ep2=[]; days_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep2.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1]])])))
            days_ep2.append(ii)
        except Exception as e:
            print(e)
    per_day_nextday_ep3=[]; days_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep3.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])])))
            print(len(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])]))))
            days_ep3.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep1=[]; days_2day_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep1.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0]])])))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep1.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0]])])))            
            else:
                per_day_next2day_ep1.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0]])])))          

            days_2day_ep1.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep2=[]; days_2day_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep2.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1]])])))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep2.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])])))            
            else:
                per_day_next2day_ep2.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])])))          

            days_2day_ep2.append(ii)
        except Exception as e:
            print(e)
            
    per_day_next2day_ep3=[]; days_2day_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep3.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1],
                    iind_goal_cells_all_per_day[ii+2][2]])])))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep3.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])])))            
            else:
                per_day_next2day_ep3.append(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])])))            

            days_2day_ep3.append(ii)
        except Exception as e:
            print(e)
    # get number of cells in each comparison
    per_day_goal_cells_num = [[len(xx) for xx in yy] for yy in iind_goal_cells_all_per_day] # incl all epochs
    per_day_nextday_ep1_num = [len(xx) for xx in per_day_nextday_ep1]
    per_day_nextday_ep2_num = [len(xx) for xx in per_day_nextday_ep2]
    per_day_nextday_ep3_num = [len(xx) for xx in per_day_nextday_ep3]
    per_day_next2day_ep1_num = [len(xx) for xx in per_day_next2day_ep1]
    per_day_next2day_ep2_num = [len(xx) for xx in per_day_next2day_ep2]
    per_day_next2day_ep3_num = [len(xx) for xx in per_day_next2day_ep3]
    # save
    per_day_goal_cells_all.append([perm_per_day,iind_goal_cells_all_per_day,per_day_nextday_ep1,
                per_day_nextday_ep2,per_day_nextday_ep3,
                per_day_next2day_ep1,per_day_next2day_ep2,per_day_next2day_ep3])
    
#%%

# get cell # per epoch
# iind_goal_cells_all_per_day - epoch combinations per 4 days

# between ep 1 and 2 or ep 2 and 3
num_clls_per_2ep_per_an = []
for xx in per_day_goal_cells_all: # per animal
    num_clls_per_2ep=[]
    for jj,yy in enumerate(xx[1]): # per day tracked; also keep track of perm
        for ii,ep in enumerate(yy): # per epoch combiantion
            if len(ep)>0:
                if ((xx[0][jj][ii]==(0,1))|(xx[0][jj][ii]==(1,2))):
                    ep2comp = len(ep) 
                    num_clls_per_2ep.append(ep2comp)
    num_clls_per_2ep_per_an.append(num_clls_per_2ep)
num_clls_per_2ep_per_an=[np.nanmean(np.array(xx)) for xx in num_clls_per_2ep_per_an]
# between ep 1,2,3
num_clls_per_3ep_per_an = []
for xx in per_day_goal_cells_all: # per animal
    num_clls_per_3ep=[]
    for jj,yy in enumerate(xx[1]): # per day tracked; also keep track of perm
        if len(yy)==3:        # if 3 combinations
            ep3comp = len(intersect_arrays(*yy)) 
            num_clls_per_3ep.append(ep3comp)
        if len(yy)==6:
            ep3collect=[] # collect 3 epoch combinations from 4 ep days
            for ii,ep in enumerate(yy): # per epoch combiantion
                if len(ep)>0:
                    if ((xx[0][jj][ii]==(0,1))|(xx[0][jj][ii]==(1,2))|(xx[0][jj][ii]==(0,2))):
                        ep3collect.append(ep)
            num_clls_per_3ep.append(len(intersect_arrays(*ep3collect)))
    num_clls_per_3ep_per_an.append(num_clls_per_3ep)            
num_clls_per_3ep_per_an=[np.nanmean(np.array(xx)) for xx in num_clls_per_3ep_per_an]

# between ep 1,2,3,4
num_clls_per_4ep_per_an = []
for xx in per_day_goal_cells_all: # per animal
    num_clls_per_4ep=[]
    for jj,yy in enumerate(xx[1]): # per day tracked; also keep track of perm
        if len(yy)==6:        # if 3 combinations
            ep4comp = len(intersect_arrays(*yy)) 
            num_clls_per_4ep.append(ep4comp)
    num_clls_per_4ep_per_an.append(num_clls_per_4ep)            
num_clls_per_4ep_per_an=[np.nanmean(np.array(xx)) for xx in num_clls_per_4ep_per_an]

#%%
# get 2,3,4,5 epoch combinations from per epoch days
# 0-perm_per_day
# 1-iind_goal_cells_all_per_day
# 2-per_day_nextday_ep1,
# 3-per_day_nextday_ep2,
# 4-per_day_nextday_ep3,
# 5-per_day_next2day_ep1,
# 6-per_day_next2day_ep2,
# 7-per_day_next2day_ep3
ind_to_test = [2,3,4,5,6,7] # check all epoch concatenated combinations
# test epochs 3,4,5,6,7,8, etc...
epochs_to_test = np.arange(3,13)

across_days_num_clls_per_ep_per_an = []
for ep in epochs_to_test:
    across_days_num_clls_per_ep=[]
    for kk,xx in enumerate(per_day_goal_cells_all): # per animal
        num_clls_per_ep=[]
        for ind in ind_to_test: # get all concatenated days
            for jj,yy in enumerate(xx[ind]): # per day tracked; also keep track of perm                
                comb=list(combinations(yy,ep))
                if len(comb)>0:
                    epcomp=np.nanmax(np.array([len(intersect_arrays(*zz)) for zz in comb]))
                    num_clls_per_ep.append(epcomp)
        across_days_num_clls_per_ep.append(num_clls_per_ep)
    across_days_num_clls_per_ep_per_an.append(across_days_num_clls_per_ep)

# av per animal
across_days_num_clls_per_ep_per_an=[[np.nanmean(np.array(yy)) for yy in xx] for xx in across_days_num_clls_per_ep_per_an]
#%%
df=pd.DataFrame()
df['reward_cell_count']=np.concatenate(across_days_num_clls_per_ep_per_an)
df['animal']=np.concatenate([animals]*len(across_days_num_clls_per_ep_per_an))
df['epoch_number']=np.repeat([list(np.arange(3,13))],len(animals))
# for per day additions
df2=pd.DataFrame()
df2['reward_cell_count']=num_clls_per_2ep_per_an
df2['animal']=animals
df2['epoch_number']=[2]*len(df2)
df3=pd.DataFrame()
df3['reward_cell_count']=num_clls_per_3ep_per_an
df3['animal']=animals
df3['epoch_number']=[3]*len(df3)
df4=pd.DataFrame()
df4['reward_cell_count']=num_clls_per_4ep_per_an
df4['animal']=animals
df4['epoch_number']=[4]*len(df4)

df=pd.concat([df,df2,df3,df4])
df=df.reset_index()
plt.rc('font', size=16) 
s=12
df=df.groupby(['animal','epoch_number']).mean(numeric_only=True)
df=df.reset_index()
fig, ax = plt.subplots(figsize=(12,9))
sns.stripplot(x='epoch_number',y='reward_cell_count',data=df, dodge=True, color='k',
        s=s)
sns.barplot(x='epoch_number',y='reward_cell_count',data=df,fill=False,color='k')
# make lines
ans = df.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df.epoch_number-2, y='reward_cell_count', 
    data=df[df.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)

#%%
# normalize to max (2 epochs) for each animal?
for an in animals:
    df.loc[df.animal==an, 'reward_cell_count_norm']=df.loc[df.animal==an,'reward_cell_count']/np.nanmax(df.loc[df.animal==an,'reward_cell_count'])

dfplt = df[df.epoch_number>2]
plt.rc('font', size=16) 
fig, ax = plt.subplots(figsize=(8,9))
sns.stripplot(x='epoch_number',y='reward_cell_count_norm',
        data=dfplt, dodge=True, color='k',s=s)
sns.barplot(x='epoch_number',y='reward_cell_count_norm',
        data=dfplt,fill=False,color='k')
# make lines
ans = df.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=dfplt.epoch_number-3, y='reward_cell_count_norm', 
    data=dfplt[dfplt.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)

#%%
# zoom

dfplt = df[df.epoch_number>3]
plt.rc('font', size=16) 
fig, ax = plt.subplots(figsize=(8,9))
sns.stripplot(x='epoch_number',y='reward_cell_count_norm',
        data=dfplt, dodge=True, color='k',s=s)
sns.barplot(x='epoch_number',y='reward_cell_count_norm',
        data=dfplt,fill=False,color='k')
# make lines
ans = df.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=dfplt.epoch_number-4, y='reward_cell_count_norm', 
    data=dfplt[dfplt.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)
