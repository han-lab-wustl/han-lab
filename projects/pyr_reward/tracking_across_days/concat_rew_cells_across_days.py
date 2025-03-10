
"""
zahra
march 2025
find proportion of cells that are considered reward cells for 
multiple epochs and days
1) get day 1 reward cells
2) get the next 2 days of reward cells
3) get proportion of cells that are reward cells across all the epochs
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
            assert suite2pind_remain.shape[0]==tcs_correct.shape[1]
            # indices per epo
            iind_goal_cells_all=[suite2pind_remain[xx] for xx in com_goal]
            iind_goal_cells_all_per_day.append(iind_goal_cells_all)
    # per day cells
    per_day_goal_cells = [intersect_arrays(*xx) for xx in iind_goal_cells_all_per_day]
    # test per day + epoch 1 from next day, n so on...
    per_day_nextday_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        per_day_nextday_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0]])]))))
    per_day_nextday_ep2=[]; days_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1]])]))))
            days_ep2.append(ii)
        except Exception as e:
            print(e)
    per_day_nextday_ep3=[]; days_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])]))))
            print(len(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])]))))
            days_ep3.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep1=[]; days_2day_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0]])]))))            
            else:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0]])]))))            

            days_2day_ep1.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep2=[]; days_2day_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])]))))            
            else:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])]))))            

            days_2day_ep2.append(ii)
        except Exception as e:
            print(e)
            
    per_day_next2day_ep3=[]; days_2day_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1],
                    iind_goal_cells_all_per_day[ii+2][2]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])]))))            
            else:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])]))))            

            days_2day_ep3.append(ii)
        except Exception as e:
            print(e)


    # get number of cells in each comparison
    per_day_goal_cells_num = [len(xx) for xx in per_day_goal_cells]
    per_day_nextday_ep1_num = [len(xx) for xx in per_day_nextday_ep1]
    per_day_nextday_ep2_num = [len(xx) for xx in per_day_nextday_ep2]
    per_day_nextday_ep3_num = [len(xx) for xx in per_day_nextday_ep3]
    per_day_next2day_ep1_num = [len(xx) for xx in per_day_next2day_ep1]
    per_day_next2day_ep2_num = [len(xx) for xx in per_day_next2day_ep2]
    per_day_next2day_ep3_num = [len(xx) for xx in per_day_next2day_ep3]
    
    per_day_goal_cells_all.append([per_day_goal_cells_num,per_day_nextday_ep1_num,
                per_day_nextday_ep2_num,per_day_nextday_ep3_num,
                per_day_next2day_ep1_num,per_day_next2day_ep2_num,per_day_next2day_ep3_num])
    
#%%

df=pd.DataFrame()
lut = ['1_day', '1_day_1_epoch', '1_day_2_epochs',
    '1_day_3_epochs', '2_days_1_epoch', '2_days_2_epochs',
    '2_days_3_epochs']
biglst = []; bigann =[]; biganimal=[]
for ii in range(len(per_day_goal_cells_all[0])):
    biglst.append(np.concatenate([xx[ii] for xx in per_day_goal_cells_all]))
    bigann.append(np.concatenate([[lut[ii]]*len(xx[ii]) for xx in per_day_goal_cells_all]))
    biganimal.append(np.concatenate([[animals[jj]]*len(xx[ii]) for jj,xx in enumerate(per_day_goal_cells_all)]))

df['reward_cell_count']=np.concatenate(biglst)
df['epoch_type']=np.concatenate(bigann)
df['animal']=np.concatenate(biganimal)

#%%
plt.rc('font', size=16) 
fig, ax = plt.subplots(figsize=(12,9))
sns.stripplot(x='epoch_type',y='reward_cell_count',hue='animal',data=df, dodge=True)
sns.barplot(x='epoch_type',y='reward_cell_count',hue='animal',data=df)
ax.tick_params(axis='x', rotation=45)
#%%
s=12
sumdf = df.groupby(['animal','epoch_type']).mean(numeric_only=True)
sumdf = sumdf.sort_index(axis=1)
sumdf=sumdf.reset_index()
fig, ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='epoch_type',y='reward_cell_count',data=sumdf, dodge=True,color='k',
        s=s)
sns.barplot(x='epoch_type',y='reward_cell_count',data=sumdf, fill=False,color='k')
ax.tick_params(axis='x', rotation=45)
# make lines
ans = sumdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='epoch_type', y='reward_cell_count', 
    data=sumdf[sumdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)

# only show ectra epochs
#%%
s=12
sumdf=sumdf[sumdf.epoch_type!='1_day']
fig, ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='epoch_type',y='reward_cell_count',data=sumdf, dodge=True,color='k',
        s=s)
sns.barplot(x='epoch_type',y='reward_cell_count',data=sumdf, fill=False,color='k')
ax.tick_params(axis='x', rotation=45)
# make lines
ans = sumdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='epoch_type', y='reward_cell_count', 
    data=sumdf[sumdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)
