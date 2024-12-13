
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

animals = ['e218','e216','e217','e201','e186','e189',
        'e190', 'e145', 'z8', 'z9']

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
goal_window_cm=20 # to search for rew cells
radian_tuning_dct = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_20cm_window.p'
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)

tracked_rew_cell_inds_all = {}
trackeddct = {}
#%%
coms = {}
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
        tracked_rew_cell_inds,tracked_rew_cell_ind_shufs = [],[]
        for ii, day in enumerate(dys): # iterate per day
                if animal!='e217' and conddf.optoep.values[ii]<2:
                        if animal=='e145' or animal=='e139': pln=2
                        else: pln=0
                        # get lut
                        tracked_lut, days= get_tracked_lut(celltrackpth,animal,pln)
                        if day in days:
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
                                k = [xx for xx in radian_alignment_saved.keys() if f'{animal}_{day:03d}' in xx]
                                if len(k)>0:
                                        k=k[0]
                                        tcs_correct, coms_correct, tcs_fail, coms_fail,\
                                                com_goal, goal_cell_shuf_ps_per_comp_av,\
                                                        goal_cell_shuf_ps_av,pdist = radian_alignment_saved[k]            
                                        assert suite2pind_remain.shape[0]==tcs_correct.shape[1]
                                        # get goal cells across all epochs        
                                        goal_cells = intersect_arrays(*com_goal)            
                                        if len(goal_cells)>0:
                                                # suite2p indices of rew cells
                                                goal_cells_s2p_ind = suite2pind_remain[goal_cells]                
                                                # if a tracked cell, add 1 to tracked cell ind
                                                goal_tracked_idx = []
                                                for c in goal_cells_s2p_ind:
                                                        tridx = np.where(tracked_lut[day]==c)[0]
                                                        if len(tridx)>0:
                                                                goal_tracked_idx.append(tridx[0])
                                                        tracked[goal_tracked_idx] += 1
                                                        # populate shuffles
                                                        goal_cells_shuf_s2pind, coms_rewrels=get_shuffled_goal_cell_indices(rewlocs, coms_correct,
                                                                goal_window,suite2pind_remain)
                                                for sh in range(shuffles):
                                                        goal_tracked_idx = []
                                                        for c in goal_cells_shuf_s2pind[sh]:
                                                                tridx = np.where(tracked_lut[day]==c)[0]
                                                                if len(tridx)>0:
                                                                        goal_tracked_idx.append(tridx[0])                
                                                        tracked_shuf[sh, goal_tracked_idx] += 1
                                
                trackeddct[animal] = [tracked, tracked_shuf]

dct = {}; dct['rew_cells_coms_tracked'] = [trackeddct]
# save pickle of dcts
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_postrew_cells.p"
with open(rew_cells_tracked_dct, "wb") as fp:   #Pickling
    pickle.dump(dct, fp) 
#
#%%
# get number of tracked rew cells across days (vs. shuf cells)
plt.rc('font', size=24)
animals = ['e218','e216','e201',
        'e186','e145', 'z8', 'z9']
df = pd.DataFrame()
df['tracked_cells_num'] = np.concatenate([trackeddct[an][0][trackeddct[an][0]>0] for an in animals]).astype(int)
df['tracked_cells_shuf_1'] = np.concatenate([trackeddct[an][1][random.randint(0,shuffles),trackeddct[an][0]>0] for an in animals]).astype(int)
df['p_values_per_cell'] = np.concatenate([sum(trackeddct[an][1][:,(trackeddct[an][0]>0)]>trackeddct[an][0][trackeddct[an][0]>0])/shuffles for an in animals])
df['animals'] = np.concatenate([[an]*len(trackeddct[an][0][trackeddct[an][0]>0]) for an in animals])
df['animals_shuf'] = np.concatenate([[an+'_shuf']*len(trackeddct[an][0][trackeddct[an][0]>0]) for an in animals])
#%%
# average counts per animal
fig,ax=plt.subplots(figsize=(3,6))

df_plt = df[df.p_values_per_cell<0.05]

sns.histplot(data=df[df.p_values_per_cell<0.05], x='tracked_cells_num', color='darkcyan',
            bins=3, label = 'Reward-distance cells')
sns.histplot(data=df[(df.p_values_per_cell<0.05) & 
        (df['tracked_cells_shuf_1']>=1)], x='tracked_cells_shuf_1',  color='dimgray',
        bins=3,alpha=0.5, label='shuffle')
ax.legend(bbox_to_anchor=(1.1, 1.1))

dfs_av = df
# reorganize
df2 = pd.DataFrame()
days = [1,2,3]
tracked_cells_per_day_per_mouse = [[sum(df.loc[df.animals==an, 'tracked_cells_num']==day) for an in animals] for day in range(1,4)]
tracked_cells_per_day_per_mouse_shuf = [[sum(df.loc[df.animals_shuf==an, 'tracked_cells_shuf_1']==day) for an in df.animals_shuf.unique()] for day in range(1,4)]
df2['num_tracked_cells_per_mouse'] = np.concatenate(tracked_cells_per_day_per_mouse)
df2['shuf_num_tracked_cells_per_mouse'] = np.concatenate(tracked_cells_per_day_per_mouse_shuf)
df2['animal'] = np.concatenate([animals]*len(days))
df2['days_tracked'] = np.concatenate(np.concatenate([[[day]*len(animals)] for day in days]))
fig,ax=plt.subplots(figsize=(3,6))
sns.stripplot(data=df2, x='days_tracked', y='num_tracked_cells_per_mouse',s=12, color='k',ax=ax)
sns.barplot(data=df2, x='days_tracked', y='num_tracked_cells_per_mouse',fill=False, color='k',ax=ax, errorbar='se')
sns.lineplot(data=df2, # correct shift
        x=df2.days_tracked.values-1, y='shuf_num_tracked_cells_per_mouse',
        color='grey', label='shuffle',ax=ax)
ax.set_xlabel('# of days tracked')
ax.set_ylabel('# of post-rew cells')
eps = [1,2,3]
y = 10
pshift = 1
fs=50
pfs = 12
for ii,ep in enumerate(eps):
        rewprop = df2.loc[(df2.days_tracked==ep), 'num_tracked_cells_per_mouse']
        shufprop = df2.loc[(df2.days_tracked==ep), 'shuf_num_tracked_cells_per_mouse']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=pfs,rotation=45)
ax.legend(bbox_to_anchor=(1.3, .95))
ax.spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(savedst, 'across_days_rew_cells.svg'), bbox_inches='tight', dpi=500)
#%%
fig,ax=plt.subplots(figsize=(7,8))
sns.barplot(data=df[df.p_values_per_cell<0.05], y='tracked_cells_num', x='animals',
            hue='animals',palette='hls', errorbar='se')
sns.barplot(data=df[df.p_values_per_cell<0.05], x='animals_shuf', y='tracked_cells_shuf_1', alpha=0.5, 
            hue='animals_shuf', palette='Greys', errorbar='se')

ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.spines[['top','right']].set_visible(False)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

#%%
# average by animal
dfs_av = df.groupby(['animals']).mean(numeric_only=True)
# reorganize
df2 = pd.DataFrame()
df2['tracked_cells_num'] = np.concatenate([dfs_av.tracked_cells_num.values,dfs_av.tracked_cells_shuf_1.values])
df2['condition'] = np.concatenate([['real']*len(dfs_av.tracked_cells_num.values),['shuffle']*len(dfs_av.tracked_cells_shuf_1.values)])
fig,ax=plt.subplots(figsize=(2,5))
sns.stripplot(data=df2, y='tracked_cells_num', x='condition',s=8, color='k')
ax.spines[['top','right']].set_visible(False)
sns.barplot(data=df2, y='tracked_cells_num', x='condition',fill=False, color='k',
            errorbar='se')

num_cells = dfs_av.tracked_cells_num.values
num_cells_ctrl = dfs_av.tracked_cells_shuf_1.values
t,pval = scipy.stats.ttest_rel(num_cells,num_cells_ctrl)
# test if cells are tracked > 1 day
t2, pval2 = scipy.stats.ttest_1samp(num_cells,1)
y = 1.7
pshift = 0.2
fs=36
ii = 0.5
# statistical annotation        
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.3g}',fontsize=10)
y = 1.5
pshift = 0.2
fs=36
ii = 0
# statistical annotation        
if pval2 < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval2 < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval2 < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval2:.3g}',fontsize=10)
