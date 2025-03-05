
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
    df = pd.DataFrame()
    df['goal_cell_num'] = []

    
#%%
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_rew_cells.p"
with open(rew_cells_tracked_dct, "rb") as fp: #unpickle
        rew_cells_tracked_dct = pickle.load(fp)
        
# get number of tracked rew cells across days (vs. shuf cells)
plt.rc('font', size=24)
animals = ['e218','e216','e201',
        'e186','e189','e145', 'z8', 'z9']
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
ax.legend(bbox_to_anchor=(1.001, 1.001))

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
sns.stripplot(data=df2, x='days_tracked', y='num_tracked_cells_per_mouse',s=8, color='k',ax=ax)
sns.barplot(data=df2, x='days_tracked', y='num_tracked_cells_per_mouse',fill=False, color='k',ax=ax, errorbar='se')
sns.lineplot(data=df2, # correct shift
        x=df2.days_tracked.values-1, y='shuf_num_tracked_cells_per_mouse',
        color='grey', label='shuffle',ax=ax)

ax.set_xlabel('# of days tracked')
ax.set_ylabel('# of reward-distance cells')
eps = [1,2,3]
y = 180
pshift = 30
fs=50
pfs = 12
for ii,ep in enumerate(eps):
        rewprop = df2.loc[(df2.days_tracked==ep), 'num_tracked_cells_per_mouse']
        shufprop = df2.loc[(df2.days_tracked==ep), 'shuf_num_tracked_cells_per_mouse']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=pfs)

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
