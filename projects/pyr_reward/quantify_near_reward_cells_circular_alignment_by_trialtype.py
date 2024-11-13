

"""
zahra
july 2024
quantify reward-relative cells near reward
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
for ii in range(len(conddf)):
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
with open(saveto, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp)
#%%
# test
# from projects.pyr_reward.rewardcell import perireward_binned_activity

# tc=radian_alignment['e218_020_index003'][0]
# for gc in goal_cell_iind:
#     plt.figure()
#     for ep in range(len(tc)):
#         plt.plot(tc[ep,gc,:])
# fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
# 'pyr_tc_s2p_cellind', 'timedFF','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
#         'stat'])
# fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
# Fc3 = fall_fc3['Fc3']
# dFF = fall_fc3['dFF']
# Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
# dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
# skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
# # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
# # skew_mask = skew_filter>2
# Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
# scalingf=2/3
# ybinned = fall['ybinned'][0]/scalingf
# track_length=180/scalingf    
# forwardvel = fall['forwardvel'][0]    
# changeRewLoc = np.hstack(fall['changeRewLoc'])
# trialnum=fall['trialnum'][0]
# rewards = fall['rewards'][0]
# if animal=='e145':
#         ybinned=ybinned[:-1]
#         forwardvel=forwardvel[:-1]
#         changeRewLoc=changeRewLoc[:-1]
#         trialnum=trialnum[:-1]
#         rewards=rewards[:-1]        # set vars
# eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))

# velocity = fall['forwardvel'][0]
# veldf = pd.DataFrame({'velocity': velocity})

# velocity = np.hstack(veldf.rolling(5).mean().values)
# # velocity - ndarray: velocity of the animal
# # thres - float: Threshold speed in cm/s
# # Fs - int: Number of frames minimum to be considered stopped
# # ftol - int: Frame tolerance for merging stop periods
# moving_middle,stop = get_moving_time_v3(velocity,2,30,10)
# pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
# nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
#         post_win_framesALL,velocity, fall['rewards'][0])
# nonrew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
# nonrew_per_plane[nonrew.astype(int)] = 1
# rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
# rew_per_plane[rew.astype(int)] = 1

# _, meannstops, __, rewnstops = perireward_binned_activity(Fc3[:,goal_cell_iind[3]], nonrew_per_plane, 
#         fall['timedFF'][0], fall['trialnum'][0], 10, .4)
# _, meanvelnonrew, __, velnonrew = perireward_binned_activity(velocity, nonrew_per_plane, 
#         fall['timedFF'][0], fall['trialnum'][0], 10, .4)

# _, meanrstops, __, rewrstops = perireward_binned_activity(Fc3[:,goal_cell_iind[3]], rew_per_plane, 
# fall['timedFF'][0], fall['trialnum'][0], 10, .4)
# _, meanvelrew, __, velrew = perireward_binned_activity(velocity, rew_per_plane, 
#         fall['timedFF'][0], fall['trialnum'][0], 10, .4)
#%%
range_val, binsize = 10, .4
fig, axes = plt.subplots(nrows = 2)
ax=axes[0]
ax.plot(meanrstops,color='darkgoldenrod', label='rewarded stops')
ax.fill_between(
range(0, int(range_val / binsize) * 2),
meanrstops - scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
meanrstops + scipy.stats.sem(rewrstops, axis=1, nan_policy='omit'),
alpha=0.5, color='darkgoldenrod'
)             
ax.plot(meannstops,color='slategray', label='unrewarded stops')
ax.fill_between(
range(0, int(range_val / binsize) * 2),
meannstops - scipy.stats.sem(rewnstops, axis=1, nan_policy='omit'),
meannstops + scipy.stats.sem(rewnstops, axis=1, nan_policy='omit'),
alpha=0.5, color='slategray'
)                    
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 5))
ax.set_xticklabels(np.arange(-range_val, range_val + 1, 2))
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Time from stop (s)')
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
ax.plot(meanvelnonrew,color='k', label='unrewarded stops')
ax.fill_between(
range(0, int(range_val / binsize) * 2),
meanvelnonrew - scipy.stats.sem(velnonrew, axis=1, nan_policy='omit'),
meanvelnonrew + scipy.stats.sem(velnonrew, axis=1, nan_policy='omit'),
alpha=0.5, color='k'
)                    
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 5))
ax.set_xticklabels(np.arange(-range_val, range_val + 1, 2))
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Time from stop (s)')
ax.legend()
ax.set_ylabel('Velocity (cm/s)')
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<1)]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_props]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_nulls]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_props]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_nulls]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
#%%
eps = df_permsav.index.get_level_values("epoch_comparison").unique()
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
    shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)

df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)
ax.set_ylabel('Post reward cell proportion')
eps = [2,3,4]
y = 0.2
pshift=.03
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45)
ax.set_title('Post-reward cells',pad=100)
plt.savefig(os.path.join(savedst, 'postrew_cell_prop_per_an.svg'), 
        bbox_inches='tight')