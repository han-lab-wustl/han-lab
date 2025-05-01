
#%%
"""
zahra
2025
dff by trial type
"""
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
from projects.pyr_reward.rewardcell import get_radian_position,reward_act_nearrew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'near_rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
num_epochs = [] 
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
lasttr=8 #  last trials
bins=90
saveto = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
tcs_correct_all=[]
tcs_fail_all=[]
# iterate through all animals
dfs = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df,tcs_correct,tcs_fail=reward_act_nearrew(ii,params_pth,\
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,
                pvals,
                total_cells)
        dfs.append(df)
        tcs_correct_all.append(tcs_correct)
        tcs_fail_all.append(tcs_fail)
pdf.close()

#%%
# get examples of correct vs. fail
# take the first epoch and first cell?
# v take all cells
# per day per animal
# settings
plt.rc('font', size=24) 
animals = [xx for ii, xx in enumerate(conddf.animals.values) if (xx != 'e217') & (conddf.optoep.values[ii] < 2)]
animals_test = np.unique(animals)
# animals_test=['e201']
# option to pick 'pre' or 'post' reward activity
activity_window = 'post'  # options: 'pre' or 'post'

dff_correct_per_an = []
dff_fail_per_an = []

for animal in animals_test:
    dff_correct = []
    dff_fail = []
    tcs_correct = []
    bins = 90

    for ii, tcs_corr in enumerate(tcs_correct_all):
        if animals[ii] == animal and tcs_corr.shape[1] > 0:
            tc = np.vstack(np.nanmean(tcs_corr, axis=0))
            tcs_correct.append(tc)
            # choose pre or post reward
            if activity_window == 'pre':
                dff_correct.append(np.quantile(tc[:, int(bins/3):int(bins/2)], .9, axis=1))
            else:
                dff_correct.append(np.quantile(tc[:, int(bins/2):], .9, axis=1))

    tcs_fail = []
    for ii, tcs_f in enumerate(tcs_fail_all):
        if animals[ii] == animal and tcs_f.shape[1] > 0:
            tc = np.vstack(np.nanmean(tcs_f, axis=0))
            tcs_fail.append(tc)
            if np.sum(np.isnan(tc)) == 0:                
                if activity_window == 'pre':
                    dff_fail.append(np.quantile(tc[:, int(bins/3):int(bins/2)], .9, axis=1))
                else:
                    dff_fail.append(np.quantile(tc[:, int(bins/2):], .9, axis=1))

    dff_correct_per_an.append(dff_correct)
    dff_fail_per_an.append(dff_fail)

    # plotting
    fig, axes = plt.subplots(
        ncols=3, nrows=2, figsize=(10, 12),
        gridspec_kw={'height_ratios': [2, 1], 'width_ratios':[1, 1, 0.05]},
        constrained_layout=True
    )
    axes = axes.flatten()

    # --- Heatmaps
    ax = axes[0]
    tc = np.vstack(tcs_correct)
    vmin = 0
    vmax = np.nanquantile(tc, 0.99)  # 95th percentile
    peak_bins = np.argmax(tc, axis=1)
    sort_idx = np.argsort(peak_bins)
    im = ax.imshow(tc[sort_idx]**.6, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axvline(bins//2, color='w', linestyle='--')
    ax.set_xticks(np.arange(0, bins, 30))
    ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2), rotation=45)
    ax.set_ylabel('Cells (sorted)')
    ax.set_xlabel('Reward-relative distance ($\Theta$)')
    ax.set_title(f'{animal}\nCorrect Trials')

    try:
        ax = axes[1]
        tc = np.vstack(tcs_fail)
        # sort by correct cells
        im2 = ax.imshow(tc[sort_idx]**.6, vmin=vmin, vmax=vmax, aspect='auto')
        ax.axvline(bins//2, color='w', linestyle='--')
        ax.set_xticks(np.arange(0, bins, 30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2))
        ax.set_title('Incorrect Trials')
    except Exception as e:
        print(f"No failed trials for {animal}: {e}")

    # --- Colorbar
    cbar_ax = axes[2]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('$\Delta$ F/F', rotation=270, labelpad=15)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

    # --- Mean traces
    ax = axes[3]
    m = np.nanmean(np.vstack(tcs_correct), axis=0)
    vmin = 0
    vmax = np.nanmax(m)+np.nanmax(m)/2  # 95th percentile
    sem = scipy.stats.sem(np.vstack(tcs_correct), axis=0, nan_policy='omit')
    ax.plot(m, color='seagreen')
    ax.fill_between(np.arange(m.size), m - sem, m + sem, color='seagreen', alpha=0.5)
    ax.axvline(bins//2, color='k', linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('$\Delta$ F/F')
    ax.set_ylim(vmin, vmax)
    ax.set_title('Correct Mean')

    try:
        ax = axes[4]
        m = np.nanmean(np.vstack(tcs_fail), axis=0)
        sem = scipy.stats.sem(np.vstack(tcs_fail), axis=0, nan_policy='omit')
        ax.plot(m, color='firebrick')
        ax.fill_between(np.arange(m.size), m - sem, m + sem, color='firebrick', alpha=0.5)
        ax.axvline(bins//2, color='k', linestyle='--')
        ax.set_xticks(np.arange(0, bins, 30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Reward-relative distance ($\Theta$)')
        ax.set_ylabel('$\Delta$ F/F')
        ax.set_ylim(vmin, vmax)
        ax.set_title('Incorrect Mean')
    except Exception as e:
        print(f"No failed trials mean plot for {animal}: {e}")

    axes[5].axis('off')  # turn off the last unused axis (bottom-right)
    fig.suptitle('Post-reward cells')
    # plt.savefig(os.path.join(savedst, f'{animal}_post_rew_correctvfail.svg'),bbox_inches='tight')

#%%

# recalculate tc
animals_unique = np.unique(animals)
df=pd.DataFrame()
correct = np.concatenate([np.concatenate(xx) for xx in dff_correct_per_an])
incorrect = np.concatenate([np.concatenate(xx) for xx in dff_fail_per_an])
df['mean_dff'] = np.concatenate([correct,incorrect])
df['trial_type']=np.concatenate([['correct']*len(correct),['incorrect']*len(incorrect)])
ancorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_correct_per_an)])
anincorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_fail_per_an)])
df['animal'] = np.concatenate([ancorr, anincorr])
bigdf=df
# average
bigdf=bigdf.groupby(['animal', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
s=13
fig,ax = plt.subplots(figsize=(2,5))
sns.stripplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7)
sns.barplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'})

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Post-reward mean tuning curve ($\Delta F/F$)')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_dff']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_dff']
t,pval = scipy.stats.wilcoxon(cor,incor)
ans = bigdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='trial_type', y='mean_dff', 
    data=bigdf[bigdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

# statistical annotation       
ii=0.5
y=.4
pshift=.04
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Post-reward cells',pad=50)
df.to_csv(r'Z:\condition_df\post_trialtype.csv')

# plt.savefig(os.path.join(savedst, 'postrew_trial_type.svg'),bbox_inches='tight')