
#%%
"""
zahra
july 2024
quantify reward-relative cells near reward
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position,reward_act_prerew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'pre_rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_prereward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
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
saveto = rf'Z:\saved_datasets\radian_tuning_curves_prereward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
# iterate through all animals
dfs = []
tcs_correct_all=[]
tcs_fail_all=[]

for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df,tcs_correct,tcs_fail=reward_act_prerew(ii,params_pth,\
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
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
# animals_test=['z9']
# option to pick 'pre' or 'post' reward activity
activity_window = 'pre'  # options: 'pre' or 'post'

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
    vmax = np.nanquantile(tc, 0.999)  # 95th percentile
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
    fig.suptitle('Pre-reward cells')
#     plt.savefig(os.path.join(savedst, f'{animal}_pre_rew_correctvfail.svg'),bbox_inches='tight')

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
df['cell_type'] = ['Pre']*len(df)
df_post = pd.read_csv(r'Z:\condition_df\post_trialtype.csv')
df_post['cell_type'] = ['Post']*len(df_post)
df_farpost = pd.read_csv(r'Z:\condition_df\farpost_trialtype.csv')
df_farpost['cell_type'] = ['Far post']*len(df_farpost)
df_farpre = pd.read_csv(r'Z:\condition_df\farpre_trialtype.csv')
df_farpre['cell_type'] = ['Far pre']*len(df_farpre)

bigdf=pd.concat([df,df_post,df_farpost,df_farpre])
cell_order = ['Pre', 'Post', 'Far pre', 'Far post']

# average
bigdf=bigdf.groupby(['animal', 'trial_type', 'cell_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
s=13
fig,ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7,    order=cell_order)
sns.barplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
            order=cell_order)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$ rel. to rew.')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_dff']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_dff']
t,pval = scipy.stats.wilcoxon(cor,incor)

# ans = bigdf.animal.unique()
# for i in range(len(ans)):
#     for j,tr in enumerate(np.unique(bigdf.cell_type.values)):
#         testdf= bigdf[(bigdf.animal==ans[i]) & (bigdf.cell_type==tr)]
#         ax = sns.lineplot(x='trial_type', y='mean_dff', 
#         data=testdf,
#         errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel

# 1) Two-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf,
    depvar='mean_dff',
    subject='animal',
    within=['trial_type','cell_type']
).fit()
print(aov)    # F-stats and p-values for main effects and interaction

# 2) Post-hoc paired comparisons: correct vs incorrect within each cell_type
posthoc = []
for ct in cell_order:
    sub = bigdf[bigdf['cell_type']==ct]
    cor = sub[sub['trial_type']=='correct']['mean_dff']
    inc = sub[sub['trial_type']=='incorrect']['mean_dff']
    t, p_unc = ttest_rel(cor, inc)
    posthoc.append({
        'cell_type': ct,
        't_stat':    t,
        'p_uncorrected': p_unc
    })

posthoc = pd.DataFrame(posthoc)
# Bonferroni
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected'] * len(posthoc), 1.0)
print(posthoc)
# map cell_type → x-position
xpos = {ct: i for i, ct in enumerate(cell_order)}
for _, row in posthoc.iterrows():
    x = xpos[row['cell_type']]
    y = bigdf[
        (bigdf['cell_type']==row['cell_type'])
    ]['mean_dff'].max() + 0.1  # just above the tallest bar
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=42)
    if p>0.05:
        ax.text(x, y, f'p={p:.2g}', ha='center', va='bottom', fontsize=12)

# remove the old legend
ax.legend_.remove()
# get new handles & labels
handles, labels = ax.get_legend_handles_labels()
# place legend outside on the right
ax.legend(handles, labels,
          loc='upper left',
          bbox_to_anchor=(1.02, 1),    # (x-offset, y-offset) in axes coords
          borderaxespad=0.)            # remove padding

# Example interpretation (fill in with your numbers)
# trial_type (Num DF = 1, Den DF = 9, F = 12.3, p = 0.006)
# -- There is a significant main effect of trial type: across all cell types, mean ΔF/F is different on correct vs. incorrect trials.

# cell_type (Num DF = 3, Den DF = 27, F = 8.7, p < 0.001)
# -- There is a significant main effect of cell type: some cell types have higher overall ΔF/F than others, regardless of trial outcome.

# trial_type × cell_type (Num DF = 3, Den DF = 27, F = 4.2, p = 0.014)
# -- The interaction is significant: the difference between correct vs. incorrect ΔF/F depends on which cell type you look at.

# Because the interaction is significant, you should then examine post-hoc tests (e.g., the paired comparisons you ran) to see for each cell type whether correct vs. incorrect is significant.
plt.savefig(os.path.join(savedst, 'allcelltype_trialtype.svg'),bbox_inches='tight')
