
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
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,reward_act_farrew, get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'near_rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_farreward_all.p'
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
saveto =saveddataset
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
        df,tcs_correct,tcs_fail=reward_act_farrew(ii,params_pth,\
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
# per day per animal
plt.rc('font', size=16) 
tcs_correct = []
for tcs_corr in tcs_correct_all:
        if tcs_corr.shape[1]>0:
                tc = tcs_corr[0,0,:]
                tcs_correct.append(tc)
tcs_fail = []
for tcs_f in tcs_fail_all:
        if tcs_f.shape[1]>0:
                tc = tcs_f[0,0,:]
                if np.sum(np.isnan(tc))==0:
                        tcs_fail.append(tc)
        
fig, axes=plt.subplots(ncols=2,sharex=True)
ax=axes[0]
# sort by peak location
alltcs = np.vstack(tcs_correct)
peak = np.argmax(alltcs,axis=1)
alltcs = alltcs[np.argsort(peak)]
ax.imshow(alltcs**.6,vmin=0,vmax=1.5)
ax.axvline(45,color='w', linestyle='--')
bins=90
ax.set_xticks(np.arange(0,bins,30))
ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
ax.set_ylabel('Epochs')
ax.set_xlabel('Reward-relative distance ($\Theta$)')
ax.set_title('Far-reward cells\nCorrect')
ax=axes[1]
alltcs = np.vstack(tcs_fail)
peak = np.argmax(alltcs,axis=1)
alltcs = alltcs[np.argsort(peak)]

im=ax.imshow(np.vstack(tcs_fail)**.6,vmin=0,vmax=1.5)
ax.axvline(45,color='w', linestyle='--')
ax.set_xticks(np.arange(0,bins,30))
ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
ax.set_title('Incorrect')

cbar=fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel('$\Delta$ F/F', rotation=270, labelpad=15)

plt.savefig(os.path.join(savedst, 'far_rew_correctvfail.svg'),bbox_inches='tight')
#%%
# mean 
fig, axes=plt.subplots(figsize=(8,2.5),ncols=2,sharey=True,sharex=True)
ax=axes[0]
m = np.nanmean(np.vstack(tcs_correct),axis=0)
ax.plot(m, color='seagreen')
ax.fill_between(
range(0, np.vstack(tcs_correct).shape[1]),
m - scipy.stats.sem(np.vstack(tcs_correct), axis=0, nan_policy='omit'),
m + scipy.stats.sem(np.vstack(tcs_correct), axis=0, nan_policy='omit'),
alpha=0.5, color='seagreen'
)             
ax.axvline(45,color='k', linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Correct')
ax.set_ylabel('$\Delta$ F/F')

bins=90
ax=axes[1]
m = np.nanmean(np.vstack(tcs_fail),axis=0)
ax.plot(m, color='firebrick')
ax.fill_between(
range(0, np.vstack(tcs_fail).shape[1]),
m - scipy.stats.sem(np.vstack(tcs_fail), axis=0, nan_policy='omit'),
m + scipy.stats.sem(np.vstack(tcs_fail), axis=0, nan_policy='omit'),
alpha=0.5, color='firebrick'
)             
ax.axvline(45,color='k', linestyle='--')
ax.set_xticks(np.arange(0,bins,30))
ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)

ax.spines[['top','right']].set_visible(False)

ax.set_xlabel('Reward-relative distance ($\Theta$)')
ax.set_title('Incorrect')
plt.savefig(os.path.join(savedst, 'far_rew_correctvfail_mean.svg'),bbox_inches='tight')

#%%
plt.rc('font', size=16)          # controls default text sizes
bigdf=pd.concat(dfs)

# average
bigdf=bigdf.groupby(['animal', 'epoch', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()

# plot
s=10
fig,ax = plt.subplots(figsize=(5,5))
sns.stripplot(x='trial_type', y='mean_tc', hue='epoch', data=bigdf,dodge=True,
            s=s,alpha=0.5)
sns.barplot(x='trial_type', y='mean_tc', hue='epoch', data=bigdf, fill=False)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Average $\Delta F/F$ of tuning curve')
ax.set_xlabel('')


# average
bigdf=bigdf.groupby(['animal', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
s=13
fig,ax = plt.subplots(figsize=(2,5))
sns.stripplot(x='trial_type', y='mean_tc', hue='trial_type', data=bigdf,dodge=True,
            s=s,alpha=0.7, palette={'correct':'seagreen', 'incorrect': 'firebrick'})
sns.barplot(x='trial_type', y='mean_tc', hue='trial_type', data=bigdf, fill=False,
            palette={'correct':'seagreen', 'incorrect': 'firebrick'})

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\int$ tuning curve ($\Delta F/F$)')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_tc']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_tc']
t,pval = scipy.stats.wilcoxon(cor,incor)
# statistical annotation       
ii=0.5
y=8
pshift=.01
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Far-reward cells',pad=30)
plt.savefig(os.path.join(savedst, 'farrew_trial_type.svg'),bbox_inches='tight')