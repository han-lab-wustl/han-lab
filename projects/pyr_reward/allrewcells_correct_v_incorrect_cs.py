
#%%
"""
zahra
april 2025
cosine similarity of correct vs incorrect tuning curves
all rew cells 
vs. com
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
from rewardcell import get_radian_position,reward_act_allrew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p'
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
saveto = saveddataset#rf'Z:\saved_datasets\radian_tuning_curves_prereward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
# iterate through all animals

tcs_correct_all=[]
tcs_fail_all=[]
coms = []
css = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        tcs_correct,tcs_fail,com_correct,cs=reward_act_allrew(ii,params_pth,\
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
                total_cells)
        tcs_correct_all.append(tcs_correct)
        tcs_fail_all.append(tcs_fail)
        coms.append(com_correct)
        css.append(cs)

pdf.close()

#%%

# cs b/wn correct and incorrect

# histogram
fig,ax = plt.subplots()
ax.hist(np.concatenate([xx[0] for xx in css if len(xx)>0]),bins=20)
ax.set_ylabel('# Reward cells')
ax.set_xlabel('Cosine similarity')
ax.set_title('Correct vs. incorrect tuning curves')
ax.spines[['top','right']].set_visible(False)

#%%
# com vs. cs
# 2 ep
eps = [0,1,2]
fig,ax = plt.subplots()
for ep in eps:
    cs = np.concatenate([xx[ep] for xx in css if len(xx)>ep])
    com = np.concatenate([xx[ep] for xx in coms if len(xx)>ep])
    ax.scatter(cs,com,alpha=0.3)
    ax.set_ylabel('COM per epoch')
    ax.set_xlabel('Cosine similarity')
    ax.set_title('Correct vs. incorrect tuning curves')
    ax.legend()

    ax.spines[['top','right']].set_visible(False)

# average ep
cs = np.concatenate([np.nanmean(xx,axis=0) for xx in css if xx.shape[0]>0])
com = np.concatenate([np.nanmean(xx,axis=0) for xx in coms if xx.shape[0]>0])
fig,ax = plt.subplots()
ax.scatter(cs,com,alpha=0.3)
ax.set_ylabel('COM (average of epochs)')
ax.set_xlabel('Cosine similarity')
ax.set_title('Correct vs. incorrect tuning curves')
ax.legend()

ax.spines[['top','right']].set_visible(False)

#%%
# pre vs. post vs. far?
df = pd.DataFrame()
cdf = conddf.copy()
cdf = cdf[(cdf.animals!='e217') & (cdf.optoep.values<2)]        

df['cosine_similarity'] = cs
bound = np.pi/4
df['cell_type'] = ['Pre' if (xx<0 and xx>-bound) else 'Post' if(xx>0 and xx<bound) else 'Far' for xx in com]
df['com'] = com

test = []
for ii in range(len(css)):
    if len(css[ii])>0:
        test.append([cdf.animals.values[ii]]*len(np.nanmean(css[ii],axis=0)))

df['animal'] = np.concatenate(test)
from itertools import combinations

s=9
df = df.groupby(['animal', 'cell_type']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize = (3,5))
sns.boxplot(x='cell_type', y='cosine_similarity', data=df, fill=False, color='peru',
        order=['Pre', 'Post', 'Far'])
sns.stripplot(x='cell_type', y='cosine_similarity', data=df, s=s, color='peru',alpha=0.7,
        order=['Pre', 'Post', 'Far'])
ax.spines[['top','right']].set_visible(False)
df = df.reset_index()

# Kruskal-Wallis test
groups = [df[df['cell_type'] == ct]['cosine_similarity'].dropna() for ct in ['Pre', 'Post', 'Far']]
kw_stat, kw_p = scipy.stats.kruskal(*groups)
print(f"Kruskal-Wallis H-statistic: {kw_stat:.3f}, p-value: {kw_p:.3e}")

# Plot within-animal lines
for animal in df['animal'].unique():
    sns.lineplot(
        x='cell_type', y='cosine_similarity',
        data=df[df.animal == animal],
        errorbar=None, color='dimgray', linewidth=2, alpha=0.7, ax=ax
    )

# --- Post hoc t-tests and annotation ---
cell_order = ['Pre', 'Post', 'Far']
pairs = list(combinations(cell_order, 2))
y_offset = 0.05  # vertical spacing between annotation lines
max_y = df['cosine_similarity'].max()
fs = 24  # font size for stars

for i, (grp1, grp2) in enumerate(pairs):
    data1 = df[df['cell_type'] == grp1]['cosine_similarity']
    data2 = df[df['cell_type'] == grp2]['cosine_similarity']
    stat, pval = scipy.stats.ranksums(data1, data2)  # paired t-test
    print(f"{grp1} vs {grp2}: p = {pval:.3e}")
    
    x1, x2 = cell_order.index(grp1), cell_order.index(grp2)
    y = max_y + (i-1) * y_offset

    # Draw the line
    ax.plot([x1, x1, x2, x2], [y - y_offset/2, y, y, y - y_offset/2], lw=1.5, color='k')

    # Add star annotation based on p-value
    ii = (x1 + x2) / 2
    if pval < 0.001:
        ax.text(ii, y + 0.002, "***", ha='center', fontsize=fs)
    elif pval < 0.01:
        ax.text(ii, y + 0.002, "**", ha='center', fontsize=fs)
    elif pval < 0.05:
        ax.text(ii, y + 0.003, "*", ha='center', fontsize=fs)
ax.set_xlabel('')
ax.set_ylabel('Cosine similarity')
ax.set_title('Correct vs. incorrect tuning curves')

plt.tight_layout()
plt.show()
#%%

# get examples of correct vs. fail
# take the first epoch and first cell?
# v take all cells
# per day per animal
animals =[xx for ii,xx in enumerate(conddf.animals.values) if (xx!='e217') & (conddf.optoep.values[ii]<2)]

plt.rc('font', size=16) 
dff_correct_per_an = []; dff_fail_per_an = [] # per cell, av epoch
for animal in np.unique(animals):
        dff_correct=[]; dff_fail=[]
        tcs_correct = []
        for ii,tcs_corr in enumerate(tcs_correct_all):
                if animals[ii]==animal:
                        if tcs_corr.shape[1]>0:
                                # all cells
                                # take average of epochs
                                tc= np.vstack(np.nanmean(tcs_corr[:,:,:],axis=0))
                                # tc = tcs_corr[0,0,:]
                                tcs_correct.append(tc)
                                # pre vs. post reward
                                #pre
                                # dff_correct.append(np.quantile(tc[:,int(bins/3):int(bins/2)],.9,axis=1))
                                #posts
                                dff_correct.append(np.quantile(tc[:,int(bins/2):],.9,axis=1))


        tcs_fail = []
        for ii,tcs_f in enumerate(tcs_fail_all):
                if animals[ii]==animal:
                        if tcs_f.shape[1]>0:
                                # tc = tcs_f[0,0,:]
                                # all cells
                                tc= np.vstack(np.nanmean(tcs_f[:,:,:],axis=0))
                                if np.sum(np.isnan(tc))==0:
                                        tcs_fail.append(tc)
                                        # pre
                                        # dff_fail.append(np.quantile(tc[:,int(bins/3):int(bins/2)],.9,axis=1))
                                        # post
                                        dff_fail.append(np.quantile(tc[:,int(bins/2):],.9,axis=1))
        dff_correct_per_an.append(dff_correct)
        dff_fail_per_an.append(dff_fail)
        fig, axes=plt.subplots(ncols=2, nrows=2,sharex=True,figsize=(8,15), 
                               height_ratios=[3.5,1])
        axes=axes.flatten()
        ax=axes[0]
        ax.imshow(np.vstack(tcs_correct)**.6,vmin=0,vmax=1.5)
        ax.axvline(45,color='w', linestyle='--')
        bins=90
        ax.set_xticks(np.arange(0,bins,30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
        ax.set_ylabel('Epochs')
        ax.set_xlabel('Reward-relative distance ($\Theta$)')
        ax.set_title(f'{animal}\nPre-reward cells\nCorrect')
        try: # if no fails
                ax=axes[1]
                im=ax.imshow(np.vstack(tcs_fail)**.6,vmin=0,vmax=1.5)
                ax.axvline(45,color='w', linestyle='--')
                ax.set_xticks(np.arange(0,bins,30))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
                ax.set_title('Incorrect')
        except Exception as e:
                print(e)
        cbar=fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('$\Delta$ F/F', rotation=270, labelpad=15)

        # mean 
        ax=axes[2]
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
        try:
                ax=axes[3]
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
        except Exception as e:
                print(e)
# plt.savefig(os.path.join(savedst, 'pre_rew_correctvfail_mean.svg'),bbox_inches='tight')
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
y=.3
pshift=.04
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Pre-reward cells',pad=50)
#%%
########## weird results, recalc tuning curve diff as above ##########
plt.rc('font', size=16)          # controls default text sizes
bigdf=pd.concat(dfs)

# average
bigdf=bigdf.groupby(['animal', 'epoch', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
# only < 4 epochs
bigdf=bigdf[bigdf.epoch<4]
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
sns.stripplot(x='trial_type', y='mean_tc', data=bigdf,dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
            s=s,alpha=0.7)
sns.barplot(x='trial_type', y='mean_tc', data=bigdf, fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'})

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\int$ tuning curve ($\Delta F/F$)')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_tc']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_tc']
t,pval = scipy.stats.ttest_rel(cor,incor)
# statistical annotation       
ii=0.5
y=6
pshift=.01
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Pre-reward cells',pad=20)
# plt.savefig(os.path.join(savedst, 'prerew_trial_type.svg'),bbox_inches='tight')