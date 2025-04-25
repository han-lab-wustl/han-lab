
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
from projects.pyr_reward.rewardcell import get_radian_position,reward_act_farrew, get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'pre_far_rew.pdf')
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
cell_type='pre'
# iterate through all animals
dfs = []
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                df,tcs_correct,tcs_fail=reward_act_farrew(ii,cell_type,params_pth,\
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
animals =[xx for ii,xx in enumerate(conddf.animals.values) if (xx!='e217') & (conddf.optoep.values[ii]<2)]
animals_test = np.unique(animals)
plt.rc('font', size=16) 
dff_correct_per_an = []; dff_fail_per_an = [] # per cell, av epoch
for animal in animals_test:
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
        tc = np.vstack(tcs_correct)
        peak_bins = np.argmax(tc, axis=1)
        sort_idx = np.argsort(peak_bins)
        ax.imshow(tc[sort_idx]**.6,vmin=0,vmax=1.5)
        ax.axvline(45,color='w', linestyle='--')
        bins=90
        ax.set_xticks(np.arange(0,bins,30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
        ax.set_ylabel('Epochs')
        ax.set_xlabel('Reward-relative distance ($\Theta$)')
        ax.set_title(f'{animal}\nFar-reward cells\nCorrect')
        try: # if no fails
                ax=axes[1]
                tc = np.vstack(tcs_fail)
                peak_bins = np.argmax(tc, axis=1)
                sort_idx = np.argsort(peak_bins)
                ax.imshow(tc[sort_idx]**.6,vmin=0,vmax=1.5)
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
        vmax = np.nanquantile(np.vstack(tcs_correct),.95)
        vmin = 0
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
        ax.set_ylim([vmin,vmax])
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
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2))
                ax.spines[['top','right']].set_visible(False)
                ax.set_xlabel('Reward-relative distance ($\Theta$)')
                ax.set_title('Incorrect')
                ax.set_ylim([vmin,vmax])
        except Exception as e:
                print(e)
# plt.savefig(os.path.join(savedst, 'post_rew_correctvfail.svg'),bbox_inches='tight')
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
y=.9
pshift=y/10
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Far-reward cells',pad=50)

# plt.savefig(os.path.join(savedst, 'farrew_trial_type.svg'),bbox_inches='tight')