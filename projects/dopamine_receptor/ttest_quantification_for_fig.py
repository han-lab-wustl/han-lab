"""
zahra
sept 2024
"""
#%%
import os, sys, scipy, imageio, pandas as pd, re
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf

# Add custom path for MATLAB functions
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') 

# Formatting for figures
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)
from projects.dopamine_receptor.drd import extract_plane_number
from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
# fluorescence mean threshold
fluor_thres = 100
# Define source directory and mouse name
src = r'Y:\drd'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'
# drd1
mice = ['e256', 'e253', 'e262', 'e261'] #'e255', 'e254',
        
conditions = ['D2','D2','CRISPR-KO D2','CRISPR-KO D2']
# list(np.arange(3,13)), list(np.arange(1,9)),
        
days_s = [list(np.arange(3,16)), list(np.arange(1,10)),
        list(np.arange(3,10)), list(np.arange(1,5))]
days_to_analyse = 5
# days = [3,4,5,6,7,8,9,10,12]
range_val, binsize = 6 , 0.2 # seconds
postrew_dff_all_mice = []
# Iterate through specified days
for ii,mouse_name in enumerate(mice):
    days = days_s[ii]
    postrew_dff_all_days = []
    condition = conditions[ii]
    for dy in days[-days_to_analyse:]: # only last 2 days for now
        day_dir = os.path.join(src, mouse_name, str(dy))
        postrew_dff_all_planes = []
        for root, dirs, files in os.walk(day_dir):
            for file in files:
                if 'plane' in root and file.endswith('roibyclick_F.mat'):
                    f = loadmat(os.path.join(root, file))
                    print(os.path.join(root, file))
                    plane = extract_plane_number(os.path.join(root, file))
                    
                    eps = np.where(f['changeRewLoc'] > 0)[1]
                    eps = np.append(eps, len(f['changeRewLoc'][0]))
                    rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                    dFF_iscell = f['dFF']
                    F_iscell = f['F']
                    means = np.nanmean(F_iscell, axis=0)
                    # remove dim cells
                    dFF_iscell = dFF_iscell[:, means>fluor_thres]
                    dFF_iscell_filtered = dFF_iscell.T
                    dff_res = []
                    perirew = []
                    if len(dFF_iscell_filtered)>0:
                        # Iterate through cells
                        for cll in range(dFF_iscell_filtered.shape[0]):
                            X = np.array([f['forwardvel'][0]]).T 
                            X = sm.add_constant(X)
                            y = dFF_iscell_filtered[cll, :]
                            
                            model = sm.GLM(y, X, family=sm.families.Gaussian())
                            result = model.fit()
                            dff_res.append(result.resid_pearson)
                            
                            dff = result.resid_pearson
                            dffdf = pd.DataFrame({'dff': dff})
                            dff = np.hstack(dffdf.rolling(5).mean().values)
                        
                            normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
                                dff, 
                                f['solenoid2'][0].astype(int), 
                                f['timedFF'][0], 
                                f['trialnum'][0], 
                                range_val, 
                                binsize,
                            )
                            perirew.append([meanrewdFF, rewdFF])
                    
                        dff_res = np.array(dff_res)
                        # normalize post reward activity
                        clls = dff_res.shape[0]
                        prewin = 1
                        binss = np.ceil(prewin/binsize).astype(int)
                        bound = int(range_val/binsize)
                        postwin = 2 #s
                        postbound = np.ceil(postwin/binsize).astype(int)
                        meanrewall = np.array([perirew[cll][0]-np.nanmean(perirew[cll][0][(bound-binss):bound]) for cll in range(clls)])
                        if condition=='drd2': #or condition=='drd2ko'
                            postrew_dff = np.nanmean(meanrewall[:, bound:bound+postbound],axis=1)
                        # # or quantile
                        else:
                            postrew_dff = np.nanquantile(meanrewall[:, bound:bound+postbound], 
                                .75, axis=1)
                        # area under curve
                        # postrew_dff = [np.trapz(xx, dx=5) for xx in meanrewall]
                        postrew_dff_all_planes.append(postrew_dff)
                    
                    
        postrew_dff_all_days.append(postrew_dff_all_planes)
    postrew_dff_all_mice.append(postrew_dff_all_days)

# ```

# ### Changes Made:
# 1. **Imports**: Consolidated imports at the beginning.
# 2. **File Path Handling**: Added handling to find `masks.jpg` specifically.
# 3. **GLM Fit and Result Handling**: Generalized the GLM fitting process for each cell.
# 4. **Conditional Check for Single Cell (clls == 1)**: 
#     - Added logic to handle plotting for `clls == 1`.
#     - If `clls == 1`, the plotting is adjusted to a single subplot without creating a grid.
# 5. **Legends and Titles**: Adjusted legend placement and subplot handling for clarity.

# This script handles the specific case where `clls = 1` and ensures proper plotting regardless of the number of cells processed.
#%%
def normalize_to_range(values, new_min=-1, new_max=1):
    """
    Normalize an array of values to a specified range [-1, 1].
    """
    old_min, old_max = np.min(values), np.max(values)
    normalized_values = new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)
    return normalized_values

# quantification
ms = []
for dd, pr_dy in enumerate(postrew_dff_all_mice):
    for d, pr in enumerate(pr_dy):
        if len(pr)>0:
            allplnpr = np.concatenate(pr)
            allplnpr = allplnpr[allplnpr<10]
            # average of all cells 
            meansuppression = np.nanmean(normalize_to_range(allplnpr, new_min=-1, new_max=1))*-1#/np.nanmin(allplnpr)
            ms.append(meansuppression)
condition_df = np.concatenate([[xx]*days_to_analyse for xx in conditions])
df = pd.DataFrame(ms, columns = ['mean_dff_postrew'])
df['condition'] = condition_df
df['animal'] = np.concatenate([[xx]*days_to_analyse for xx in mice])
#
import seaborn as sns
dfan = df.groupby(['animal', 'condition']).mean(numeric_only=True)
dfan.reset_index()
dfan=dfan.sort_values(by=['condition'])
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='condition',y='mean_dff_postrew',data=dfan, s=16,
            hue='condition', palette='colorblind',ax=ax,alpha=.6)
sns.barplot(x='condition',y='mean_dff_postrew',data=dfan, errorbar='se',
            hue='condition', palette='colorblind', fill=False,ax=ax,
            linewidth=4, errwidth=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Modulation Index')
ax.set_xlabel('')
# ax.set_xticklabels(['D1', 'D2'])

x1= dfan.loc[dfan.index.get_level_values('condition')=='D2', 
        'mean_dff_postrew'].values
x2= dfan.loc[dfan.index.get_level_values('condition')=='CRISPR-KO D2',
        'mean_dff_postrew'].values
t,pval=scipy.stats.ttest_ind(x1,x2)

ax.set_title(f'ttest ind pval: {pval:.4f}')
plt.savefig(os.path.join(savedst, 'ko_per_mouse_ttest_mod_index.svg'), bbox_inches='tight')

#%% 
# quant with cells only
# quantification
ms = []
condarr = []; anarr=[]
for dd, pr_dy in enumerate(postrew_dff_all_mice):
    pr = pr_dy[-1]
    cond = conditions[dd]
    an = mice[dd]
    if len(pr)>0:
        allplnpr = np.concatenate(pr)
        allplnpr = allplnpr[allplnpr<5]        
        meansuppression = normalize_to_range(allplnpr, new_min=-1, new_max=1)*-1
        ms.append(meansuppression)
        condarr.append([cond]*len(meansuppression))
        anarr.append([an]*len(meansuppression))
df = pd.DataFrame(np.concatenate(ms), columns = ['mean_dff_postrew'])
df['condition'] = np.concatenate(condarr)
df['animal'] = np.concatenate(anarr)
#%%
import seaborn as sns
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='condition',y='mean_dff_postrew',data=df, s=8,
            hue='condition', palette='colorblind',ax=ax,alpha=.3)
sns.barplot(x='condition',y='mean_dff_postrew',data=df, errorbar='se',
            hue='condition', palette='colorblind', fill=False,ax=ax,
            linewidth=3, errwidth=3)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Modulation Index')
ax.set_xlabel('')
# ax.set_xticklabels(['D1', 'D2'])

x1= df.loc[df.condition=='D2', 'mean_dff_postrew'].values
x2= df.loc[df.condition=='CRISPR-KO D2', 'mean_dff_postrew'].values
t,pval=scipy.stats.ranksums(x1,x2)

ax.set_title(f'per cell, 1 day, ranksum pval: {pval:.7f}')
plt.savefig(os.path.join(savedst, 'per_cell_mod_index.svg'), bbox_inches='tight')
