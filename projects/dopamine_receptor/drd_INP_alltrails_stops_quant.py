"""zahra
sept 2024
"""

#%%
import os, sys, scipy, imageio, pandas as pd
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

# Define save path for PDF
condition = 'drd2ko'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'
savepth = os.path.join(savedst, f'{condition}_stops.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
from drd import get_moving_time_v3, get_stops

# fluorescence mean threshold
fluor_thres = 400

# Define source directory and mouse name
src = r'Y:\drd'
if condition=='drd2ko':
    mouse_name = 'e262'        
elif condition=='drd2':
    mouse_name = 'e256'
elif condition=='drd1':
    mouse_name = 'e255'

# days = [3,4,5,6,7,9]
days = [6,7,8,9]
# days = [13]
range_val, binsize = 6 , 0.2 # seconds
perirew_all_days = []; perinonrew_all_days = []
# Iterate through specified days
for dy in days:
    day_dir = os.path.join(src, mouse_name, str(dy))
    postrew_dff_all_planes = []; perirew_all_planes = []; perinonrew_all_planes = []    
    for root, dirs, files in os.walk(day_dir):
        planes = len([xx for xx in os.listdir(os.path.dirname(root)) if 'plane' in xx])
        for file in files:
            if 'plane' in root and file.endswith('roibyclick_F.mat'):
                f = loadmat(os.path.join(root, file))
                plane = int(root.split("plane")[1])
                eps = np.where(f['changeRewLoc'] > 0)[1]
                eps = np.append(eps, len(f['changeRewLoc'][0]))
                rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                plane = int(root.split("plane")[1][0])
                dFF_iscell = f['dFF']
                F_iscell = f['F']
                means = np.nanmean(F_iscell, axis=0)
                # remove dim cells
                dFF_iscell = dFF_iscell[:, means>fluor_thres]
                dFF_iscell_filtered = dFF_iscell.T
                dff_res = []
                perirew = []
                # find stops
                velocity = f['forwardvelALL'][0]
                veldf = pd.DataFrame({'velocity': velocity})
                velocity = np.hstack(veldf.rolling(5).mean().values)
                moving_middle,stop = get_moving_time_v3(velocity,5,10,10)
                pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
                nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
                        post_win_framesALL,velocity, f['rewardsALL'][0])
                nonrew_per_plane = np.zeros_like(f['changeRewLoc'][0])
                nonrew = np.ceil(nonrew/planes)
                nonrew_per_plane[nonrew.astype(int)] = 1
                
                rew_per_plane = np.zeros_like(f['changeRewLoc'][0])
                rew = np.ceil(rew/planes)
                rew_per_plane[rew.astype(int)] = 1

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
                    
                    # rewarded vs. non stops peri dff
                    _, meannstops, __, rewnstops = perireward_binned_activity(
                        dff, 
                        nonrew_per_plane, 
                        f['timedFF'][0], 
                        f['trialnum'][0], 
                        range_val, 
                        binsize,
                    )
                    _, meanrstops, __, rewrstops = perireward_binned_activity(
                        dff, 
                        rew_per_plane, 
                        f['timedFF'][0], 
                        f['trialnum'][0], 
                        range_val, 
                        binsize,
                    )
                    perirew.append([meannstops, rewnstops, meanrstops,rewrstops])
            
                dff_res = np.array(dff_res)
                # normalize post reward activity
                # clls = dff_res.shape[0]
                # prewin = 2
                # binss = np.ceil(prewin/binsize).astype(int)
                # bound = int(range_val/binsize)
                # postwin = 2 #s
                # postbound = np.ceil(postwin/binsize).astype(int)
                # meanrewall = np.array([perirew[cll][0]-np.nanmean(perirew[cll][0][(bound-binss):bound]) for cll in range(clls)])
                # postrew_dff = np.nanmean(meanrewall[:, bound:bound+postbound],axis=1)
                # # or quantile
                # # postrew_dff = np.nanquantile(meanrewall[:, bound:bound+postbound], 
                # #         .75, axis=1)
                # postrew_dff_all_planes.append(postrew_dff)                            
                clls = dff_res.shape[0]        
                subpl = int(np.ceil(np.sqrt(clls)))
                # nonrewarded stops
                fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))
                if clls > 1:
                    axes = axes.flatten()
                perirewcllnonrew = []; perirewcllrew = []
                for cll in range(clls):
                    if clls > 1:
                        ax = axes[cll]
                    else:
                        ax = axes  # single cell case

                    meanrew = perirew[cll][0]
                    rewall = perirew[cll][1]
                    perirewcllnonrew.append(meanrew)
                    ax.plot(meanrew, 'slategray', label = 'Unrewarded')
                    ax.fill_between(
                        range(0, int(range_val / binsize) * 2),
                        meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        alpha=0.5, color='slategray'
                    )                    
                    
                    meanrew = perirew[cll][2]
                    rewall = perirew[cll][3]
                    perirewcllrew.append(meanrew)
                    ax.plot(meanrew, 'steelblue', label = 'Rewarded')
                    ax.fill_between(
                        range(0, int(range_val / binsize) * 2),
                        meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        alpha=0.5, color='steelblue'
                    )                    
                    
                    ax.set_title(f'Cell {cll + 1}')
                    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
                    ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
                    ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
                    ax.spines[['top', 'right']].set_visible(False)
                    if cll==clls-1:
                        ax.legend()
                #save
                perinonrew_all_planes.append(perirewcllnonrew)
                perirew_all_planes.append(perirewcllrew)
                if clls > 1:
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'Peri- stops \n {mouse_name}, Day={dy}, Plane {plane}')
                plt.show()
                pdf.savefig(fig)
                plt.close(fig)

                if clls > 1:
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'Peri- rewarded stop \n {mouse_name}, Day={dy}, Plane {plane}')
                plt.show()
                pdf.savefig(fig)
                plt.close(fig)
    perirew_all_days.append(perirew_all_planes); perinonrew_all_days.append(perinonrew_all_planes)

pdf.close()
#%%
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
# average cell activity for one day

fig, ax = plt.subplots()
meanrew = np.vstack(perirew_all_planes)
ax.plot(np.nanmean(meanrew,axis=0), color='slategray',linewidth=3)
ax.fill_between(
    range(0, int(range_val / binsize) * 2),
    np.nanmean(meanrew,axis=0) - scipy.stats.sem(meanrew, axis=0, nan_policy='omit'),
    np.nanmean(meanrew,axis=0) + scipy.stats.sem(meanrew, axis=0, nan_policy='omit'),
    alpha=0.5, color='slategray'
)                    
ax.set_ylabel('$\Delta$ F/F')
ax.set_xlabel('Time from Conditioned Stimulus (s)')
ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(f'{condition}, {meanrew.shape[0]} cells')
ax.axvline(int(range_val / binsize), linestyle='--', color='k', linewidth=3)
plt.savefig(os.path.join(savedst, f'{condition}_av_trace.svg'), bbox_inches='tight')
#%%
# histogram of post rew activity per plane
arr = np.concatenate([np.concatenate(pr) for pr in postrew_dff_all_days])
arr = arr[arr<5] # remove outliers
bins=np.histogram(arr, bins=30)[1] #get the bin edges
plns = 4
fig,axes = plt.subplots(nrows=plns, sharex=True, figsize = (8,12))
for d,pr in enumerate(postrew_dff_all_days):
    for pln in range(plns):
        ax = axes[pln]
        try:
            pr[pln] = pr[pln][pr[pln]<5] # remove outliers
            ax.hist(pr[pln],bins=bins,label = f'Day {d:02d}', alpha=0.4)
            ax.set_title(f'Plane {pln}')
            ax.axvline(0, color='slategray', linestyle='--', linewidth=4)
            ax.spines[['top','right']].set_visible(False)                        
            ax.set_xlim([-1, .5])        
        except Exception as e:
            print(e)
        if pln==3: ax.legend(bbox_to_anchor=(1.01, 1.30))
            
    ax.set_xlabel('Mean $\Delta$ F/F-Baseline(pre-reward)')
    ax.set_ylabel('# cells')
        
    fig.suptitle(f'{condition} \n Post reward activity \n planes (0: superficial; 4: deep)')
    fig.tight_layout()

#%%
# all planes
# histogram of post rew activity per plane
arr = np.concatenate([np.concatenate(pr) for pr in postrew_dff_all_days])
arr = arr[arr<5] # remove outliers
bins=np.histogram(arr, bins=20)[1] #get the bin edges
plns = 4
fig,ax = plt.subplots(figsize = (8,7))
for d,pr in enumerate(postrew_dff_all_days):
    allplnpr = np.concatenate(pr)
    
    ax.hist(allplnpr,bins=bins,label = f'Day {d:02d}', alpha=0.6)        
    ax.axvline(0, color='slategray', linestyle='--', linewidth=4)
    ax.spines[['top','right']].set_visible(False)                        
    ax.set_xlim([-1, .5])            
ax.set_xlabel('Mean $\Delta$ F/F-Baseline(pre-reward)')
ax.set_ylabel('# cells')
ax.legend(bbox_to_anchor=(1.01, 1.01))
fig.suptitle(f'{condition} \n Post reward activity, all planes')
fig.tight_layout()
