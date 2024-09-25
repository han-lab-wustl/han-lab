"""zahra
sept 2024
"""

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
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
savepth = os.path.join(savedst, f'{condition}.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# Define source directory and mouse name
src = r'Y:\drd'
mouse_name = 'e262'
# days = [3,4,5,6,7,9]
# days = [3,4,5,6,7,8,9,10,12]
days = [7]
range_val, binsize = 7 , 0.2 # seconds
postrew_dff_all_days = []
# Iterate through specified days
for dy in days:
    day_dir = os.path.join(src, mouse_name, str(dy))
    postrew_dff_all_planes = []
    for root, dirs, files in os.walk(day_dir):
        for file in files:
            if 'plane' in root and file.endswith('roibyclick_F.mat'):
                f = loadmat(os.path.join(root, file))
                plane = int(root.split("plane")[1])
                ops = np.load(os.path.join(root, 'ops.npy'), allow_pickle=True)

                # Filename pattern to match
                target_filename = 'masks.jpg'
                blotches_file_path = None
                
                # Check for the blotches.jpg file
                for file in os.listdir(root):
                    if os.path.isfile(os.path.join(root, file)) and file.endswith(target_filename):
                        blotches_file_path = os.path.join(root, file)
                        break

                eps = np.where(f['changeRewLoc'] > 0)[1]
                eps = np.append(eps, len(f['changeRewLoc'][0]))
                rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                plane = int(root.split("plane")[1][0])
                dFF_iscell = f['dFF']
                F_iscell = f['F']
                means = np.nanmean(F_iscell, axis=0)
                # remove dim cells
                dFF_iscell = dFF_iscell[:, means>450]
                dFF_iscell_filtered = dFF_iscell.T
                dff_res = []
                perirew = []

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
                prewin = 2
                binss = np.ceil(prewin/binsize).astype(int)
                bound = int(range_val/binsize)
                postwin = 2 #s
                postbound = np.ceil(postwin/binsize).astype(int)
                meanrewall = np.array([perirew[cll][0]-np.nanmean(perirew[cll][0][(bound-binss):bound]) for cll in range(clls)])
                postrew_dff = np.nanmean(meanrewall[:, bound:bound+postbound],axis=1)
                # or quantile
                # postrew_dff = np.nanquantile(meanrewall[:, bound:bound+postbound], 
                #         .75, axis=1)
                postrew_dff_all_planes.append(postrew_dff)
                
                # Plot mean image
                fig, ax = plt.subplots()
                image = imageio.imread(blotches_file_path)
                ax.imshow(image)
                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close(fig)
                
                if clls == 1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(dff_res[0, :])
                    ax.set_title('Cell 1')
                    pdf.savefig(fig)
                    plt.close(fig)
                else:
                    subpl = int(np.ceil(np.sqrt(clls)))
                    fig, axes = plt.subplots(subpl, subpl, figsize=(20, 10))
                    axes = axes.flatten()

                    for cll in range(clls):
                        axes[cll].plot(dff_res[cll, :])
                        axes[cll].set_title(f'Cell {cll + 1}')

                    for i in range(clls, len(axes)):
                        axes[i].axis('off')
                    
                    fig.suptitle(f'GLM residuals \n {mouse_name}, Day={dy}, Plane {plane}')
                    pdf.savefig(fig)
                    plt.close(fig)
            
                subpl = int(np.ceil(np.sqrt(clls)))
                fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))
                if clls > 1:
                    axes = axes.flatten()
                for cll in range(clls):
                    if clls > 1:
                        ax = axes[cll]
                    else:
                        ax = axes  # single cell case

                    meanrew = perirew[cll][0]
                    rewall = perirew[cll][1]
                    ax.plot(meanrew, 'slategray', label='early_trials')
                    ax.fill_between(
                        range(0, int(range_val / binsize) * 2),
                        meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        alpha=0.5, color='slategray'
                    )                    
                    
                    ax.set_title(f'Cell {cll + 1}')
                    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
                    ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
                    ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
                    ax.spines[['top', 'right']].set_visible(False)
                
                if clls > 1:
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'Peri-reward \n {mouse_name}, Day={dy}, Plane {plane}')
                plt.show()
                pdf.savefig(fig)
                plt.close(fig)
    postrew_dff_all_days.append(postrew_dff_all_planes)

pdf.close()
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
