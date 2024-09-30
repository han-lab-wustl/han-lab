
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
condition='drd1'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
savepth = os.path.join(savedst, f'{condition}_allep_earlyvlate.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late

# Define source directory and mouse name
src = r'Y:\drd'
mouse_name = 'e255'
days = [10]
range_val, binsize = 6, 0.2 # seconds

# Iterate through specified days
for dy in days:
    day_dir = os.path.join(src, mouse_name, str(dy))
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
                
                    early_v_late = perireward_binned_activity_early_late(
                        dff, 
                        f['solenoid2'][0].astype(int), 
                        f['timedFF'][0], 
                        f['trialnum'][0], 
                        range_val, 
                        binsize,
                        early_trial=2, 
                        late_trial=8
                    )
                    perirew.append([
                        [early_v_late['first_5']['meanrewdFF'], early_v_late['first_5']['rewdFF']],
                        [early_v_late['last_5']['meanrewdFF'], early_v_late['last_5']['rewdFF']]
                    ])
            
                dff_res = np.array(dff_res)
                
                # Plot mean image
                fig, ax = plt.subplots()
                image = imageio.imread(blotches_file_path)
                ax.imshow(image)
                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close(fig)
            
                clls = dff_res.shape[0]
                
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

                    meanrew = perirew[cll][0][0]
                    rewall = perirew[cll][0][1]
                    ax.plot(meanrew, 'slategray', label='early_trials')
                    ax.fill_between(
                        range(0, int(range_val / binsize) * 2),
                        meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        alpha=0.5, color='slategray'
                    )
                    
                    meanrew = perirew[cll][1][0]
                    rewall = perirew[cll][1][1]
                    ax.plot(meanrew, 'darkcyan', label='late_trials')
                    ax.fill_between(
                        range(0, int(range_val / binsize) * 2),
                        meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                        alpha=0.5, color='darkcyan'
                    )
                    
                    ax.set_title(f'Cell {cll + 1}')
                    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
                    ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
                    ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
                    ax.spines[['top', 'right']].set_visible(False)
                    if cll == clls - 1 and plane == 0:
                        ax.legend(bbox_to_anchor=(1.01, 1.05))
                
                if clls > 1:
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'Peri-reward \n {mouse_name}, Day={dy}, Plane {plane}')
                plt.show()
                pdf.savefig(fig)
                plt.close(fig)

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