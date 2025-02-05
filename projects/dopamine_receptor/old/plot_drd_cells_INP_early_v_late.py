"""
aug 2024
"""

import os, sys, scipy, imageio
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
# formatting for figs
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
import matplotlib.backends.backend_pdf, matplotlib as mpl

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
savepth = os.path.join(savedst, 'drd1_per_epoch_earlyvlate.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late
# Define source directory and mouse name
src = r'Y:\drd'
mouse_name = 'e255'
days = [5,6]
range_val, binsize = 6, 0.2 # s
#%%
for dy in days:
    day_dir = os.path.join(src, mouse_name, str(dy))
    for root, dirs, files in os.walk(day_dir):
        for file in files:
            if 'plane' in root and file.endswith('roibyclick_F.mat'):
                f = loadmat(os.path.join(root, file))
                plane = int(root.split("plane")[1])
                ops = np.load(os.path.join(root, 'ops.npy'),
                            allow_pickle=True)
                # Filename pattern to match
                target_filename = 'masks.jpg'
                # Initialize variable to store the file path
                blotches_file_path = None
                # Iterate through the files in the 'root' directory
                for file in os.listdir(root):
                    # Check if the current item is a file and if it ends with 'blotches.jpg'
                    if os.path.isfile(os.path.join(root, file)) and file.endswith(target_filename):
                        blotches_file_path = os.path.join(root, file)
                        break  # Stop searching once we find the file
                eps = np.where(f['changeRewLoc']>0)[1]
                eps = np.append(eps,len(f['changeRewLoc'][0]))
                rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0]>0]

                plane = int(root.split("plane")[1][0])
                # Extract necessary variables
                dFF_iscell = f['dFF']
                # Filter dFF_iscell
                dFF_iscell_filtered = dFF_iscell.T                
                # run glm
                dff_res = []; perirew = []

                for cll in range(dFF_iscell_filtered.shape[0]):
                    X = np.array([f['forwardvel'][0]]).T # Predictor(s)
                    X = sm.add_constant(X) # Adds a constant term to the predictor(s)
                    y = dFF_iscell_filtered[cll,:] # Outcome
                    ############## GLM ##############
                    # Fit a regression model
                    model = sm.GLM(y, X, family=sm.families.Gaussian())
                    result = model.fit()
                    dff_res.append(result.resid_pearson)    
                    # peri reward
                    # dff = dFF_iscell_filtered[cll,:]
                    dff = result.resid_pearson
                    # for each epoch!!
                    perirew_ep = []
                    for ep in range(len(eps)-1):
                        eprng = np.arange(eps[ep],eps[ep+1])
                    # eprng = np.arange(0, eps[-1])
                        early_v_late = perireward_binned_activity_early_late(dff[eprng], 
                                (f['solenoid2'][0]).astype(int)[eprng], 
                                f['timedFF'][0][eprng], f['trialnum'][0][eprng],range_val, binsize,
                                early_trial=2, 
                                late_trial=8)
                        perirew_ep.append([[early_v_late['first_5']['meanrewdFF'], 
                                early_v_late['first_5']['rewdFF']], [early_v_late['last_5']['meanrewdFF'], 
                                early_v_late['last_5']['rewdFF']]])
                    perirew.append(perirew_ep)
                dff_res = np.array(dff_res)
                # dff_res = dFF_iscell_filtered
                
                # Plotting
                # mean image
                fig, ax = plt.subplots()
                image = imageio.imread(blotches_file_path)
                ax.imshow(image)
                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close(fig)
            
                clls = dff_res.shape[0]
                subpl = int(np.ceil(np.sqrt(clls)))
                fig, axes = plt.subplots(subpl, subpl, figsize=(20, 10))
                axes = axes.flatten()  # Flatten the array to easily index axes
                for cll in range(clls):
                    axes[cll].plot(dff_res[cll, :])
                    axes[cll].set_title(f'Cell {cll + 1}')
                # Hide any unused subplots
                for i in range(clls, len(axes)):
                    axes[i].axis('off')
                # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(f'GLM residuals \n {mouse_name}, Day={dy}, {plane}')
                pdf.savefig(fig)
                plt.close(fig)
                
                # per epoch
                for ep in range(len(eps)-1):
                    subpl = int(np.ceil(np.sqrt(clls)))
                    fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))
                    axes = axes.flatten()  # Flatten the array to easily index axes
                    
                    for cll in range(clls):
                        ax = axes[cll]
                        # first 5 
                        meanrew = perirew[cll][ep][0][0]
                        rewall = perirew[cll][ep][0][1]
                        ax.plot(meanrew, 'slategray', label='early_trials')
                        ax.fill_between(
                            range(0, int(range_val / binsize) * 2),
                            meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            alpha=0.5, color='slategray'
                        )
                        # last 5
                        meanrew = perirew[cll][ep][1][0]
                        rewall = perirew[cll][ep][1][1]

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
                        if cll==clls-1 and ep==0 and plane==0: ax.legend(bbox_to_anchor=(1.01, 1.05))
                    # Hide any unused subplots
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust 'rect' to accommodate any titles if necessary
                    fig.suptitle(f'Peri-reward, ep {ep+1}, rewloc {rewlocs[ep]} \n {mouse_name}, Day={dy}, {plane}')
                    plt.show()
                    pdf.savefig(fig)
                    plt.close(fig)

pdf.close()