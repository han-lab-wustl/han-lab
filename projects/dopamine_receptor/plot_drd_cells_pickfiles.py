"""
zahra
Updated: Sept 2024
"""

import os, sys, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')  # Custom path to your repository
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Suppress the Tkinter root window
Tk().withdraw()

# Formatting for figures
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)  # Controls default text sizes

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity


# Function to select day folders using file explorer
def select_day_folders():
    day_folders = []
    n_days = int(input("Enter the number of day folders you want to select: "))
    for i in range(n_days):
        print(f"Select the folder for Day {i + 1}:")
        day_folder = askdirectory()  # Opens file explorer to select folder
        if day_folder:
            day_folders.append(day_folder)
        else:
            print("No folder selected, skipping.")
    return day_folders

# Get user input for day folders
day_folders = select_day_folders()

# Lookup table for planes
planelut = {0: 'SR', 1: 'SP', 2: 'SO'}

# Loop through each selected day folder
for day_dir in day_folders:
    for root, dirs, files in os.walk(day_dir):
        for file in files:
            if 'plane' in root and file.endswith('Fall.mat'):
                f = loadmat(os.path.join(root, file))
                
                # Extract necessary variables
                dFF_iscell = f['dFF_iscell']
                stat = f['stat'][0]
                iscell = f['iscell'][:, 0].astype(bool)
                
                # Determine the cells to keep, excluding merged ROIs
                statiscell = [stat[i] for i in range(len(stat)) if iscell[i]]
                garbage = []
                for st in statiscell:
                    if 'imerge' in st.dtype.names and len(st['imerge'][0]) > 0:
                        garb =  st['imerge'][0].flatten().tolist()
                        garbage.extend(garb)
                arr = [x[0] for x in garbage if len(x)>0]
                if len(arr)>0:
                    garbage = np.unique(np.concatenate(arr))
                else:
                    garbage = []
                cllsind = np.arange(f['F'].shape[0])
                cllsindiscell = cllsind[iscell]
                keepind = ~np.isin(cllsindiscell, garbage)

                # Filter dFF_iscell
                dFF_iscell_filtered = dFF_iscell[keepind, :]

                # Run GLM
                dff_res = []
                perirew = []
                range_val, binsize = 12, 0.2  # s
                for cll in range(dFF_iscell_filtered.shape[0]):
                    X = np.array([f['forwardvel'][0]]).T  # Predictor(s)
                    X = sm.add_constant(X)  # Adds a constant term to the predictor(s)
                    y = dFF_iscell_filtered[cll, :]  # Outcome
                    # Fit a regression model
                    model = sm.GLM(y, X, family=sm.families.Gaussian())
                    result = model.fit()
                    dff_res.append(result.resid_pearson)    
                    # Peri-reward
                    # no glm
                    # dff = dFF_iscell_filtered[cll, :]
                    # glm 
                    dff = result.resid_pearson
                    normmeanrewdFF, meanrewdFF, normrewdFF, \
                    rewdFF = perireward_binned_activity(dff, (f['rewards'][0] == 0.5).astype(int), 
                                                        f['timedFF'][0], f['trialnum'][0], range_val, binsize)
                    perirew.append([meanrewdFF, rewdFF])
                dff_res = np.array(dff_res)
                dff_res = dFF_iscell_filtered

                # Plotting GLM residuals
                clls = dff_res.shape[0]
                subpl = int(np.ceil(np.sqrt(clls)))
                plt.figure(figsize=(20, 10))
                for cll in range(clls):
                    plt.subplot(subpl, subpl, cll + 1)
                    plt.plot(dff_res[cll, :])
                    plt.title(f'Cell {cll + 1}')
                plt.suptitle(f'GLM residuals; {day_dir.split(os.sep)[-1]}')
                fig.tight_layout()
                # Peri-reward plots
                subpl = int(np.ceil(np.sqrt(clls)))
                fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))      
                rr = 0
                cc = 0        
                for cll in range(clls):
                    if subpl > 1:
                        ax = axes[rr, cc]
                    else:
                        ax = axes
                    ax.plot(perirew[cll][0], 'slategray')
                    ax.fill_between(range(0, int(range_val / binsize) * 2), 
                                    perirew[cll][0] - scipy.stats.sem(perirew[cll][1], axis=1, nan_policy='omit'),
                                    perirew[cll][0] + scipy.stats.sem(perirew[cll][1], axis=1, nan_policy='omit'),
                                    alpha=0.5, color='slategray')
                    ax.set_title(f'Cell {cll + 1}')
                    ax.axvline(int(range_val / binsize), color='k', linestyle='--')
                    ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
                    ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
                    ax.spines[['top', 'right']].set_visible(False)
                    if rr >= subpl - 1:
                        rr = 0
                        cc += 1
                    else:
                        rr += 1
                fig.tight_layout()
                fig.suptitle(f'Peri-reward activity \n {day_dir.split(os.sep)[-1]}')
                plt.show()
