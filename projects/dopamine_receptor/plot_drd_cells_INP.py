"""
aug 2024
"""

import os, sys, scipy
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

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity
# Define source directory and mouse name
src = r'Y:\drd'
mouse_name = 'e256'
days = [2]
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
range_val, binsize = 8, 0.2 # s
#%%
for dy in days:
    day_dir = os.path.join(src, mouse_name, str(dy))
    for root, dirs, files in os.walk(day_dir):
        for file in files:
            if 'plane' in root and file.endswith('roibyclick_F.mat'):
                f = loadmat(os.path.join(root, file))
                
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
                    dff = dFF_iscell_filtered[cll,:]
                    # dff = result.resid_pearson
                    normmeanrewdFF, meanrewdFF, normrewdFF, \
                    rewdFF = perireward_binned_activity(dff, (f['solenoid2'][0]).astype(int), 
                            f['timedFF'][0], f['trialnum'][0],range_val, binsize)
                    perirew.append([meanrewdFF, rewdFF])
                dff_res = np.array(dff_res)
                # dff_res = dFF_iscell_filtered
                
                
                # Plotting
                clls = dff_res.shape[0]
                subpl = int(np.ceil(np.sqrt(clls)))
                plt.figure(figsize=(20, 10))
                for cll in range(clls):
                    plt.subplot(subpl, subpl, cll + 1)
                    plt.plot(dff_res[cll, :])
                    plt.title(f'Cell {cll + 1}')
                plt.suptitle(f'GLM residuals; Day={dy}, {planelut[int(root.split("plane")[1])]}')
                # peri reward
                subpl = int(np.ceil(np.sqrt(clls)))
                fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))      
                rr=0;cc=0        
                for cll in range(clls):
                    if subpl>1:
                        ax = axes[rr,cc]
                    else: ax = axes
                    ax.plot(perirew[cll][0], 'slategray')
                    ax.fill_between(range(0,int(range_val/binsize)*2), 
                        perirew[cll][0]-scipy.stats.sem(perirew[cll][1],axis=1,nan_policy='omit'),
                        perirew[cll][0]+scipy.stats.sem(perirew[cll][1],axis=1,nan_policy='omit'),
                    alpha=0.5, color='slategray')
                    ax.set_title(f'Cell {cll + 1}')
                    ax.axvline(int(range_val/binsize),color='k', linestyle='--')
                    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
                    ax.set_xticklabels(np.arange(-range_val, range_val+1, 4))
                    ax.spines[['top','right']].set_visible(False)
                    if rr>=subpl-1:
                        rr=0;cc+=1
                    else:
                        rr+=1
                fig.tight_layout()
                plt.suptitle(f'GLM residuals; Day={dy}, {planelut[int(root.split("plane")[1])]}')
                plt.show()
