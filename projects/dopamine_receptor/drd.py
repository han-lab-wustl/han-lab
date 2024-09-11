"""
functions for drd cell analysis
"""
import os, sys
import numpy as np
import statsmodels.api as sm
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from projects.pyr_reward.rewardcell import perireward_binned_activity

# Function to load and filter dFF_iscell data
def load_and_filter_fall_data(fall_file):
    f = loadmat(fall_file)
    
    # Extract necessary variables
    dFF_iscell = f['dFF_iscell']
    stat = f['stat'][0]
    iscell = f['iscell'][:, 0].astype(bool)
    
    # Determine cells to keep, excluding merged ROIs
    statiscell = [stat[i] for i in range(len(stat)) if iscell[i]]
    garbage = []
    for st in statiscell:
        if 'imerge' in st.dtype.names and len(st['imerge'][0]) > 0:
            garb =  st['imerge'][0].flatten().tolist()
            garbage.extend(garb)
    
    arr = [x[0] for x in garbage if len(x) > 0]
    if len(arr) > 0:
        garbage = np.unique(np.concatenate(arr))
    else:
        garbage = []
    
    cllsind = np.arange(f['F'].shape[0])
    cllsindiscell = cllsind[iscell]
    keepind = ~np.isin(cllsindiscell, garbage)

    # Filter dFF_iscell
    dFF_iscell_filtered = dFF_iscell[keepind, :]
    return dFF_iscell_filtered, f

# Function to perform GLM on dFF_iscell data
def run_glm(dFF_iscell_filtered, f):
    dff_res = []
    range_val, binsize = 12, 0.2  # s
    perirew = []
    
    for cll in range(dFF_iscell_filtered.shape[0]):
        X = np.array([f['forwardvel'][0]]).T  # Predictor(s)
        X = sm.add_constant(X)  # Adds a constant term to the predictor(s)
        y = dFF_iscell_filtered[cll, :]  # Outcome

        # Fit a regression model
        model = sm.GLM(y, X, family=sm.families.Gaussian())
        result = model.fit()
        dff_res.append(result.resid_pearson)

        # Peri-reward activity
        dff = dFF_iscell_filtered[cll, :]
        normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
            dff, (f['rewards'][0] == 0.5).astype(int),
            f['timedFF'][0], f['trialnum'][0], range_val, binsize
        )
        perirew.append([meanrewdFF, rewdFF])

    dff_res = np.array(dff_res)
    return dff_res, perirew

# Function to plot GLM results
def plot_glm_results(dff_res, dy, planelut, root):
    clls = dff_res.shape[0]
    subpl = int(np.ceil(np.sqrt(clls)))
    
    plt.figure(figsize=(20, 10))
    for cll in range(clls):
        plt.subplot(subpl, subpl, cll + 1)
        plt.plot(dff_res[cll, :])
        plt.title(f'Cell {cll + 1}')
    
    plt.suptitle(f'GLM residuals; Day={dy}, {planelut[int(root.split("plane")[1])]}')
    plt.show()

# Function to plot peri-reward activity
def plot_peri_reward(perirew, dy, planelut, root, range_val, binsize):
    clls = len(perirew)
    subpl = int(np.ceil(np.sqrt(clls)))
    fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))      
    rr, cc = 0, 0

    for cll in range(clls):
        ax = axes[rr, cc] if subpl > 1 else axes
        ax.plot(perirew[cll][0], 'slategray')
        ax.fill_between(range(0, int(range_val/binsize)*2), 
            perirew[cll][0] - scipy.stats.sem(perirew[cll][1], axis=1, nan_policy='omit'),
            perirew[cll][0] + scipy.stats.sem(perirew[cll][1], axis=1, nan_policy='omit'),
            alpha=0.5, color='slategray')
        ax.set_title(f'Cell {cll + 1}')
        ax.axvline(int(range_val/binsize), color='k', linestyle='--')
        ax.set_xticks(np.arange(0, (int(range_val/binsize) * 2) + 1, 20))
        ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
        ax.spines[['top', 'right']].set_visible(False)
        
        if rr >= subpl - 1:
            rr = 0
            cc += 1
        else:
            rr += 1

    fig.tight_layout()
    plt.suptitle(f'Peri-reward activity; Day={dy}, {planelut[int(root.split("plane")[1])]}')
    plt.show()
