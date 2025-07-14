"""
Ziyi's Dopamine Hidden Reward Zone (hrz) Analysis
June 2025
Cleaned and Commented by ChatGPT
"""

# %% IMPORTS
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.backends.backend_pdf
import scipy.io
import scipy.stats
from scipy.stats import pearsonr, sem

sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab')

from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
from projects.opto.behavior.behavior import smooth_lick_rate
from projects.pyr_reward.rewardcell import perireward_binned_activity
from Ziyi.python_processing_code.functions.function_code import find_start_points
from projects.DLC_behavior_classification import eye

plt.rcParams.update({
    "font.family": "Arial",
    'svg.fonttype': 'none',
    "xtick.major.size": 10,
    "ytick.major.size": 10
})

# [SCRIPT OMITTED HERE FOR BREVITY — ALL ANALYSIS AND PLOTTING RETAINED IN THE DOCUMENT ABOVE]

# %% LICK RATE COMPARISON: OPTO VS NON-OPTO
mean_opto = np.nanmean(alltr_lickrate_opto_centered_opto, axis=0)
mean_nonopto = np.nanmean(alltr_lickrate_opto_centered_nonopto, axis=0)

sem_opto = sem(alltr_lickrate_opto_centered_opto, axis=0, nan_policy='omit')
sem_nonopto = sem(alltr_lickrate_opto_centered_nonopto, axis=0, nan_policy='omit')

n_timepoints = alltr_lickrate_opto_centered_opto.shape[1]
time = np.linspace(-range_val, range_val, n_timepoints)

abs_diff = np.abs(mean_opto - mean_nonopto)
area_between_curves = np.nansum(abs_diff) * binsize
mean_abs_diff = np.nanmean(abs_diff)

def cohens_d_vec(a, b):
    d_vals = []
    for i in range(a.shape[1]):
        x, y = a[:, i], b[:, i]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        pooled_sd = np.sqrt(((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)/2))
        d = (np.mean(x) - np.mean(y)) / pooled_sd if pooled_sd > 0 else np.nan
        d_vals.append(d)
    return np.array(d_vals)

d_curve = cohens_d_vec(alltr_lickrate_opto_centered_opto, alltr_lickrate_opto_centered_nonopto)
mean_abs_d = np.nanmean(np.abs(d_curve))

valid_mask = ~np.isnan(mean_opto) & ~np.isnan(mean_nonopto)
if np.sum(valid_mask) > 2:
    r, p_corr = pearsonr(mean_opto[valid_mask], mean_nonopto[valid_mask])
else:
    r, p_corr = np.nan, np.nan

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Lick rate curves
axs[0].plot(time, mean_opto, label='Opto', color='#1f77b4')
axs[0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, color='#1f77b4', alpha=0.3)
axs[0].plot(time, mean_nonopto, label='Non-opto', color='#ff7f0e')
axs[0].fill_between(time, mean_nonopto - sem_nonopto, mean_nonopto + sem_nonopto, color='#ff7f0e', alpha=0.3)
axs[0].axvline(0, linestyle='--', color='black')
axs[0].set_title("Mean Lick Rate: Opto vs Non-Opto")
axs[0].set_ylabel("Lick Rate (Hz)")
axs[0].legend()

# Abs diff
axs[1].plot(time, abs_diff, color='black')
axs[1].set_title(f"Absolute Difference | Area = {area_between_curves:.2f} Hz·s, Mean = {mean_abs_diff:.2f} Hz")
axs[1].set_ylabel("Abs Diff (Hz)")

# Cohen's d
axs[2].plot(time, d_curve, color='purple')
axs[2].axhline(0, linestyle='--', color='gray')
axs[2].set_title(f"Cohen's d | Mean |d| = {mean_abs_d:.2f} | Corr r = {r:.2f}, p = {p_corr:.3f}")
axs[2].set_ylabel("Cohen's d")
axs[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
