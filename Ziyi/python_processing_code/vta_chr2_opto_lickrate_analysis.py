import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr

# Load data
data = np.load("lickrate_data.npz", allow_pickle=True)

# Define comparison sets: (title, opto_data, control_data)
comparisons = [
    ("Opto Trials: Day 19 vs Day 21", data["day19_opto"], data["day21_opto"]),
    ("Non-opto Trials: Day 19 vs Day 21", data["day19_nonopto"], data["day21_nonopto"]),
    ("Opto Trials: Day 20 vs Day 24", data["day20_opto"], data["day24_opto"]),
    ("Non-opto Trials: Day 20 vs Day 24", data["day20_nonopto"], data["day24_nonopto"]),
    ("Combined Opto Trials: Day 19+20 vs Day 21+24",
     np.vstack([data["day19_opto"], data["day20_opto"]]),
     np.vstack([data["day21_opto"], data["day24_opto"]]))
]

# Time axis parameters
range_val = 2
binsize = 0.1
n_timepoints = data["day19_opto"].shape[1]
time = np.linspace(-range_val, range_val, n_timepoints)

# Create figure: 5 comparisons × 2 panels
fig, axs = plt.subplots(len(comparisons), 2, figsize=(12, len(comparisons) * 3), sharex=True)

for i, (title, opto_data, control_data) in enumerate(comparisons):
    mean_opto = np.nanmean(opto_data, axis=0)
    mean_ctrl = np.nanmean(control_data, axis=0)
    sem_opto = sem(opto_data, axis=0, nan_policy='omit')
    sem_ctrl = sem(control_data, axis=0, nan_policy='omit')

    # Signed difference
    signed_diff = mean_opto - mean_ctrl
    area_signed = np.nansum(signed_diff) * binsize
    mean_signed = np.nanmean(signed_diff)

    # Optional: correlation
    valid_mask = ~np.isnan(mean_opto) & ~np.isnan(mean_ctrl)
    if np.sum(valid_mask) > 2:
        r, p_corr = pearsonr(mean_opto[valid_mask], mean_ctrl[valid_mask])
    else:
        r, p_corr = np.nan, np.nan

    # Panel 1: Lick rate curves
    axs[i, 0].plot(time, mean_opto, label="Opto", color="#1f77b4")
    axs[i, 0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, color="#1f77b4", alpha=0.3)
    axs[i, 0].plot(time, mean_ctrl, label="Control", color="#ff7f0e")
    axs[i, 0].fill_between(time, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color="#ff7f0e", alpha=0.3)
    axs[i, 0].axvline(0, linestyle="--", color="black")
    axs[i, 0].set_ylabel("Lick Rate (Hz)")
    axs[i, 0].set_title(title)
    axs[i, 0].legend()

    # Panel 2: Signed difference
    axs[i, 1].plot(time, signed_diff, color="black")
