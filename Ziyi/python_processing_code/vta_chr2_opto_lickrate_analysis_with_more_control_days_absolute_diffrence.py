import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr

# Load data
data = np.load("lickrate_data.npz", allow_pickle=True)

# Define day groups
opto_days = [19,20,22, 23]
control_days = [13,15,17,18,21, 24]

# Function to load and stack trial data by day list and type
def get_group_data(day_list, dtype):
    return np.vstack([data[f"day{d}_{dtype}"] for d in day_list])

# Time axis setup
range_val = 10
binsize = 0.1
n_timepoints = data["day19_opto"].shape[1]
time = np.linspace(-range_val, range_val, n_timepoints)

# Prepare plot: 3 comparisons × 2 columns (curves, signed diff)
fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

# Comparison 1: Opto trials on opto_days vs control_days
for i, dtype in enumerate(["opto", "nonopto"]):
    groupA = get_group_data(opto_days, dtype)
    groupB = get_group_data(control_days, dtype)

    meanA = np.nanmean(groupA, axis=0)
    meanB = np.nanmean(groupB, axis=0)
    semA = sem(groupA, axis=0, nan_policy='omit')
    semB = sem(groupB, axis=0, nan_policy='omit')

    signed_diff = meanA - meanB
    area_signed = np.nansum(signed_diff) * binsize
    mean_signed = np.nanmean(signed_diff)

    valid_mask = ~np.isnan(meanA) & ~np.isnan(meanB)
    if np.sum(valid_mask) > 2:
        r, p_corr = pearsonr(meanA[valid_mask], meanB[valid_mask])
    else:
        r, p_corr = np.nan, np.nan

    # Panel 1: Mean lick rate curves
    axs[i, 0].plot(time, meanA, label=f"{dtype.capitalize()} days {opto_days}", color="#1f77b4")
    axs[i, 0].fill_between(time, meanA - semA, meanA + semA, color="#1f77b4", alpha=0.3)
    axs[i, 0].plot(time, meanB, label=f"Control days {control_days}", color="#ff7f0e")
    axs[i, 0].fill_between(time, meanB - semB, meanB + semB, color="#ff7f0e", alpha=0.3)
    axs[i, 0].axvline(0, linestyle="--", color="black")
    axs[i, 0].set_ylabel("Lick Rate (Hz)")
    axs[i, 0].set_title(f"{dtype.capitalize()} Trials: Group Comparison")
    axs[i, 0].legend()

    # Panel 2: Signed difference
    axs[i, 1].plot(time, signed_diff, color="black")
    axs[i, 1].axvline(0, linestyle="--", color="gray")
    axs[i, 1].axhline(0, linestyle="--", color="gray")
    axs[i, 1].axhline(mean_signed, linestyle=":", color="red", label="Mean diff")
    axs[i, 1].set_title(f"Signed Diff: Area={area_signed:.3f}, Mean={mean_signed:.3f}, r={r:.2f}")
    axs[i, 1].legend()

# Comparison 3: Opto vs non-opto trials within opto_days
group_opto = get_group_data(opto_days, "opto")
group_nonopto = get_group_data(opto_days, "nonopto")

mean_opto = np.nanmean(group_opto, axis=0)
mean_nonopto = np.nanmean(group_nonopto, axis=0)
sem_opto = sem(group_opto, axis=0, nan_policy='omit')
sem_nonopto = sem(group_nonopto, axis=0, nan_policy='omit')

signed_diff = mean_opto - mean_nonopto
area_signed = np.nansum(signed_diff) * binsize
mean_signed = np.nanmean(signed_diff)

valid_mask = ~np.isnan(mean_opto) & ~np.isnan(mean_nonopto)
if np.sum(valid_mask) > 2:
    r, p_corr = pearsonr(mean_opto[valid_mask], mean_nonopto[valid_mask])
else:
    r, p_corr = np.nan, np.nan

# Panel 3.1: Mean curves
axs[2, 0].plot(time, mean_opto, label="Opto trials", color="#1f77b4")
axs[2, 0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, color="#1f77b4", alpha=0.3)
axs[2, 0].plot(time, mean_nonopto, label="Non-opto trials", color="#ff7f0e")
axs[2, 0].fill_between(time, mean_nonopto - sem_nonopto, mean_nonopto + sem_nonopto, color="#ff7f0e", alpha=0.3)
axs[2, 0].axvline(0, linestyle="--", color="black")
axs[2, 0].set_ylabel("Lick Rate (Hz)")
axs[2, 0].set_title(f"Opto vs Non-opto Trials on Days {opto_days}")
axs[2, 0].legend()

# Panel 3.2: Signed difference
axs[2, 1].plot(time, signed_diff, color="black")
axs[2, 1].axvline(0, linestyle="--", color="gray")
axs[2, 1].axhline(0, linestyle="--", color="gray")
axs[2, 1].axhline(mean_signed, linestyle=":", color="red", label="Mean diff")
axs[2, 1].set_title(f"Signed Diff: Area={area_signed:.3f}, Mean={mean_signed:.3f}, r={r:.2f}")
axs[2, 1].legend()

plt.tight_layout()
plt.show()
