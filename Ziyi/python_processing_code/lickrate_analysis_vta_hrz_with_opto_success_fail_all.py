import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr

# Load the saved lickrate data
lick_all = dict(np.load("lickrate_opto_aligned_all_days.npz", allow_pickle=True))

# Define day groups
control_days = [27, 30, 33]
opto_days = [28, 29, 31, 32]

# Plot settings
trial_types = ['str', 'ftr', 'ttr']
titles = {
    'str': 'Success Trials (STR)',
    'ftr': 'Failure Trials (FTR)',
    'ttr': 'Total Trials (TTR)',
}
range_val = 10
binsize = 0.2
n_timepoints = 100
time = np.linspace(-range_val, range_val, n_timepoints)

# Loop through each trial type
for ttype in trial_types:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    def stack_data(days, cond):
        all_trials = []
        for d in days:
            key = f"day_{d}"
            arr = lick_all[key].item()[cond][ttype]
            all_trials.append(arr[:n_timepoints, :])  # ensure time alignment
        return np.hstack(all_trials)

    # Group data
    control_licks = stack_data(control_days, 'nonopto_opto_epoch')
    opto_licks = stack_data(opto_days, 'nonopto_opto_epoch')

    # Mean ± SEM
    mean_control = np.nanmean(control_licks, axis=1)
    sem_control = sem(control_licks, axis=1, nan_policy='omit')
    mean_opto = np.nanmean(opto_licks, axis=1)
    sem_opto = sem(opto_licks, axis=1, nan_policy='omit')

    # Signed diff
    signed_diff = mean_opto - mean_control
    area_signed = np.nansum(signed_diff) * binsize
    mean_signed = np.nanmean(signed_diff)

    valid_mask = ~np.isnan(mean_opto) & ~np.isnan(mean_control)
    r, _ = pearsonr(mean_opto[valid_mask], mean_control[valid_mask]) if np.sum(valid_mask) > 2 else (np.nan, np.nan)

    # Plot mean traces
    axs[0].axvspan(0, 2, color='lightgreen', alpha=0.3, label='Opto Light')
    axs[0].plot(time, mean_opto, label=f'Opto Days ({opto_days})', color='royalblue')
    axs[0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, color='royalblue', alpha=0.3)
    axs[0].plot(time, mean_control, label=f'Control Days ({control_days})', color='darkorange')
    axs[0].fill_between(time, mean_control - sem_control, mean_control + sem_control, color='darkorange', alpha=0.3)
    axs[0].axvline(0, linestyle="--", color="gray")
    axs[0].set_title(f"{titles[ttype]}")
    axs[0].set_ylabel("Lick Rate (Hz)")
    axs[0].legend()

    # Plot signed difference
    axs[1].axvspan(0, 2, color='lightgreen', alpha=0.3, label='Opto Light')
    axs[1].plot(time, signed_diff, color="black")
    axs[1].axvline(0, linestyle="--", color="gray")
    axs[1].axhline(0, linestyle="--", color="gray")
    axs[1].axhline(mean_signed, linestyle=":", color="red", label="Mean diff")
    axs[1].set_title(f"Diff: AUC={area_signed:.2f}, Mean={mean_signed:.2f}, r={r:.2f}")
    axs[1].legend()

    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"Opto-Aligned Lick Rate Comparison — {ttype.upper()}")
    plt.tight_layout()
    plt.show()
