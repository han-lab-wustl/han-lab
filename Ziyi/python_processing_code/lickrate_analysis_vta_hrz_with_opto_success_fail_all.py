import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr

# Load the saved lickrate data
lick_all = dict(np.load("lick_rate_block_consumption_licks.npz", allow_pickle=True))

#['day_13', 'day_15', 'day_17', 'day_19', 'day_20', 'day_21', 
# 'day_23', 'day_24', 'day_26', 'day_27', 'day_28', 'day_29', 
# 'day_30', 'day_31', 'day_32', 'day_33']
# Define day groups
control_days = [27,30,36,38,40,42]  # e.g., nonopto_opto_epoch trials on these days
opto_days = [32,34,35,37,39,41]     # e.g., opto trials on these days

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
    control_licks = stack_data(control_days, 'opto')
    opto_licks = stack_data(opto_days, 'opto')

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr



# Setup
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

# Helper to stack trials from saved data
def stack_trials(days, cond, ttype):
    trials = []
    for d in days:
        key = f'day_{d}'
        arr = lick_all[key].item()[cond][ttype]
        if arr.shape[1] > 0:
            trials.append(arr[:n_timepoints, :])
    return np.hstack(trials) if trials else np.empty((n_timepoints, 0))

# Loop over trial types
for ttype in trial_types:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for i, (group_name, group_days) in enumerate([('Opto Days', opto_days), ('Control Days', control_days)]):
        lick_opto = stack_trials(group_days, 'opto', ttype)
        lick_nonopto = stack_trials(group_days, 'nonopto_opto_epoch', ttype)

        if lick_opto.shape[1] == 0 or lick_nonopto.shape[1] == 0:
            print(f"Skipping {group_name} {ttype} due to missing data")
            continue

        # Mean ± SEM
        mean_opto = np.nanmean(lick_opto, axis=1)
        sem_opto = sem(lick_opto, axis=1, nan_policy='omit')
        mean_nonopto = np.nanmean(lick_nonopto, axis=1)
        sem_nonopto = sem(lick_nonopto, axis=1, nan_policy='omit')

        # Signed difference
        signed_diff = mean_opto - mean_nonopto
        area_signed = np.nansum(signed_diff) * binsize
        mean_signed = np.nanmean(signed_diff)
        valid = ~np.isnan(mean_opto) & ~np.isnan(mean_nonopto)
        r, _ = pearsonr(mean_opto[valid], mean_nonopto[valid]) if np.sum(valid) > 2 else (np.nan, np.nan)

        # Plot: mean traces
        axs[i, 0].axvspan(0, 2, color='lightgreen', alpha=0.3)
        axs[i, 0].plot(time, mean_opto, label='Opto', color='royalblue')
        axs[i, 0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, alpha=0.3, color='royalblue')
        axs[i, 0].plot(time, mean_nonopto, label='Non-Opto', color='darkorange')
        axs[i, 0].fill_between(time, mean_nonopto - sem_nonopto, mean_nonopto + sem_nonopto, alpha=0.3, color='darkorange')
        axs[i, 0].axvline(0, linestyle='--', color='gray')
        axs[i, 0].set_title(f'{group_name} {group_days} — {titles[ttype]}')
        axs[i, 0].legend()
        axs[i, 0].set_ylabel('Lick Rate (Hz)')

        # Plot: signed diff
        axs[i, 1].axvspan(0, 2, color='lightgreen', alpha=0.3)
        axs[i, 1].plot(time, signed_diff, color='black')
        axs[i, 1].axvline(0, linestyle='--', color='gray')
        axs[i, 1].axhline(0, linestyle='--', color='gray')
        axs[i, 1].axhline(mean_signed, linestyle=':', color='red', label='Mean diff')
        axs[i, 1].set_title(f'Diff: AUC={area_signed:.2f}, Mean={mean_signed:.2f}, r={r:.2f}')
        axs[i, 1].legend()

        for j in [0, 1]:
            axs[i, j].spines[['top', 'right']].set_visible(False)
            axs[i, j].set_xlabel('Time (s)')

    fig.suptitle(f'Opto vs Non-Opto Trial Comparison — {ttype.upper()}')
    plt.tight_layout()
    plt.show()
