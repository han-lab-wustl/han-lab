import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr
import os

# === Load your position-binned data ===
data_all = dict(np.load("position_binned_summary_all_days.npz", allow_pickle=True))

# === Define days ===
control_days = [30, 33,36]  # e.g., nonopto_opto_epoch trials on these days
opto_days = [28, 29, 32,34,35]     # e.g., opto trials on these days

# === Setup ===
trial_types = ['str', 'ftr', 'ttr']
signals = ['lick']  # Choose one or loop through both
titles = {
    'str': 'Success Trials (STR)',
    'ftr': 'Failure Trials (FTR)',
    'ttr': 'Total Trials (TTR)',
}
n_bins = 30
positions = np.linspace(0, 270, n_bins)
binsize = positions[1] - positions[0]

# === Save directory ===
save_dir = r"C:\Users\HanLab\Downloads\comparison"
os.makedirs(save_dir, exist_ok=True)

# === Helper function ===
def stack_trials(days, group, ttype, signal):
    stacked = []
    for d in days:
        key = f'day_{d}'
        try:
            arr = data_all[key].item()[group][ttype][signal]
            if arr.shape[1] > 0:
                stacked.append(arr)
        except KeyError:
            print(f"Skipping {key} due to missing data")
    return np.hstack(stacked) if stacked else np.empty((n_bins, 0))

# === Main loop ===
for signal in signals:
    for ttype in trial_types:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        # Stack data
        opto_data = stack_trials(opto_days, 'opto', ttype, signal)
        ctrl_data = stack_trials(control_days, 'nonopto_opto_epoch', ttype, signal)

        # Mean ± SEM
        mean_opto = np.nanmean(opto_data, axis=1)
        sem_opto = sem(opto_data, axis=1, nan_policy='omit')
        mean_ctrl = np.nanmean(ctrl_data, axis=1)
        sem_ctrl = sem(ctrl_data, axis=1, nan_policy='omit')

        # Signed diff & stats
        signed_diff = mean_opto - mean_ctrl
        area_signed = np.nansum(signed_diff) * binsize
        mean_signed = np.nanmean(signed_diff)
        valid = ~np.isnan(mean_opto) & ~np.isnan(mean_ctrl)
        r, _ = pearsonr(mean_opto[valid], mean_ctrl[valid]) if np.sum(valid) > 2 else (np.nan, np.nan)

        # Plot: traces
        axs[0].plot(positions, mean_opto, label=f'Opto (days {opto_days})', color='royalblue')
        axs[0].fill_between(positions, mean_opto - sem_opto, mean_opto + sem_opto, alpha=0.3, color='royalblue')
        axs[0].plot(positions, mean_ctrl, label=f'Control (days {control_days})', color='darkorange')
        axs[0].fill_between(positions, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, alpha=0.3, color='darkorange')
        axs[0].set_title(f'{titles[ttype]} — {signal.upper()}')
        axs[0].set_ylabel(signal.upper())
        axs[0].set_xlabel('Position (cm)')
        axs[0].legend()

        # Plot: signed diff
        axs[1].plot(positions, signed_diff, color='black')
        axs[1].axhline(0, linestyle='--', color='gray')
        axs[1].axhline(mean_signed, linestyle=':', color='red', label='Mean diff')
        axs[1].set_title(f'Diff: AUC={area_signed:.2f}, Mean={mean_signed:.2f}, r={r:.2f}')
        axs[1].legend()
        axs[1].set_xlabel('Position (cm)')

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle(f'Position-Binned {signal.upper()} Comparison — {ttype.upper()}\n'
                     f'Opto Days: {opto_days} | Control Days: {control_days}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        filename = f'{signal}_{ttype}_comparison_opto{"-".join(map(str, opto_days))}_ctrl{"-".join(map(str, control_days))}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)

        # Show and close
        plt.show()
        plt.close()