import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, ranksums

# user settings
npz_path = "lick_rate_block_consumption_licks_new.npz"
save_dir = r"C:\Users\HanLab\Downloads\day_comparison"
control_days = [27, 30, 36, 38, 40, 42,44,46,48]
opto_days    = [32, 34, 35, 37, 39, 41,43,45,47,49]
#control_days = [30, 36, 38, 40, 42]
#control_days = [38]

#opto_days    = [32]


trial_types  = [ 'str','ftr', 'ttr']
#trial_types  = ["ftr"]
range_val    = 10

os.makedirs(save_dir, exist_ok=True)

# load and infer time axis
data = dict(np.load(npz_path, allow_pickle=True))
first_item = next(iter(data.values())).item()
n_time = first_item["opto"]["str"]["lick"].shape[0]
time = np.linspace(-range_val, range_val, n_time)
mask_2_4 = (time >= 2) & (time <= 4)

for trial_type in trial_types:
    # 1 within opto days  opto trials vs nonopto trials
    mats_A, mats_B = [], []
    for d in opto_days:
        key = f"day_{d}"
        A = data[key].item()["opto"][trial_type]["lick"]
        B = data[key].item()["nonopto_opto_epoch"][trial_type]["lick"]
        if A.shape[0] == n_time and A.size > 0: mats_A.append(A)
        if B.shape[0] == n_time and B.size > 0: mats_B.append(B)
    A_mat = np.hstack(mats_A) if mats_A else np.empty((n_time, 0))
    B_mat = np.hstack(mats_B) if mats_B else np.empty((n_time, 0))

    A_mean = np.nanmean(A_mat, axis=1) if A_mat.size else np.full(n_time, np.nan)
    A_sem  = sem(A_mat, axis=1, nan_policy="omit") if A_mat.size else np.full(n_time, np.nan)
    B_mean = np.nanmean(B_mat, axis=1) if B_mat.size else np.full(n_time, np.nan)
    B_sem  = sem(B_mat, axis=1, nan_policy="omit") if B_mat.size else np.full(n_time, np.nan)
    signed_diff = A_mean - B_mean

    A_day_vals, B_day_vals = [], []
    for d in opto_days:
        key = f"day_{d}"
        A = data[key].item()["opto"][trial_type]["lick"]
        B = data[key].item()["nonopto_opto_epoch"][trial_type]["lick"]
        A_day_vals.append(np.nanmean(A[mask_2_4, :]))
        B_day_vals.append(np.nanmean(B[mask_2_4, :]))
    stat, pval = ranksums(A_day_vals, B_day_vals)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], wspace=0.4, hspace=0.3)
    fig.suptitle(f"Opto days {trial_type.upper()}  Days used {opto_days}\nOpto trials vs Nonopto trials", fontsize=12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, A_mean, label="Opto trials", color="royalblue")
    ax1.fill_between(time, A_mean - A_sem, A_mean + A_sem, alpha=0.3, color="royalblue")
    ax1.plot(time, B_mean, label="Nonopto trials", color="darkorange")
    ax1.fill_between(time, B_mean - B_sem, B_mean + B_sem, alpha=0.3, color="darkorange")
    ax1.axvline(0, ls="--", c="gray"); ax1.axvspan(0, 2, color="lightgreen", alpha=0.3)
    ax1.set_ylabel("Lick rate Hz"); ax1.legend(); ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, signed_diff, color="black")
    ax2.axhline(0, ls="--", c="gray"); ax2.axvline(0, ls="--", c="gray"); ax2.axvspan(2, 4, color="gray", alpha=0.15)
    ax2.set_ylabel("Delta A minus B"); ax2.set_xlabel("Time s"); ax2.spines[['top','right']].set_visible(False)

    ax3 = fig.add_subplot(gs[:, 1])
    ax3.boxplot([A_day_vals, B_day_vals], positions=[1, 2], widths=0.6,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"))
    ax3.scatter(np.full(len(A_day_vals), 1.0), A_day_vals, color="royalblue", zorder=3)
    ax3.scatter(np.full(len(B_day_vals), 2.0), B_day_vals, color="darkorange", zorder=3)
    ax3.set_xticks([1, 2]); ax3.set_xticklabels(["Opto trials", "Nonopto trials"])
    ax3.set_ylabel("Mean lick rate 2 to 4 s")
    ax3.set_title(f"Rank sum p = {pval:.4g}")
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{trial_type}_optoDays_opto_vs_nonopto.png"), dpi=300)
    plt.show()

    # 2 within control days  opto trials vs nonopto trials
    mats_A, mats_B = [], []
    for d in control_days:
        key = f"day_{d}"
        A = data[key].item()["opto"][trial_type]["lick"]
        B = data[key].item()["nonopto_opto_epoch"][trial_type]["lick"]
        if A.shape[0] == n_time and A.size > 0: mats_A.append(A)
        if B.shape[0] == n_time and B.size > 0: mats_B.append(B)
    A_mat = np.hstack(mats_A) if mats_A else np.empty((n_time, 0))
    B_mat = np.hstack(mats_B) if mats_B else np.empty((n_time, 0))

    A_mean = np.nanmean(A_mat, axis=1) if A_mat.size else np.full(n_time, np.nan)
    A_sem  = sem(A_mat, axis=1, nan_policy="omit") if A_mat.size else np.full(n_time, np.nan)
    B_mean = np.nanmean(B_mat, axis=1) if B_mat.size else np.full(n_time, np.nan)
    B_sem  = sem(B_mat, axis=1, nan_policy="omit") if B_mat.size else np.full(n_time, np.nan)
    signed_diff = A_mean - B_mean

    A_day_vals, B_day_vals = [], []
    for d in control_days:
        key = f"day_{d}"
        A = data[key].item()["opto"][trial_type]["lick"]
        B = data[key].item()["nonopto_opto_epoch"][trial_type]["lick"]
        A_day_vals.append(np.nanmean(A[mask_2_4, :]))
        B_day_vals.append(np.nanmean(B[mask_2_4, :]))
    stat, pval = ranksums(A_day_vals, B_day_vals)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], wspace=0.4, hspace=0.3)
    fig.suptitle(f"Control days {trial_type.upper()}  Days used {control_days}\nOpto trials vs Nonopto trials", fontsize=12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, A_mean, label="Opto trials", color="royalblue")
    ax1.fill_between(time, A_mean - A_sem, A_mean + A_sem, alpha=0.3, color="royalblue")
    ax1.plot(time, B_mean, label="Nonopto trials", color="darkorange")
    ax1.fill_between(time, B_mean - B_sem, B_mean + B_sem, alpha=0.3, color="darkorange")
    ax1.axvline(0, ls="--", c="gray"); ax1.axvspan(0, 2, color="lightgreen", alpha=0.3)
    ax1.set_ylabel("Lick rate Hz"); ax1.legend(); ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, signed_diff, color="black")
    ax2.axhline(0, ls="--", c="gray"); ax2.axvline(0, ls="--", c="gray"); ax2.axvspan(2, 4, color="gray", alpha=0.15)
    ax2.set_ylabel("Delta A minus B"); ax2.set_xlabel("Time s"); ax2.spines[['top','right']].set_visible(False)

    ax3 = fig.add_subplot(gs[:, 1])
    ax3.boxplot([A_day_vals, B_day_vals], positions=[1, 2], widths=0.6,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"))
    ax3.scatter(np.full(len(A_day_vals), 1.0), A_day_vals, color="royalblue", zorder=3)
    ax3.scatter(np.full(len(B_day_vals), 2.0), B_day_vals, color="darkorange", zorder=3)
    ax3.set_xticks([1, 2]); ax3.set_xticklabels(["Opto trials", "Nonopto trials"])
    ax3.set_ylabel("Mean lick rate 2 to 4 s")
    ax3.set_title(f"Rank sum p = {pval:.4g}")
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{trial_type}_controlDays_opto_vs_nonopto.png"), dpi=300)
    plt.show()

    # 3 across day types for opto trials
    mats_A, mats_B = [], []
    for d in opto_days:
        A = data[f"day_{d}"].item()["opto"][trial_type]["lick"]
        if A.shape[0] == n_time and A.size > 0: mats_A.append(A)
    for d in control_days:
        B = data[f"day_{d}"].item()["opto"][trial_type]["lick"]
        if B.shape[0] == n_time and B.size > 0: mats_B.append(B)
    A_mat = np.hstack(mats_A) if mats_A else np.empty((n_time, 0))
    B_mat = np.hstack(mats_B) if mats_B else np.empty((n_time, 0))

    A_mean = np.nanmean(A_mat, axis=1) if A_mat.size else np.full(n_time, np.nan)
    A_sem  = sem(A_mat, axis=1, nan_policy="omit") if A_mat.size else np.full(n_time, np.nan)
    B_mean = np.nanmean(B_mat, axis=1) if B_mat.size else np.full(n_time, np.nan)
    B_sem  = sem(B_mat, axis=1, nan_policy="omit") if B_mat.size else np.full(n_time, np.nan)
    signed_diff = A_mean - B_mean

    A_day_vals = []
    for d in opto_days:
        A = data[f"day_{d}"].item()["opto"][trial_type]["lick"]
        A_day_vals.append(np.nanmean(A[mask_2_4, :]))
    B_day_vals = []
    for d in control_days:
        B = data[f"day_{d}"].item()["opto"][trial_type]["lick"]
        B_day_vals.append(np.nanmean(B[mask_2_4, :]))
    stat, pval = ranksums(A_day_vals, B_day_vals)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], wspace=0.4, hspace=0.3)
    fig.suptitle(f"{trial_type.upper()}  Opto trials across day types\nOpto days {opto_days}  Control days {control_days}", fontsize=12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, A_mean, label="Opto days", color="royalblue")
    ax1.fill_between(time, A_mean - A_sem, A_mean + A_sem, alpha=0.3, color="royalblue")
    ax1.plot(time, B_mean, label="Control days", color="darkorange")
    ax1.fill_between(time, B_mean - B_sem, B_mean + B_sem, alpha=0.3, color="darkorange")
    ax1.axvline(0, ls="--", c="gray"); ax1.axvspan(0, 2, color="lightgreen", alpha=0.3)
    ax1.set_ylabel("Lick rate Hz"); ax1.legend(); ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, signed_diff, color="black")
    ax2.axhline(0, ls="--", c="gray"); ax2.axvline(0, ls="--", c="gray"); ax2.axvspan(2, 4, color="gray", alpha=0.15)
    ax2.set_ylabel("Delta A minus B"); ax2.set_xlabel("Time s"); ax2.spines[['top','right']].set_visible(False)

    ax3 = fig.add_subplot(gs[:, 1])
    ax3.boxplot([A_day_vals, B_day_vals], positions=[1, 2], widths=0.6,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"))
    ax3.scatter(np.full(len(A_day_vals), 1.0), A_day_vals, color="royalblue", zorder=3)
    ax3.scatter(np.full(len(B_day_vals), 2.0), B_day_vals, color="darkorange", zorder=3)
    ax3.set_xticks([1, 2]); ax3.set_xticklabels(["Opto days", "Control days"])
    ax3.set_ylabel("Mean lick rate 2 to 4 s")
    ax3.set_title(f"Rank sum p = {pval:.4g}")
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{trial_type}_acrossDays_optoTrials.png"), dpi=300)
    plt.show()

    # 4 across day types for nonopto trials
    mats_A, mats_B = [], []
    for d in opto_days:
        A = data[f"day_{d}"].item()["nonopto_opto_epoch"][trial_type]["lick"]
        if A.shape[0] == n_time and A.size > 0: mats_A.append(A)
    for d in control_days:
        B = data[f"day_{d}"].item()["nonopto_opto_epoch"][trial_type]["lick"]
        if B.shape[0] == n_time and B.size > 0: mats_B.append(B)
    A_mat = np.hstack(mats_A) if mats_A else np.empty((n_time, 0))
    B_mat = np.hstack(mats_B) if mats_B else np.empty((n_time, 0))

    A_mean = np.nanmean(A_mat, axis=1) if A_mat.size else np.full(n_time, np.nan)
    A_sem  = sem(A_mat, axis=1, nan_policy="omit") if A_mat.size else np.full(n_time, np.nan)
    B_mean = np.nanmean(B_mat, axis=1) if B_mat.size else np.full(n_time, np.nan)
    B_sem  = sem(B_mat, axis=1, nan_policy="omit") if B_mat.size else np.full(n_time, np.nan)
    signed_diff = A_mean - B_mean

    A_day_vals = []
    for d in opto_days:
        A = data[f"day_{d}"].item()["nonopto_opto_epoch"][trial_type]["lick"]
        A_day_vals.append(np.nanmean(A[mask_2_4, :]))
    B_day_vals = []
    for d in control_days:
        B = data[f"day_{d}"].item()["nonopto_opto_epoch"][trial_type]["lick"]
        B_day_vals.append(np.nanmean(B[mask_2_4, :]))
    stat, pval = ranksums(A_day_vals, B_day_vals)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], wspace=0.4, hspace=0.3)
    fig.suptitle(f"{trial_type.upper()}  Nonopto trials across day types\nOpto days {opto_days}  Control days {control_days}", fontsize=12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, A_mean, label="Opto days", color="royalblue")
    ax1.fill_between(time, A_mean - A_sem, A_mean + A_sem, alpha=0.3, color="royalblue")
    ax1.plot(time, B_mean, label="Control days", color="darkorange")
    ax1.fill_between(time, B_mean - B_sem, B_mean + B_sem, alpha=0.3, color="darkorange")
    ax1.axvline(0, ls="--", c="gray"); ax1.axvspan(0, 2, color="lightgreen", alpha=0.3)
    ax1.set_ylabel("Lick rate Hz"); ax1.legend(); ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, signed_diff, color="black")
    ax2.axhline(0, ls="--", c="gray"); ax2.axvline(0, ls="--", c="gray"); ax2.axvspan(2, 4, color="gray", alpha=0.15)
    ax2.set_ylabel("Delta A minus B"); ax2.set_xlabel("Time s"); ax2.spines[['top','right']].set_visible(False)

    ax3 = fig.add_subplot(gs[:, 1])
    ax3.boxplot([A_day_vals, B_day_vals], positions=[1, 2], widths=0.6,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"))
    ax3.scatter(np.full(len(A_day_vals), 1.0), A_day_vals, color="royalblue", zorder=3)
    ax3.scatter(np.full(len(B_day_vals), 2.0), B_day_vals, color="darkorange", zorder=3)
    ax3.set_xticks([1, 2]); ax3.set_xticklabels(["Opto days", "Control days"])
    ax3.set_ylabel("Mean lick rate 2 to 4 s")
    ax3.set_title(f"Rank sum p = {pval:.4g}")
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{trial_type}_acrossDays_nonoptoTrials.png"), dpi=300)
    plt.show()

