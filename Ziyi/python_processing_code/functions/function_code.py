import numpy as np  # ✅ Moved to the top

# Putting some functions here
def find_start_points(data):
    # Check if the first element is 1
    starts = [0] if data[0] == 1 else []

    # Find the indices where data changes from 0 to 1
    changes = np.where(np.diff(data) == 1)[0]

    # Since np.diff reduces the length by 1, add 1 to each index to get the actual start points
    starts.extend(changes + 1)

    return starts

def spatial_binned_activity(dFF, ypos, trialnum, binsize=2, track_length=270):
    """
    Bins dF/F activity across the entire track by y-position.

    Args:
        dFF (np.ndarray): dF/F signal, shape (n_frames,)
        ypos (np.ndarray): y-position per frame in cm (n_frames,)
        trialnum (np.ndarray): trial number per frame (n_frames,)
        binsize (float): spatial bin size in cm
        track_length (float): length of track in cm

    Returns:
        normmeanFF: normalized average dF/F by location (n_bins,)
        meanFF: raw average dF/F by location (n_bins,)
        normFF_trials: trial-by-trial normalized dF/F (n_bins × n_trials)
        dFF_matrix: raw dF/F binned by location and trial (n_bins × n_trials)
    """
    n_bins = int(np.ceil(track_length / binsize))
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    unique_trials = np.unique(trialnum[trialnum > 0])
    dFF_matrix = np.ones((n_bins, len(unique_trials))) * np.nan

    for t_idx, t in enumerate(unique_trials):
        idx = np.where(trialnum == t)[0]
        for b in range(n_bins):
            bin_mask = (ypos[idx] >= bin_edges[b]) & (ypos[idx] < bin_edges[b + 1])
            bin_indices = idx[bin_mask]
            if len(bin_indices) > 0:
                dFF_matrix[b, t_idx] = np.nanmean(dFF[bin_indices])

    meanFF = np.nanmean(dFF_matrix, axis=1)
    normmeanFF = (meanFF - np.nanmin(meanFF)) / (np.nanmax(meanFF) - np.nanmin(meanFF))
    normFF_trials = np.array([
        (trial - np.nanmin(trial)) / (np.nanmax(trial) - np.nanmin(trial)) if np.nanmax(trial) != np.nanmin(trial) else trial
        for trial in dFF_matrix.T
    ]).T

    return normmeanFF, meanFF, normFF_trials, dFF_matrix

# %%
def has_internal_nan_gap(trial, gap_len=1):
    isnan = np.isnan(trial)

    # Trim NaNs at start and end
    start = 0
    end = len(trial) - 1
    while start < len(trial) and isnan[start]:
        start += 1
    while end > 0 and isnan[end]:
        end -= 1

    # Search for contiguous NaN gap *within* trimmed region
    count = 0
    for val in isnan[start:end+1]:
        count = count + 1 if val else 0
        if count >= gap_len:
            return True
    return False

def remove_trials_with_internal_nan_gap(data, gap_len=1):
    """Remove trials (columns) that have a long NaN gap in the middle"""
    return data[:, [not has_internal_nan_gap(data[:, i], gap_len) for i in range(data.shape[1])]]

def pad_with_nan(arr, target_shape):
    padded = np.full(target_shape, np.nan)
    padded[:arr.shape[0], :arr.shape[1]] = arr
    return padded


def nan_after_reward_per_trial(signal, reward, trialnum):
    """
    Nans out values in `signal` after the first reward within each trial.

    Parameters:
    - signal: 1D array of the signal to modify (same length as reward)
    - reward: 1D binary array marking reward delivery (same length)
    - trialnum: 1D array marking trial identity (same length)

    Returns:
    - signal_nan: copy of `signal` with values after reward nan'ed out per trial
    """
    signal_nan = signal.copy()
    unique_trials = np.unique(trialnum)

    for tr in unique_trials:
        trial_mask = trialnum == tr
        signal_ = signal_nan[trial_mask]
        reward_ = reward[trial_mask]

        reward_idx = np.where(reward_ > 0)[0]
        if len(reward_idx) > 0:
            cut_idx = reward_idx[0] + 1  # nan everything after reward
            signal_[cut_idx:] = np.nan

        signal_nan[trial_mask] = signal_

    return signal_nan

def nan_after_reward_all_epochs(signal, reward, trialnum, eps):


    """
    Nans out signal values after the first reward in each trial, for each epoch.

    Parameters:
    - signal: 1D array of signal values (e.g., lick_rate or dFF)
    - reward: 1D binary array of same length indicating reward delivery
    - trialnum: 1D array of same length, trial numbers (reset to 0 at each epoch)
    - eps: 1D array of indices marking epoch start (e.g., [0, 5000, 10000, ...])

    Returns:
    - signal_nan: 1D array with same shape as `signal`, but NaNs after reward in each trial
    """
    signal_nan = signal.copy()

    for e in range(len(eps) - 1):
        start = eps[e]
        end = eps[e + 1]

        signal_ep = signal[start:end]
        reward_ep = reward[start:end]
        trialnum_ep = trialnum[start:end]

        unique_trials = np.unique(trialnum_ep)

        for tr in unique_trials:
            trial_mask = trialnum_ep == tr
            signal_ = signal_ep[trial_mask]
            reward_ = reward_ep[trial_mask]

            reward_idx = np.where(reward_ > 0)[0]
            if len(reward_idx) > 0:
                cut_idx = reward_idx[0] + 1
                signal_[cut_idx:] = np.nan

            signal_ep[trial_mask] = signal_

        signal_nan[start:end] = signal_ep

    return signal_nan

