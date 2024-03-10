import numpy as np
from scipy.ndimage import gaussian_filter1d
import numpy as np

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def get_rewzones(rewlocs, gainf):
    # Initialize the reward zone numbers array with zeros
    rewzonenum = np.zeros(len(rewlocs))
    
    # Iterate over each reward location to determine its reward zone
    for kk, loc in enumerate(rewlocs):
        if loc <= 86 * gainf:
            rewzonenum[kk] = 1  # Reward zone 1
        elif 101 * gainf <= loc <= 120 * gainf:
            rewzonenum[kk] = 2  # Reward zone 2
        elif loc >= 135 * gainf:
            rewzonenum[kk] = 3  # Reward zone 3
            
    return rewzonenum

def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = [x[:break_point[0]]]
    for i in range(1, len(break_point)):
        xx = x[break_point[i - 1] + 1:break_point[i]]
        y.append(xx)
    y.append(x[break_point[-1] + 1:])
    
    return y

def normalize_tuning_curve(tuning_curve):
    """
    Normalize a tuning curve to have values between 0 and 1.
    
    Parameters:
    tuning_curve (numpy.ndarray): The tuning curve to normalize.
    
    Returns:
    numpy.ndarray: Normalized tuning curve.
    """
    min_val = np.min(tuning_curve)
    max_val = np.max(tuning_curve)
    normalized = (tuning_curve - min_val) / (max_val - min_val)
    return normalized

def find_differentially_activated_cells(tuning_curve1, tuning_curve2, threshold, binsize):
    """
    Identify cells that are differentially inactivated between two conditions.
    
    Parameters:
    tuning_curve1 (np.ndarray): Tuning curve for condition 1 (cells x bins).
    tuning_curve2 (np.ndarray): Tuning curve for condition 2 (cells x bins).
    threshold (float): The threshold for considering a cell differentially inactivated.
    
    Returns:
    np.ndarray: Indices of cells considered differentially inactivated.
    """
    # Calculate the AUC across bins for each cell in each condition
    auc_tc1 = []; auc_tc2 = []
    for cll in range(tuning_curve1.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve1[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc1.append(np.nanmean([np.trapz(tuning_curve1[cll,tr],dx=binsize) for tr in transients]))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanmean([np.trapz(tuning_curve2[cll,tr],dx=binsize) for tr in transients]))
    
    mean_activity1 = np.array(auc_tc1)
    mean_activity2 = np.array(auc_tc2)
    
    # Find the difference in mean activity between conditions
    activity_diff = mean_activity1 - mean_activity2
    
    differentially_activated_cells = np.where(activity_diff < -threshold)[0]
    
    return differentially_activated_cells


def find_differentially_inactivated_cells(tuning_curve1, tuning_curve2, threshold, binsize):
    """
    Identify cells that are differentially inactivated between two conditions.
    
    Parameters:
    tuning_curve1 (np.ndarray): Tuning curve for condition 1 (cells x bins).
    tuning_curve2 (np.ndarray): Tuning curve for condition 2 (cells x bins).
    threshold (float): The threshold for considering a cell differentially inactivated.
    
    Returns:
    np.ndarray: Indices of cells considered differentially inactivated.
    """
    # Calculate the AUC across bins for each cell in each condition
    auc_tc1 = []; auc_tc2 = []
    for cll in range(tuning_curve1.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve1[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc1.append(np.nanmean([np.trapz(tuning_curve1[cll,tr],dx=binsize) for tr in transients]))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanmean([np.trapz(tuning_curve2[cll,tr],dx=binsize) for tr in transients]))
    
    mean_activity1 = np.array(auc_tc1)
    mean_activity2 = np.array(auc_tc2)
    
    # Find the difference in mean activity between conditions
    activity_diff = mean_activity1 - mean_activity2
    
    # Identify cells with a decrease in activity beyond the threshold
    differentially_inactivated_cells = np.where(activity_diff > threshold)[0]
    
    return differentially_inactivated_cells

        
def calculate_difference(tuning_curve1, tuning_curve2):
    """
    Calculate the difference between two normalized tuning curves.
    
    Parameters:
    tuning_curve1, tuning_curve2 (numpy.ndarray): The two tuning curves.
    
    Returns:
    numpy.ndarray: The difference between the tuning curves.
    """
    diff = tuning_curve1 - tuning_curve2
    return diff

def plot_tuning_curves(tuning_curve1, tuning_curve2, difference):
    """
    Plot two tuning curves and their difference.
    
    Parameters:
    tuning_curve1, tuning_curve2, difference (numpy.ndarray): The two tuning curves and their difference.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(tuning_curve1, label='Tuning Curve 1')
    plt.title('Tuning Curve 1')
    plt.xlabel('Spatial Bin')
    plt.ylabel('Normalized Fluorescence')
    
    plt.subplot(1, 3, 2)
    plt.plot(tuning_curve2, label='Tuning Curve 2')
    plt.title('Tuning Curve 2')
    plt.xlabel('Spatial Bin')
    
    plt.subplot(1, 3, 3)
    plt.plot(difference, label='Difference', color='red')
    plt.title('Difference in Tuning Curves')
    plt.xlabel('Spatial Bin')
    
    plt.tight_layout()
    plt.show()

def calc_COM_EH(spatial_act, bin_width):
    """
    Calculate the interpolated center of mass (COM) of spatial activity for each cell.

    Parameters:
    spatial_act (numpy.ndarray): 2D array representing the spatial activity (tuning curve) of cells across spatial bins.
    Shape is #cells x #bins.
    bin_width (float): The width of each spatial bin in cm.

    Returns:
    numpy.ndarray: The interpolated COM of each cell's spatial activity in cm.
    """
    num_cells, _ = spatial_act.shape
    bin_indices = np.zeros(num_cells)  # First bin above mid-point for each cell
    frac = np.zeros(num_cells)  # Fraction for interpolated COM
    com = np.zeros(num_cells)  # Interpolated COM in cm

    # Calculate cumulative sums and mid-points
    sum_spatial_act = np.nansum(spatial_act, axis=1)  # Total fluorescence from tuning curve, omitting NaN
    mid_sum = sum_spatial_act / 2  # Mid-point of total fluorescence
    spatial_act_cum_sum = np.nancumsum(spatial_act, axis=1)  # Cumulative sum of fluorescence in tuning curve

    # Find the bins above the mid-point of fluorescence
    idx_above_mid = spatial_act_cum_sum >= mid_sum[:, None]

    for i in range(num_cells):
        if not np.isnan(sum_spatial_act[i]):
            bin_indices[i] = np.argmax(idx_above_mid[i, :]) + 1  # Find index of first bin above mid-point, adjusting for Python indexing
            if bin_indices[i] == 1:  # If mid-point is in the first bin
                frac[i] = (spatial_act_cum_sum[i, int(bin_indices[i]-1)] - mid_sum[i]) / spatial_act_cum_sum[i, int(bin_indices[i]-1)]
                com[i] = frac[i] * bin_width
            else:
                frac[i] = (spatial_act_cum_sum[i, int(bin_indices[i]-1)] - mid_sum[i]) / \
                        (spatial_act_cum_sum[i, int(bin_indices[i]-1)] - spatial_act_cum_sum[i, int(bin_indices[i]-2)])
                com[i] = ((bin_indices[i]-1) + frac[i]) * bin_width
        else:
            com[i] = np.NaN

    return com

# Example usage
if __name__ == "__main__":
    # Example spatial activity data and bin width
    spatial_act = np.random.rand(10, 20)  # 10 cells x 20 bins
    bin_width = 5  # in cm

    com = calc_COM_EH(spatial_act, bin_width)
    print("Interpolated COM of each cell's spatial activity in cm:", com)

def get_moving_time(velocity, thres, Fs, ftol):
    """
    Returns time points when the animal is considered moving based on the change in y position.
    """
    vr_speed = velocity
    vr_thresh = thres
    moving = np.where(vr_speed > vr_thresh)[0]
    stop = np.where(vr_speed <= vr_thresh)[0]

    stop_time_stretch = consecutive_stretch(stop)

    stop_time_length = [len(stretch) for stretch in stop_time_stretch]
    stop_time_stretch = [stretch for stretch, length in zip(stop_time_stretch, stop_time_length) if length >= Fs]

    if len(stop_time_stretch) > 0:
        s = 0
        while s < len(stop_time_stretch) - 1:
            d = 1
            while s + d < len(stop_time_stretch) and abs(stop_time_stretch[s][-1] - stop_time_stretch[s + d][0]) <= ftol:
                stop_time_stretch[s] = np.concatenate([stop_time_stretch[s], np.arange(stop_time_stretch[s][-1] + 1, stop_time_stretch[s + d][0]), stop_time_stretch[s + d]])
                stop_time_stretch[s + d] = np.array([np.nan])  # Mark for deletion
                d += 1

            s += 1

        # Filter out NaN arrays
        stop_time_stretch = [stretch for stretch in stop_time_stretch if not np.isnan(stretch).all()]
        stop = np.concatenate(stop_time_stretch)
        moving_time = np.ones(len(vr_speed), dtype=int)
        moving_time[stop.astype(int)] = 0
    else:
        moving_time = np.ones(len(vr_speed), dtype=int)

    moving_middle = np.where(moving_time == 1)[0]
    return moving_middle, stop

def make_tuning_curves(eps, trialnum, rewards, ybinned, gainf, ntrials, licks, forwardvel, thres, Fs, ftol, bin_size, track_length, fc3, dff):
    nbins = int(track_length / bin_size)
    tuning_curves = []
    coms = []

    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep], eps[ep+1])
        trn = trialnum[eprng]
        rew = rewards[eprng] > 0.5
        
        strials = np.full(len(np.unique(trn)), np.nan)
        for ii,trial in enumerate(np.unique(trn)):
            if trial >= 3 and trial >= max(trn) - ntrials:
                strials[ii] = trial
        
        strials = strials[~np.isnan(strials)]
        mask = np.isin(trn, strials)
        eprng = eprng[mask]
        
        if len(eprng) > 0:
            ypos = ybinned[eprng]
            ypos = ypos * gainf
            fv = forwardvel[eprng]
            time_moving, _ = get_moving_time_V3(fv, thres, Fs, ftol)
            ypos_mov = ypos[time_moving]
            
            time_in_bin = [time_moving[(ypos_mov >= (i-1)*bin_size) & (ypos_mov < i*bin_size)] for i in range(1, nbins+1)]
            
            fc3_pc = fc3[eprng, :]
            dff_pc = dff[eprng, :]
            
            cell_activity = np.zeros((nbins, fc3_pc.shape[1]))
            cell_activity_dff = np.zeros((nbins, fc3_pc.shape[1]))
            for i in range(fc3_pc.shape[1]):
                for bin_idx in range(nbins):
                    if len(time_in_bin[bin_idx]) > 0:
                        cell_activity[bin_idx, i] = np.nanmean(fc3_pc[time_in_bin[bin_idx], i])
                        cell_activity_dff[bin_idx, i] = np.nanmean(dff_pc[time_in_bin[bin_idx], i])
                    else:
                        cell_activity[bin_idx, i] = np.nan
                        cell_activity_dff[bin_idx, i] = np.nan
            
            cell_activity[np.isnan(cell_activity)] = 0
            cell_activity_dff[np.isnan(cell_activity_dff)] = 0
            
        tuning_curves.append(cell_activity.T)
        
        median_com = calc_COM_EH(cell_activity.T, bin_size)
        coms.append(median_com)
        
        peak = np.array([bin_size * np.argmax(cell_activity[:, i]) if np.sum(cell_activity[:, i]) > 0 else 0 for i in range(cell_activity.shape[1])])
    
    return tuning_curves, coms, median_com, peak


# # Example usage
# if __name__ == "__main__":
#     # Example data
#     velocity = np.random.rand(1000) * 2  # Random velocities between 0 and 2
#     thres = 0.5  # Threshold velocity
#     Fs = 10  # Minimum number of frames to be considered stopped
#     ftol = 10  # Frame tolerance

#     moving_middle, stop = get_moving_time_V3(velocity, thres, Fs, ftol)
#     print("Moving:", moving_middle)
#     print("Stop:", stop)
