import numpy as np, math, scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import numpy as np
from scipy.signal import gaussian

def get_moving_time(velocity, thres, Fs, ftol):
    """
    Returns time points when the animal is considered moving based on animal's change in y position.

    Parameters:
    velocity (numpy.ndarray): forward velocity
    thres (float): Threshold speed in cm/s
    Fs (int): number of frames length minimum to be considered stopped
    ftol (int): frame tolerance (e.g., 10 frames)

    Returns:
    numpy.ndarray: moving_middle (time points when the animal is considered moving)
    numpy.ndarray: stop (time points when the animal is considered stopped)
    """

    vr_speed = velocity
    vr_thresh = thres

    moving = np.where(vr_speed > vr_thresh)[0]
    stop = np.where(vr_speed <= vr_thresh)[0]

    stop_time_stretch = consecutive_stretch(stop)
    stop_time_length = [len(stretch) for stretch in stop_time_stretch]
    delete_idx = np.array(stop_time_length) < Fs
    stop_time_stretch = ([np.array(stretch) for i, stretch in enumerate(stop_time_stretch) if not delete_idx[i]])

    if len(stop_time_stretch) > 0:
        for s in range(len(stop_time_stretch) - 1):
            if s + 1 < len(stop_time_stretch):
                if not np.isnan(stop_time_stretch[s + d]).any():
                    while (abs(stop_time_stretch[s][-1] - stop_time_stretch[s + d][0]) <= ftol) and s + d < len(stop_time_stretch):
                        stop_time_stretch[s] = np.concatenate((stop_time_stretch[s], np.arange(stop_time_stretch[s][-1] + 1, stop_time_stretch[s + d][0]), stop_time_stretch[s + d]))

        stop_time_stretch = [stretch for stretch in stop_time_stretch if not np.isnan(stretch).any()]
        stop = np.concatenate(stop_time_stretch)
        moving_time = np.ones(len(vr_speed), dtype=int)
        moving_time[stop] = 0
    else:
        moving_time = np.ones(len(vr_speed), dtype=int)

    moving = np.where(moving_time == 1)[0]
    moving_middle = moving

    return moving_middle, stop

def get_spatial_info_per_cell(Fc3, fv, thres, ftol, position, Fs, nBins, track_length):
    """
    Fc3: dFF of 1 cell
    position: position of animal on track
    Fs: Frame rate of acquisition
    nBins: number of bins in which you want to divide the track into
    track_length: Length of track
    """

    time_moving, _ = get_moving_time(fv, thres, Fs, ftol)
    bin_size = track_length / nBins
    pos_moving = position[time_moving]

    time_in_bin = {i: time_moving[np.logical_and(pos_moving > (i - 1) * bin_size, pos_moving <= i * bin_size)] for i in range(1, nBins + 1)}

    cell_activity = np.array([np.mean(Fc3[time_in_bin[bin]]) for bin in range(1, nBins + 1)])
    # cell_activity = gaussian(cell_activity, 5)  # Uncomment if you want to apply Gaussian smoothing

    lambda_all = np.mean(Fc3[time_moving])
    time_fraction = np.array([len(time_in_bin[bin]) / len(time_moving) for bin in range(1, nBins + 1)])

    temp = time_fraction * cell_activity * np.log2(cell_activity / lambda_all)
    temp[np.isinf(temp)] = 0
    temp[np.isnan(temp)] = 0

    info = np.sum(temp / lambda_all)

    if np.isnan(info):
        info = 0

    return info


def convert_coordinates(coordinates, center_location, track_length=270):
    """
    Convert track coordinates from 0 to track_length (default: 270 cm) to -pi to pi radians,
    centered at a specified location.

    Args:
        coordinates (numpy.ndarray): 1D array of track coordinates in cm.
        center_location (float): Location to center the coordinates at, in cm.
        track_length (float, optional): Length of the track in cm (default: 270).

    Returns:
        numpy.ndarray: Converted coordinates in radians, centered at the specified location.
    """
    # Convert coordinates and center_location to radians
    coordinates_radians = coordinates * (2 * np.pi / track_length)
    center_radians = center_location * (2 * np.pi / track_length)

    # Center coordinates_radians around center_radians
    centered_coordinates_radians = coordinates_radians - center_radians

    # Wrap the centered_coordinates_radians to -pi to pi range
    centered_coordinates_radians = (centered_coordinates_radians + np.pi) % (2 * np.pi) - np.pi

    return centered_coordinates_radians


def intersect_arrays(*arrays):
    """
    Find the intersection between multiple NumPy arrays.

    Args:
        *arrays: Variable number of NumPy arrays.

    Returns:
        numpy.ndarray: Array containing the intersection of all input arrays.
    """
    # Convert arguments to a list of arrays
    arrays = list(arrays)

    # Base case: If there is only one array, return it
    if len(arrays) == 1:
        return arrays[0]

    # Find the intersection between the first two arrays
    intersection = np.intersect1d(arrays[0], arrays[1])

    # Find the intersection between the result and the remaining arrays
    for arr in arrays[2:]:
        intersection = np.intersect1d(intersection, arr)

    return intersection
    
def evaluate_place_field_width(tuning_curve, bin_centers, threshold=0.3):
    """
    Evaluate the width of a place field from a tuning curve calculated from calcium imaging data.

    Args:
        tuning_curve (numpy.ndarray): 1D array containing the tuning curve values.
        bin_centers (numpy.ndarray): 1D array containing the bin centers corresponding to the tuning curve values.
        threshold (float, optional): Threshold for determining the place field boundaries (default: 0.5).

    Returns:
        float: Width of the place field in the same units as bin_centers.
        None: If no place field is detected.
    """
    # Normalize the tuning curve to [0, 1] range
    tuning_curve = (tuning_curve - np.min(tuning_curve)) / (np.max(tuning_curve) - np.min(tuning_curve))

    # Find the indices where the tuning curve crosses the threshold
    above_threshold = tuning_curve >= threshold
    crossings = np.where(np.diff(above_threshold.astype(int)))[0]

    # If there are no crossings or an odd number of crossings, no place field is detected
    if len(crossings) == 0 or len(crossings) % 2 != 0:
        return None

    # Find the bin centers corresponding to the place field boundaries
    field_boundaries = []
    for i in range(0, len(crossings), 2):
        boundary_left = bin_centers[crossings[i]]
        boundary_right = bin_centers[crossings[i + 1]]
        field_boundaries.append((boundary_left, boundary_right))

    # Calculate the width of the place field as the difference between the boundaries
    place_field_widths = [right - left for left, right in field_boundaries]

    # Return the maximum width (in case of multiple place fields)
    return max(place_field_widths)


def convert_com_to_radians(com, reward_location, track_length):
    """
    Convert the center of mass of pyramidal cell activity from 0 to 270 cm
    to -pi to pi radians, centered at the reward location.

    Args:
        com (float): Center of mass of pyramidal cell activity in cm (0 to 270).
        reward_location (float): Reward location in cm (0 to 270).

    Returns:
        float: Center of mass in radians (-pi to pi), centered at the reward location.
    """
    # Convert com and reward_location to radians
    com_radians = com * (2 * math.pi / track_length)
    reward_radians = reward_location * (2 * math.pi / track_length)

    # Center com_radians around reward_radians
    centered_com_radians = com_radians - reward_radians

    # Wrap the centered_com_radians to -pi to pi range
    while centered_com_radians > math.pi:
        centered_com_radians -= 2 * math.pi
    while centered_com_radians < -math.pi:
        centered_com_radians += 2 * math.pi

    return centered_com_radians
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
        auc_tc1.append(np.nanquantile([np.trapz(tuning_curve1[cll,tr],dx=binsize) for tr in transients],0.75))
    for cll in range(tuning_curve2.shape[0]):
        transients = consecutive_stretch(np.where(tuning_curve2[cll,:]>0)[0])
        transients = [xx for xx in transients if len(xx)>0]
        auc_tc2.append(np.nanquantile([np.trapz(tuning_curve2[cll,tr],dx=binsize) for tr in transients],0.75))
    
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

def get_pyr_metrics_opto(conddf, dd, day, threshold=5, pc = False):
    track_length = 270
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    if not pc:
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
            'tuning_curves_late_trials', 'coms_early_trials'])
        coms = fall['coms'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
    else:
        fall = scipy.io.loadmat(params_pth, variable_names=['coms_pc_late_trials', 'changeRewLoc', 'tuning_curves_pc_early_trials',\
            'tuning_curves_pc_late_trials', 'coms_pc_early_trials'])
        coms = fall['coms_pc_late_trials'][0]
        tcs_early = fall['tuning_curves_pc_early_trials'][0]
        tcs_late = fall['tuning_curves_pc_late_trials'][0]
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    eps = np.append(eps, len(changeRewLoc))    
    if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    bin_size = 3    
    tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0]]]))
    tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1]]]))
    tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0]]]))
    tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1]]]))    
    # replace nan coms
    # peak = np.nanmax(tc1_late,axis=1)
    # coms1_max = np.array([np.where(tc1_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
    # peak = np.nanmax(tc2_late,axis=1)
    # coms2_max = np.array([np.where(tc2_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])    
    coms1 = np.hstack(coms[comp[0]])
    coms2 = np.hstack(coms[comp[1]])
    # coms1[np.isnan(coms1)]=coms1_max[np.isnan(coms1)]
    # coms2[np.isnan(coms2)]=coms2_max[np.isnan(coms2)]
    # take fc3 in area around com
    difftc1 = tc1_late-tc1_early
    coms1_bin = np.floor(coms1/bin_size).astype(int)
    difftc1 = np.array([np.nanmean(difftc1[ii,com-2:com+2]) for ii,com in enumerate(coms1_bin)])
    difftc2 = tc2_late-tc2_early
    coms2_bin = np.floor(coms2/bin_size).astype(int)
    difftc2 = np.array([np.nanmean(difftc2[ii,com-2:com+2]) for ii,com in enumerate(coms2_bin)])

    # Find differentially inactivated cells
    # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    # differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late, tc2_late, threshold, bin_size)
    differentially_activated_cells = find_differentially_activated_cells(tc1_late, tc2_late, threshold, bin_size)
    # tc1_pc_width = evaluate_place_field_width(tc1_late, bin_centers, threshold=0.5)
    rewloc_shift = rewlocs[comp[1]]-rewlocs[comp[0]]
    com_shift = [np.nanmean(coms[comp[1]][differentially_inactivated_cells]-coms[comp[0]][differentially_inactivated_cells]), \
                np.nanmean(coms[comp[1]][differentially_activated_cells]-coms[comp[0]][differentially_activated_cells]), \
                    np.nanmean(coms[comp[1]]-coms[comp[0]])]
    # circular alignment
    rel_coms1 = [convert_com_to_radians(com, rewlocs[comp[0]], track_length) for com in coms1]
    rel_coms2 = [convert_com_to_radians(com, rewlocs[comp[1]], track_length) for com in coms2]
    # rel_coms2 = np.hstack([(coms2[coms2<=rewlocs[comp[1]]]-rewlocs[comp[1]])/rewlocs[comp[1]],(coms2[coms2>rewlocs[comp[1]]]-rewlocs[comp[1]])/(track_length-rewlocs[comp[1]])])
    # rel_coms2 = (coms2-rewlocs[comp[1]])/rewlocs[comp[1]]
    dct['rel_coms1'] = np.array(rel_coms1)
    dct['rel_coms2'] = np.array(rel_coms2)
    dct['learning_tc1'] = [tc1_early, tc1_late]
    dct['learning_tc2'] = [tc2_early, tc2_late]
    dct['difftc1'] = difftc1
    dct['difftc2'] = difftc2
    dct['rewzones_comp'] = rewzones[comp]
    dct['coms1'] = coms1
    dct['coms2'] = coms2
    dct['frac_place_cells_tc1'] = sum((coms1>(rewlocs[comp[0]]-(track_length*.07))) & (coms1<(rewlocs[comp[0]])+5))/len(coms1[(coms1>bin_size) & (coms1<=(track_length/bin_size))])
    dct['frac_place_cells_tc2'] = sum((coms2>(rewlocs[comp[1]]-(track_length*.07))) & (coms2<(rewlocs[comp[1]])+5))/len(coms2[(coms2>bin_size) & (coms2<=(track_length/bin_size))])
    dct['rewloc_shift'] = rewloc_shift
    dct['com_shift'] = com_shift
    dct['inactive'] = differentially_inactivated_cells
    dct['active'] = differentially_activated_cells
    dct['rewlocs_comp'] = rewlocs[comp]
    return dct

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
