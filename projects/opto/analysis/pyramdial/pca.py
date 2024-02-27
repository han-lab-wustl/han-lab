"""zahra's analysis for clustering/dimensionality reduction of pyramidal cell data

"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

#%%
days = [35, 38, 41, 44, 47, 50]
optoep = [3, 2, 3, 2, 3, 2]
track_length = 270
range_to_test = 55 # window around rew loc to look for coms
density_per_day = []
keys_to_keep = ['stat', 'ops', 'F', 'Fneu', 'spks', 'iscell', \
                'redcell', 'VR', 'scanstart', 'scanstop', 'timedFF_ALL', 'timedFF', 'rewards', \
                'licks', 'lickVoltage', 'ybinned', 'trialnum', 'forwardvel', 'changeRewLoc', \
                'dFF', 'Fc3', 'putative_pcs', 'bordercells', 'coms', 'tuning_curves', \
                'coms_tracked_cells_pos', 'tuning_curves_tracked_cells_pos']
falls_per_day = {}
for day in days:
    params_pth = rf"Y:\analysis\fmats\e218\days\e218_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth)
    fall_keys = fall.keys()
    # fix formatting
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    eps = np.append(eps, len(changeRewLoc))
    # try on tuning curves
    density_coms = []

    for ii, ep in enumerate(eps):
        if not ii==len(eps)-1:
            # ep3 = fall['tuning_curves'][0][2]
            # ep2 = fall['tuning_curves'][0][1]
            # ep1 = fall['tuning_curves'][0][0]
            # binsize = 3 # cm
            # ep3_pre_rew = ep3[int(np.ceil((rewlocs[ep-1]-50)/binsize)):int(np.ceil(rewlocs[ep-1]/binsize))]
            # ep2_pre_rew = ep2[int(np.ceil((rewlocs[ep-2]-50)/binsize)):int(np.ceil(rewlocs[ep-2]/binsize))]
            # ep1_pre_rew = ep2[int(np.ceil((rewlocs[ep-3]-50)/binsize)):int(np.ceil(rewlocs[ep-3]/binsize))]
            # look at cells that have their coms 50 cm before rew in ep1
            # what happens to their coms in the next eps?
            ep_com = np.hstack(fall['coms'][0][ii])            
        
    density_per_day.append(density_coms)
    fall_dct = {}
    for k in keys_to_keep:
        fall_dct[k] = fall[k]
    falls_per_day[day] = fall_dct
    print(day)

with open(r"X:\e218_falls_opto.p", "wb") as fp:   #Pickling
        pickle.dump(falls_per_day, fp)
# plot com density results
optoep = [3, 2, 3, 2, 3, 2]
density_opto = [xx[optoep[yy]-1]for yy,xx in enumerate(density_per_day)]
density_prev = [xx[optoep[yy]-2]for yy,xx in enumerate(density_per_day)]

from placecell import normalize_tuning_curve, calculate_difference, plot_tuning_curves, \
find_differentially_inactivated_cells, plot_differential_activity, make_tuning_curves
trialnum, rewards, ybinned = fall['trialnum'][0], fall['rewards'][0], fall['ybinned'][0]
gainf, ntrials, licks, forwardvel = 3/2, 5, fall['licks'][0], fall['forwardvel'][0]
thres, Fs, ftol, bin_size, track_length = 5, 31.25, 10, 3, 270
# Convert list of arrays to a 2D NumPy array (equivalent to MATLAB's cell2mat and reshape)
pcs = np.vstack(putative_pcs)
pc = fall['iscell'][:, 0].astype(bool)
# Use boolean indexing to mask border cells (equivalent to MATLAB's logical indexing)
bordercells_pc = fall['bordercells'][0][pc]
# Assuming Fc3 and dFF are NumPy arrays that have been defined or loaded elsewhere
# Filter out border cells and apply place cell filter
fc3_pc = fall['Fc3'][:, ~bordercells_pc]
dff_pc = fall['dFF'][:, ~bordercells_pc]
tc,coms = make_tuning_curves(eps, trialnum, rewards, ybinned, gainf, ntrials, licks, forwardvel, thres, Fs, ftol, bin_size, track_length, fc3_pc, dff_pc)
# Normalize the tuning curves
normalized_tc1 = normalize_tuning_curve(fall['tuning_curves'][0][0])
normalized_tc2 = normalize_tuning_curve(fall['tuning_curves'][0][1])

tc1 = (fall['tuning_curves'][0][0]).T
tc2 = (fall['tuning_curves'][0][1]).T

# Define a threshold for differential inactivation (e.g., 0.2 difference in normalized mean activity)
threshold = 0.02
# Find differentially inactivated cells
differentially_inactivated_cells = find_differentially_inactivated_cells(normalized_tc1.T, normalized_tc2.T, threshold)
# Optionally, plot the tuning curves of these cells for both conditions
plot_differential_activity(differentially_inactivated_cells, tc1, tc2)