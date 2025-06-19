"""
zahra
circularly shuffle dff
95% spatial info cells
split into place and rew
june 25
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

def fast_si_vectorized(dff, pos_bins, n_bins):
    """
    Compute spatial information using fully vectorized operations.
    
    Parameters:
        dff : ndarray
            Shape (n_cells, n_timepoints) or (n_cells, n_timepoints, n_shuffles)
        pos_bins : 1D array of ints, shape (n_timepoints,)
        n_bins : int, total number of spatial bins

    Returns:
        si : ndarray
            Shape (n_cells,) or (n_cells, n_shuffles)
    """
    pos_bins = np.asarray(pos_bins)
    time_mask = ~np.isnan(pos_bins)
    pos_bins = pos_bins.astype(int)

    # Create 2D mask: (n_bins, n_timepoints)
    bin_mask = np.equal.outer(np.arange(n_bins), pos_bins)  # (n_bins, timepoints)

    # Normalize occupancy (p_i)
    p_i = bin_mask.sum(axis=1) / len(pos_bins)  # (n_bins,)

    # If dff is 2D: (n_cells, n_timepoints)
    if dff.ndim == 2:
        n_cells, n_timepoints = dff.shape
        bin_mask = bin_mask.astype(bool)

        # Compute mean rate per bin: (n_cells, n_bins)
        r_i = np.array([
            np.nanmean(dff[:, bin_mask[b]], axis=1)
            for b in range(n_bins)
        ]).T  # (n_cells, n_bins)

        # Mean rate per cell
        r = np.nanmean(dff, axis=1, keepdims=True)  # (n_cells, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = r_i / r
            log_term = np.log2(ratio, where=(ratio > 0))
            si = np.nansum(p_i * ratio * log_term, axis=1)

        return np.nan_to_num(si)

    # If dff is 3D: (n_cells, n_timepoints, n_shuffles)
    elif dff.ndim == 3:
        n_cells, n_timepoints, n_shuffles = dff.shape
        bin_mask = bin_mask.astype(bool)

        # Preallocate
        r_i = np.empty((n_cells, n_bins, n_shuffles))

        for b in range(n_bins):
            # Mean over timepoints in bin b: (n_cells, n_shuffles)
            r_i[:, b, :] = np.nanmean(dff[:, bin_mask[b], :], axis=1)

        # Mean rate per cell & shuffle: (n_cells, 1, n_shuffles)
        r = np.nanmean(dff, axis=1, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = r_i / r
            log_term = np.log2(ratio, where=(ratio > 0))
            si = np.nansum(p_i[:, None] * ratio * log_term, axis=1)

        return np.nan_to_num(si)

    else:
        raise ValueError("dff must be 2D or 3D (cells x time or cells x time x shuffles)")

def compute_spatial_information_all(dff_shuffled, pos_bins, n_bins):
   """
   Compute spatial information for all shuffles.
   
   Parameters:
      dff_shuffled : np.ndarray, shape (n_cells, n_timepoints, n_shuffles)
      pos_bins : np.ndarray, shape (n_timepoints,)
      n_bins : int, number of spatial bins
   
   Returns:
      si_shuffled : np.ndarray, shape (n_cells, n_shuffles)
   """
   n_cells, _, n_shuffles = dff_shuffled.shape
   si_shuffled = np.zeros((n_cells, n_shuffles))

   for s in range(n_shuffles):
      if s%10==0: print(s)
      for i in range(n_cells):
         si_shuffled[i, s] = compute_spatial_information(dff_shuffled[i, :, s], pos_bins, n_bins)

   return si_shuffled

def blockwise_circular_permute_dff(dff, segment_len=100, n_shuffles=1000, random_state=None):
   """
   Perform blockwise circular permutation on dF/F traces.
   
   Parameters:
      dff : np.ndarray, shape (n_cells, n_timepoints)
      segment_len : int, length of each segment
      n_shuffles : int, number of shuffled iterations
      random_state : int or np.random.Generator
   
   Returns:
      dff_shuffled : np.ndarray, shape (n_cells, n_timepoints, n_shuffles)
   """
   rng = np.random.default_rng(random_state)
   n_cells, n_timepoints = dff.shape
   n_segments = n_timepoints // segment_len

   if n_timepoints % segment_len != 0:
      dff = dff[:, :(n_segments * segment_len)]  # trim to full segments
      n_timepoints = dff.shape[1]

   # reshape into blocks: (n_cells, n_segments, segment_len)
   dff_blocks = dff.reshape(n_cells, n_segments, segment_len)

   # storage
   dff_shuffled = np.zeros((n_cells, n_timepoints, n_shuffles))

   for s in range(n_shuffles):
      shifts = rng.integers(0, n_segments, size=n_cells)
      for i in range(n_cells):
         # circular shift along segments
         perm_blocks = np.roll(dff_blocks[i], shift=shifts[i], axis=0)
         dff_shuffled[i, :, s] = perm_blocks.reshape(-1)

   return dff_shuffled

#%%
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
# initialize var
datadct = {} # overwrite
place_window = 20
num_iterations=1000
bin_size=3 # cm
lasttr=8 # last trials
bins=90

# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if animal=='e145': pln=2 
   else: pln=0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
   print(params_pth)
   fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF','licks',
      'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
      'stat'])
   pcs = np.vstack(np.array(fall['putative_pcs'][0]))
   VR = fall['VR'][0][0][()]
   scalingf = VR['scalingFACTOR'][0][0]
   try:
      rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
   except:
      rewsize = 10
   ybinned = fall['ybinned'][0]/scalingf
   track_length=180/scalingf    
   forwardvel = fall['forwardvel'][0]    
   changeRewLoc = np.hstack(fall['changeRewLoc'])
   trialnum=fall['trialnum'][0]
   rewards = fall['rewards'][0]
   time = fall['timedFF'][0]
   lick = fall['licks'][0]
   if animal=='e145':
      ybinned=ybinned[:-1]
      forwardvel=forwardvel[:-1]
      changeRewLoc=changeRewLoc[:-1]
      trialnum=trialnum[:-1]
      rewards=rewards[:-1]
      time=time[:-1]
      lick=lick[:-1]
   # set vars
   eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
   # only test opto vs. ctrl
   eptest = conddf.optoep.values[ii]
   if conddf.optoep.values[ii]<2: 
      eptest = random.randint(2,3)   
      if len(eps)<4: eptest = 2 # if no 3 epochs    
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   dFF_org = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   skew = scipy.stats.skew(dFF_org, nan_policy='omit', axis=0)
   dFF=dFF_org[:, skew>2]
   # low cells
   if animal=='e217' or animal=='z17' or animal=='z14' or animal=='e200':
      dFF=dFF_org[:, skew>1]
   # Assume dff is a NumPy array of shape (n_cells, n_timepoints)
   dff_permuted = circularly_permute_dff(dFF, random_state=42)
   n_bins = int(np.max(np.ceil(ybinned))) + 1  # if pos_bins are 0-indexed
   # Compute true SI
   real_si = np.array([compute_spatial_information(dFF.T[i], np.ceil(ybinned), n_bins) for i in range(dFF.T.shape[0])])
   # To get percentile threshold:
   # Generate shuffle distribution (circular permutation)
   pos_bins=np.ceil(ybinned)
   dff_shuffle = blockwise_circular_permute_dff(dFF.T, segment_len=100, n_shuffles=1000, random_state=42)
   # Shuffled SI (blockwise shuffles from before)
   si_shuff = fast_si_vectorized(dff_shuffle, pos_bins, n_bins)
   si_threshold = np.percentile(si_shuff, 95, axis=1)  # 95th percentile per cell
   spatial_tuned_cells = np.where(real_si>si_threshold)[0]
   dFF=dFF[:,spatial_tuned_cells]
   if conddf.optoep.values[ii]<2: 
      eptest = random.randint(2,3)      
   if len(eps)<4: eptest = 2 # if no 3 epochs
   comp = [eptest-2,eptest-1] # eps to compare, python indexing   
   # pre-reward vs. all
   dff_prev_prerew = np.nanmean(dFF[eps[comp[0]]:eps[comp[1]],:][ybinned[eps[comp[0]]:eps[comp[1]]]<rewlocs[comp[0]]-rewsize/2,:])
   dff_opto_prerew = np.nanmean(dFF[eps[comp[1]]:eps[comp[1]+1],:][ybinned[eps[comp[1]]:eps[comp[1]+1]]<rewlocs[comp[1]]-rewsize/2,:])
   # all
   dff_prev = np.nanmean(dFF[eps[comp[0]]:eps[comp[1]],:])
   dff_opto_prerew = np.nanmean(dFF[eps[comp[1]]:eps[comp[1]+1],:])
   # just dark time ffun 
   dff_prev_dt = np.nanmean(dFF[eps[comp[0]]:eps[comp[1]],:][ybinned[eps[comp[0]]:eps[comp[1]]]<4])
   dff_opto_prerew_dt = np.nanmean(dFF[eps[comp[1]]:eps[comp[1]+1],:][ybinned[eps[comp[1]]:eps[comp[1]+1]]<4])
   # activity of all spatially tuned cells
   datadct[f'{animal}_{day:03d}']=[dff_prev_prerew,dff_opto_prerew,dff_prev,dff_opto_prerew,dff_prev_dt,dff_opto_prerew_dt]
   print(f'{animal}, optoep{conddf.optoep.values[ii]}, diff:{dff_opto_prerew-dff_prev_prerew}')
#%%