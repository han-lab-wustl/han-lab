"""
zahra
get trial by trial heatmap of rew cells
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
savepth = os.path.join(savedst, 'trial_by_trial_cell_cell_corr.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
bins = 150
goal_cm_window=20
dfs = []; lick_dfs = []
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
ii=64 
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if animal=='e145' or animal=='e139': pln=2 
else: pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
   'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells','ops',
   'stat', 'licks','putative_pcs'])
VR = fall['VR'][0][0][()]
pcs=np.squeeze(np.stack(fall['putative_pcs'][0]))
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
lick=fall['licks'][0]
time=fall['timedFF'][0]
if animal=='e145':
   ybinned=ybinned[:-1]
   forwardvel=forwardvel[:-1]
   changeRewLoc=changeRewLoc[:-1]
   trialnum=trialnum[:-1]
   rewards=rewards[:-1]
   lick=lick[:-1]
   time=time[:-1]
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
rz = get_rewzones(rewlocs,1/scalingf)       
# dark time params
track_length_dt = 550 # cm estimate based on 99.9% of ypos
track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
bins_dt=150 
bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
# added to get anatomical info
# takes time

fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
#if pc in all but 1
pc_bool = np.sum(pcs,axis=0)>=len(eps)-2
# looser restrictions
pc_bool = np.sum(pcs,axis=0)>=1        
if animal=='e200' or animal=='e217' or animal=='z17':
   Fc3 = Fc3[:,(skew>1)]
else:
   Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greater than 2
# if no cells pass these crit
if Fc3.shape[1]>0:
   print('#############making tcs#############\n')
   bin_size=3
   tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
   Fc3,trialnum,rewards,forwardvel,
   rewsize,bin_size) # last 5 trials

   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 
   goal_window = 20*(2*np.pi/track_length) # cm converted to rad
   coms_rewrel = np.array([com-np.pi for com in coms_correct])
   perm = list(combinations(range(len(coms_correct)), 2)) 
   # account for cells that move to the end/front
   # Define a small window around pi (e.g., epsilon)
   epsilon = .7 # 20 cm
   # Find COMs near pi and shift to -pi
   com_loop_w_in_window = []
   for pi,p in enumerate(perm):
      for cll in range(coms_rewrel.shape[1]):
         com1_rel = coms_rewrel[p[0],cll]
         com2_rel = coms_rewrel[p[1],cll]
         # print(com1_rel,com2_rel,com_diff)
         if ((abs(com1_rel - np.pi) < epsilon) and 
         (abs(com2_rel + np.pi) < epsilon)):
                  com_loop_w_in_window.append(cll)
   # get abs value instead
   coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
   com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
   com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
   com_goal=[xx for xx in com_goal if len(xx)>0]
   if len(com_goal)>0:
      goal_cells = intersect_arrays(*com_goal)
      # np.unique(np.concatenate(com_goal))
   else:
      goal_cells=[]
   # get cells that maintain their coms across at least 2 epochs
   place_window = 20 # cm converted to rad                
   com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
   compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
   # get cells across all epochs that meet crit
   pcs = np.unique(np.concatenate(compc))
   compc=[xx for xx in compc if len(xx)>0]
   if len(compc)>0:
      pcs_all = intersect_arrays(*compc)
      # exclude goal cells
      pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
   else:
      pcs_all=[]      
      
   meanim = fall['ops'][0]['meanImg'][0]
   stat = fall['stat'][0]
   # get cell stats
   if animal=='e200' or animal=='e217' or animal=='z17':
      stat = stat[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))][(skew>1)]
   else:
      stat = stat[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))][(skew>2)] # only keep cells with skew greater than 2
   pc_stat = stat[pcs_all]
   gc_stat = stat[goal_cells]
      
   # create a color mask (RGB) same size as mean image
   mask_rgb = np.zeros(meanim.shape + (3,), dtype=np.float32)  # float for overlay
   # overlay place cells in red
   for cell in pc_stat:
      ypix = cell['ypix'][0][0][0]
      xpix = cell['xpix'][0][0][0]
      mask_rgb[ypix, xpix, 0] = 1  # Red channel

   # overlay goal cells in blue
   for cell in gc_stat:
      ypix = cell['ypix'][0][0][0]
      xpix = cell['xpix'][0][0][0]
      mask_rgb[ypix, xpix, 2] = 1  # Blue channel

   # normalize mean image for display
   meanim_norm = (meanim - meanim.min()) / (meanim.max() - meanim.min())
   # display
   plt.figure(figsize=(8,8))
   plt.imshow(meanim_norm, cmap='gray')           # base image
   plt.imshow(mask_rgb, alpha=0.5)               # overlay masks
   plt.title(f'{animal},{day},Place Cells (Red) and Reward Cells (Blue)')
   plt.axis('off')
   plt.show()