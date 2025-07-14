"""
compare led off vs. led on PER DAY
"""
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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves_by_trialtype_w_darktime_early, make_tuning_curves

import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
#%%
ii=40
iis=[159,172] # inhib
cm_window=20
for ii in iis:
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   if animal=='e145': pln=2  
   else: pln=0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
   print(params_pth)

   fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
   'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
   'stat', 'licks'])
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
   eptest=int(eptest)   
   lasttr=8 # last trials
   bins=90
   rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
   track_length_rad = track_length*(2*np.pi/track_length)
   bin_size=track_length_rad/bins

   if True:  
   # takes time
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      # for inhibi
      # Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
      # dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]

      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      # if animal!='z14' and animal!='e200' and animal!='e189':                
      Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
      # tc w/ dark time
      print('making tuning curves...\n')
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8) 
      bin_size=3
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size) # last 5 trials
      ###### rew
      goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
      coms_correct=coms_correct[[eptest-2,eptest-1],:]
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      perm = list(combinations(range(len(coms_correct)), 2)) 
      print(perm)
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
      goal_cells=np.unique(np.concatenate(com_goal))
      ######## place
      # get cells that maintain their coms across at least 2 epochs
      place_window = 20 # cm converted to rad      
      coms_correct_abs=coms_correct_abs[[eptest-2,eptest-1],:]          
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      pcs = [xx for xx in pcs if xx not in goal_cells]   

#%%
plt.rc('font', size=20)          # controls default text sizes
from matplotlib import colors
import matplotlib.patches as patches

clls_goal = goal_cells
# excit
tcs_correct_r = tcs_correct_abs[eptest-2:eptest,clls_goal]
coms_correct_r = coms_correct_abs[:,clls_goal]
epstotest = [eptest-2,eptest-1]
# Assuming tcs_correct is a list of 2D arrays and coms_correct is defined
# Determine the global min and max for normalization
# Create subplots
fig, axes = plt.subplots(ncols=len(tcs_correct_r), figsize=(9,7),sharex=True, sharey=True)
vmin = min(np.min(tcs) for tcs in tcs_correct_r)
vmax = max(np.max(tcs) for tcs in tcs_correct_r)
# Normalize to range [0, 1]
def normalize_rows(arr):
   arr_new = np.copy(arr)
   rowmins = np.nanmin(arr_new, axis=1, keepdims=True)
   rowmaxs = np.nanmax(arr_new, axis=1, keepdims=True)
   denom = rowmaxs - rowmins
   # Prevent division by zero
   denom[denom == 0] = 1
   arr_new = (arr_new - rowmins) / denom
   # Set entire row to 0 where max == min
   mask = (rowmaxs == rowmins).flatten()
   arr_new[mask, :] = 0
   return arr_new
lbl=['LED off','LED on']
# Apply normalization before plotting
for kk, tcs in enumerate(tcs_correct_r):
   tcs_normalized = normalize_rows(tcs)
   ax = axes[kk]
   im = ax.imshow(tcs_normalized[np.argsort(coms_correct_r[0])], aspect='auto',vmin=0,vmax=1)
   ax.set_title(f'{lbl[kk]} \n Correct trials')
   ax.axvline(rewlocs[epstotest][kk] / 3, color='w', linestyle='--',linewidth=3)
   if kk == 0:
      ax.set_ylabel('Reward cell #')
      ax.set_xlabel('Track position (cm)')
   patch = patches.Rectangle((0, 0),  # (x, y)
   width=rewlocs[epstotest][kk] / 3,
   height=tcs_normalized.shape[0],
   color='mediumspringgreen',
   alpha=0.2)
   if kk==1:ax.add_patch(patch)

ax.set_xticks([0, bins // 2, bins]) 
ax.set_xticklabels([0,135,270])

cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(f'Norm. $\Delta$ F/F')
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
plt.savefig(os.path.join(savedst, f'{animal}_{day}_tuning_curve_correct_v_incorrect_eg.svg'), bbox_inches='tight')
#%%
clls_goal = pcs
# excit
tcs_correct_r = tcs_correct_abs[eptest-2:eptest,clls_goal]
coms_correct_r = coms_correct_abs[:,clls_goal]
epstotest = [eptest-2,eptest-1]
# Assuming tcs_correct is a list of 2D arrays and coms_correct is defined
# Determine the global min and max for normalization
# Create subplots
fig, axes = plt.subplots(ncols=len(tcs_correct_r), figsize=(9,7),sharex=True, sharey=True)
vmin = min(np.min(tcs) for tcs in tcs_correct_r)
vmax = max(np.max(tcs) for tcs in tcs_correct_r)
# Normalize to range [0, 1]
def normalize_rows(arr):
   arr_new = np.copy(arr)
   rowmins = np.nanmin(arr_new, axis=1, keepdims=True)
   rowmaxs = np.nanmax(arr_new, axis=1, keepdims=True)
   denom = rowmaxs - rowmins
   # Prevent division by zero
   denom[denom == 0] = 1
   arr_new = (arr_new - rowmins) / denom
   # Set entire row to 0 where max == min
   mask = (rowmaxs == rowmins).flatten()
   arr_new[mask, :] = 0
   return arr_new
lbl=['LED off','LED on']
# Apply normalization before plotting
for kk, tcs in enumerate(tcs_correct_r):
   tcs_normalized = tcs#normalize_rows(tcs)
   ax = axes[kk]
   im = ax.imshow(tcs_normalized[np.argsort(coms_correct_r[0])], aspect='auto')
   ax.set_title(f'{lbl[kk]} \n Correct trials')
   ax.axvline(rewlocs[epstotest][kk] / 3, color='w', linestyle='--',linewidth=2)
   if kk == 0:
      ax.set_ylabel('Place cell #')
      ax.set_xlabel('Track position (cm)')
   patch = patches.Rectangle((0, 0),  # (x, y)
   width=rewlocs[epstotest][kk] / 3,
   height=tcs_normalized.shape[0],
   color='mediumspringgreen',
   alpha=0.2)
   if kk==1:ax.add_patch(patch)

ax.set_xticks([0, bins // 2, bins]) 
ax.set_xticklabels([0,135,270])

cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(f'Norm. $\Delta$ F/F')
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
plt.savefig(os.path.join(savedst, f'{animal}_{day}_tuning_curve_correct_v_incorrect_pcs_eg.svg'), bbox_inches='tight')