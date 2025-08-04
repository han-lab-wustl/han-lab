"""
compare led off vs. led on PER DAY
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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves_by_trialtype_w_darktime_early, make_tuning_curves
from projects.opto.behavior.behavior import smooth_lick_rate

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
#%%
# test rewzone
ii=40
iis=conddf[(conddf.animals=='e218') & (conddf.optoep>0)].index # control v inhib x ex
# iis=[126,166,49] # control v inhib x ex

datarasters=[]
cm_window=20
for ii in iis:
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   if animal=='e145': pln=2  
   else: pln=0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
   print(params_pth)

   fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'VR'])   
   VR = fall['VR'][0][0][()]
   scalingf = VR['scalingFACTOR'][0][0]
   try:
      rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
   except:
      rewsize = 10
   changeRewLoc = np.hstack(fall['changeRewLoc'])
   # set vars
   eps = np.where(changeRewLoc>0)[0]
   rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
   # only test opto vs. ctrl
   eptest = conddf.optoep.values[ii]
   if conddf.optoep.values[ii]<2: 
      eptest = random.randint(2,3)   
      if len(eps)<4: eptest = 2 # if no 3 epochs 
   eptest=int(eptest)   
   print(ii,rewlocs[eptest-2],rewlocs[eptest-1])

# initialize var
#%%
ii=40
iis=[60,172,49] # control v inhib x ex
# iis=[126,166,49] # control v inhib x ex

datarasters=[]
cm_window=20
for ii in iis:
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   if animal=='e145': pln=2  
   else: pln=0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
   print(params_pth)

   fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'putative_pcs',
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
   print(rewlocs[eptest-1])
   lasttr=8 # last trials
   bins=90
   rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
   track_length_rad = track_length*(2*np.pi/track_length)
   bin_size=track_length_rad/bins

   # takes time
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0]).astype(bool))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0]).astype(bool))]
   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   # if animal!='z14' and animal!='e200' and animal!='e189':      
   putative_pcs = np.array([xx[0] for xx in fall['putative_pcs'][0]])
   pcs = np.sum(putative_pcs,axis=0)>0          
   Fc3 = Fc3[:, ((skew>2))] # only keep cells with skew greater than 2
   # tc w/ dark time
   print('making tuning curves...\n')
   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8,velocity_filter=True) 
   bin_size=3
   tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,velocity_filter=True) # last 5 trials
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
   # also get lick and velocity:
   dt=np.nanmedian(np.diff(time))
   lick_rate=smooth_lick_rate(lick,dt)
   lick_tcs_correct_abs, lick_coms_correct_abs,lick_tcs_fail_abs, lick_coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,velocity_filter=True) # last 5 trials
   vel_tcs_correct_abs, vel_coms_correct_abs,vel_tcs_fail_abs, vel_coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,velocity_filter=True) # last 5 trials

   pcs = [xx for xx in pcs if xx not in goal_cells]  
   datarasters.append([tcs_correct_abs,coms_correct_abs,rewlocs, eptest,lick_tcs_correct_abs, vel_tcs_correct_abs] )

#%%
plt.rc('font', size=20)          # controls default text sizes
from matplotlib import colors
from matplotlib import patches
vmax=1.5
# Create subplots
fig, axes = plt.subplots(ncols=3,nrows=6, figsize=(12,15),height_ratios=[3,1,1,3,1,1],sharex=True)
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
   # return arr_new
   return arr_new # do not norm
lbls=['Control', 'VIP Inhibition', 'VIP Excitation']
colors=['grey','mediumspringgreen','lightcoral']
# Apply normalization before plotting
lickmax=8
velmax=150
from matplotlib.colors import LinearSegmentedColormap

# Example: Define the color points for Parula (simplified for illustration)
# The actual Parula colormap has more precise color definitions.
# You would typically get these from a source that provides Parula's RGB data.
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
[0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
[0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
[0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
[0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
[0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
[0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
[0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
[0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
[0.0589714286, 0.6837571429, 0.7253857143], 
[0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
[0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
[0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
[0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
[0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
[0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
[0.7184095238, 0.7411333333, 0.3904761905], 
[0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
[0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
[0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
[0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
[0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
[0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
[0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
[0.9763, 0.9831, 0.0538]]

parula_cmap = LinearSegmentedColormap.from_list('parula', cm_data)


for kk, lbl in enumerate(lbls):
   tcs_correct,coms_correct, rewlocs, eptest, lick_tcs_correct_abs, vel_tcs_correct_abs=datarasters[kk]
   # place
   # just get top 200?
   if lbl =='VIP Inhibition':
      tcs_correct = tcs_correct[:,:,:]
      coms_correct = coms_correct[:,:]
   tcs_normalized = tcs_correct[eptest-2][np.argsort(coms_correct[0])]
   # tcs_normalized = tcs_normalized[~np.all(tcs_normalized == 0, axis=1)]

   ax = axes[0,kk]
   # 1 = ledon
   im = ax.imshow(tcs_normalized, aspect='auto',vmin=0,vmax=vmax,cmap=parula_cmap)
   ax.set_title(f'{lbls[kk]} \n All spatially tuned cells')
   ax.axvline((rewlocs[eptest-2]-5) / 3, color='w', linestyle='--',linewidth=3)
   ax.set_yticks([0,len(tcs_normalized)-1])
   ax.set_yticks([1,len(tcs_normalized)])
   ax.set_xticks([0, bins // 2, bins]) 
   ax.set_xticklabels([0,135,270])
   if kk == 0:
      ax.set_ylabel('Cell # (sorted)')
      cbar_ax = fig.add_axes([.91, .65, 0.015, 0.2])  # [left, bottom, width, height]
      cbar = fig.colorbar(im, cax=cbar_ax)
      cbar.set_label(f'$\Delta$ F/F')
   # reward
   ax = axes[1,kk]
   # 1 = ledon
   ax.plot(lick_tcs_correct_abs[eptest-2][0],color='k')
   ax.axvline(rewlocs[eptest-2]/3,color='k', linestyle='--',linewidth=3)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   if kk==0: ax.set_ylabel('Lick rate')
   ax.set_ylim([0,lickmax])
   ax = axes[2,kk]
   # 1 = ledon
   ax.plot(vel_tcs_correct_abs[eptest-2][0],color='grey')
   ax.axvline(rewlocs[eptest-2]/3,color='k', linestyle='--',linewidth=3)
   ax.set_xlabel('Track position (cm)')
   ax.spines[['top','right']].set_visible(False)
   if kk==0: ax.set_ylabel('Velocity')
   ax.set_ylim([0,velmax])
   # opto epoch
   tcs_normalized = tcs_correct[eptest-1][np.argsort(coms_correct[0])]
   # tcs_normalized = tcs_normalized[~np.all(tcs_normalized == 0, axis=1)]

   ax = axes[3,kk]
   # 1 = ledon
   im = ax.imshow(tcs_normalized, aspect='auto',vmin=0,vmax=vmax,cmap=parula_cmap)
   ax.axvline((rewlocs[eptest-1]-5) / 3, color='w', linestyle='--',linewidth=3)
   # Add red patch above raster
   pre_rew_x = 0
   pre_rew_width = (rewlocs[eptest-1]) / 3  # convert to bins
   pre_rew_y = -2  # just above the top row (adjust as needed)
   patch_height = 10  # height of the patch (adjust as needed)
   red_patch = patches.Rectangle(
      (pre_rew_x, pre_rew_y),      # (x, y)
      width=pre_rew_width,
      height=patch_height,
      color='red',
      alpha=0.5
   )
   ax.add_patch(red_patch)
   ax.set_yticks([0,len(tcs_normalized)-1])
   ax.set_yticks([1,len(tcs_normalized)])
   ax.set_xticks([0, bins // 2, bins]) 
   ax.set_xticklabels([0,135,270])
   if kk == 0:
      ax.set_ylabel('Cell # (sorted by previous epoch)')
      cbar_ax = fig.add_axes([.91, .65, 0.015, 0.2])  # [left, bottom, width, height]
      cbar = fig.colorbar(im, cax=cbar_ax)
      cbar.set_label(f'$\Delta$ F/F')
   # reward
   ax = axes[4,kk]
   # 1 = ledon
   ax.plot(lick_tcs_correct_abs[eptest-1][0],color='k')
   ax.axvline(rewlocs[eptest-1]/3,color='k', linestyle='--',linewidth=3)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   if kk==0: ax.set_ylabel('Lick rate')
   ax.set_ylim([0,lickmax])
   ax = axes[5,kk]
   # 1 = ledon
   ax.plot(vel_tcs_correct_abs[eptest-1][0],color='grey')
   ax.axvline(rewlocs[eptest-1]/3,color='k', linestyle='--',linewidth=3)
   ax.set_xlabel('Track position (cm)')
   ax.spines[['top','right']].set_visible(False)
   if kk==0: ax.set_ylabel('Velocity')
   ax.set_ylim([0,velmax])
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
plt.savefig(os.path.join(savedst, f'supp_fig5_eg_tuning_curves.svg'), bbox_inches='tight')
