
"""
zahra
april 2025
get place cells and plot com histogram
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'true_pc.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\tuning_curves_pcs_nopto.p"
# with open(saveddataset, "rb") as fp: #unpickle
#         radian_alignment_saved = pickle.load(fp)
# initialize var
#%%
dataraster = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
num_iterations=1000
place_cell_null=[]
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') and (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
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
      if animal=='e145':
               ybinned=ybinned[:-1]
               forwardvel=forwardvel[:-1]
               changeRewLoc=changeRewLoc[:-1]
               trialnum=trialnum[:-1]
               rewards=rewards[:-1]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
      eps = np.append(eps, len(changeRewLoc))     
      if len(eps)>3:   
         print(params_pth)
         lasttr=8 # last trials
         bins=90
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
         Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
         # if no cells pass these crit
         if Fc3.shape[1]==0:
                  Fc3 = fall_fc3['Fc3']
                  Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                  pc_bool = np.sum(pcs,axis=0)>=1
                  Fc3 = Fc3[:,((skew>2)&pc_bool)]
         bin_size=3 # cm
         # get abs dist tuning 
         tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
         Fc3,trialnum,rewards,forwardvel,
         rewsize,bin_size)
         # get cells that maintain their coms across at least 2 epochs
         place_window = 20 # cm converted to rad                
         perm = list(combinations(range(len(coms_correct_abs)), 2))     
         com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
         compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
         # get cells across all epochs that meet crit
         pcs = np.unique(np.concatenate(compc))
         pcs_all = intersect_arrays(*compc)
         pcs_ep12 = intersect_arrays(*compc[:1])
         dropped = [xx for xx in pcs_ep12 if xx not in pcs_all]
         dataraster[f'{animal}_{day}']=[tcs_correct_abs[:,pcs_all], tcs_correct_abs[:,dropped]]

#%%
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt

animals = ['e218', 'e216', 'e201', 'e186', 'e190',
       'z8', 'z9', 'z16']
plt.rc('font', size=16) 
for an in animals:
   # Stack and drop rows with NaNs
   # first epoch of both types
   dedicated_place = np.vstack([v[0][0,:,:] for k, v in dataraster.items() if an in k])
   dedicated_place = np.vstack([v[0][0,:,:] for k, v in dataraster.items() if an in k])
   dedicated_place_ep3 = np.vstack([v[0][-1,:,:] for k, v in dataraster.items() if an in k])
   dropped_place = np.vstack([v[1][0,:,:] for k, v in dataraster.items() if an in k])
   dropped_place_ep = np.vstack([v[1][-1,:,:] for k, v in dataraster.items() if an in k])
   dedicated_place[np.isnan(dedicated_place).any(axis=1)]=0
   dropped_place[np.isnan(dropped_place).any(axis=1)]=0
   dropped_place_ep[np.isnan(dropped_place_ep).any(axis=1)]=0
   dedicated_place_ep3[np.isnan(dedicated_place_ep3).any(axis=1)]=0
   # or drop nans
   # dedicated_place = dedicated_place[~np.isnan(dedicated_place).any(axis=1)]
   # dedicated_place_ep3 = dedicated_place_ep3[~np.isnan(dedicated_place_ep3).any(axis=1)]
   # dropped_place = dropped_place[~np.isnan(dropped_place).any(axis=1)]
   # dropped_place_ep = dropped_place_ep[~np.isnan(dropped_place_ep).any(axis=1)]

   # Normalize each row to 0–1
   # dedicated_place = minmax_scale(dedicated_place, axis=1)
   # dropped_place = minmax_scale(dropped_place, axis=1)
   # Sort each by peak (argmax of each row)
   dedicated_sort_idx = np.argmax(dedicated_place, axis=1).argsort()
   dedicated_sort_idx_ep3 = np.argmax(dedicated_place, axis=1).argsort()
   dropped_sort_idx = np.argmax(dropped_place, axis=1).argsort()
   dropped_ep_sort_idx = np.argmax(dropped_place, axis=1).argsort()

   dedicated_place_ep3 = dedicated_place_ep3[dedicated_sort_idx_ep3]
   dedicated_place = dedicated_place[dedicated_sort_idx]
   dropped_place = dropped_place[dropped_sort_idx]
   dropped_place_ep = dropped_place_ep[dropped_ep_sort_idx]
   # Gamma normalization (e.g., gamma=0.5 for square root)
   gamma = 0.5
   import matplotlib.colors as mcolors
   norm = mcolors.PowerNorm(gamma=gamma)

   fig,axes=plt.subplots(ncols=4,nrows=2,sharex=True,figsize=(10,5))
   # Top row: imshow with gamma + colorbar
   fs=12
   img0 = axes[0, 0].imshow(dedicated_place, aspect='auto', norm=norm, cmap='viridis')
   axes[0, 0].set_title("Dedicated Epoch 1",fontsize=fs)
   fig.colorbar(img0, ax=axes[0, 0], orientation='vertical')
   axes[0,0].set_ylabel('Place cell #')

   img0 = axes[0, 1].imshow(dedicated_place_ep3, aspect='auto', norm=norm, cmap='viridis')
   axes[0, 1].set_title("Dedicated Epoch 3\nSorted by Epoch 1",fontsize=fs)
   fig.colorbar(img0, ax=axes[0, 1], orientation='vertical')

   img1 = axes[0, 2].imshow(dropped_place, aspect='auto', norm=norm, cmap='viridis')
   axes[0,2].set_title("Non-dedicated Epoch 1",fontsize=fs)
   fig.colorbar(img1, ax=axes[0, 2], orientation='vertical')

   img2 = axes[0, 3].imshow(dropped_place_ep, aspect='auto', norm=norm, cmap='viridis')
   axes[0, 3].set_title("Non-dedicated Epoch 3\nSorted by Epoch 1",fontsize=fs)
   fig.colorbar(img2, ax=axes[0, 3], orientation='vertical')
   dedicated_place = np.vstack([v[0][0,:,:] for k, v in dataraster.items() if an in k])
   dedicated_place_ep3 = np.vstack([v[0][-1,:,:] for k, v in dataraster.items() if an in k])
   dropped_place = np.vstack([v[1][0,:,:] for k, v in dataraster.items() if an in k])
   dropped_place_ep = np.vstack([v[1][-1,:,:] for k, v in dataraster.items() if an in k])
   dedicated_place[np.isnan(dedicated_place).any(axis=1)]=0
   dropped_place[np.isnan(dropped_place).any(axis=1)]=0
   dropped_place_ep[np.isnan(dropped_place_ep).any(axis=1)]=0
   dedicated_place_ep3[np.isnan(dedicated_place_ep3).any(axis=1)]=0

   vmin = 0
   vmax = 0.6
   if an == 'z16':
      vmax = 1

   # Calculate mean and SEM
   def plot_mean_with_sem(ax, data, color='k', label=''):
      mean = np.nanmean(data, axis=0)
      error = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
      x = np.arange(data.shape[1])
      ax.plot(x, mean, color=color, label=label)
      ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.3)
      ax.set_ylim([vmin, vmax])

   # Dedicated
   col='indigo'
   ax = axes[1, 0]
   plot_mean_with_sem(ax, dedicated_place, color=col, label='Dedicated')
   ax.set_title("Mean $\pm$ SEM")
   ax.set_xticks([0,90])
   ax.set_xticklabels([0,270])
   ax.spines[['top', 'right']].set_visible(False)
   ax = axes[1, 1]
   plot_mean_with_sem(ax, dedicated_place_ep3, color=col, label='Dedicated')
   ax.set_xticks([0,90])
   ax.spines[['top', 'right']].set_visible(False)

   # Dropped
   ax = axes[1, 2]
   plot_mean_with_sem(ax, dropped_place, color=col, label='Dropped')
   ax.set_xticks([0,90])
   ax.set_yticklabels([])
   ax.spines[['top', 'right']].set_visible(False)
   # Dropped EP
   ax = axes[1, 3]
   plot_mean_with_sem(ax, dropped_place_ep, color=col, label='Dropped EP')
   ax.set_xticks([0,90])
   ax.set_yticklabels([])
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_xlabel('Track positon (cm)')

   fig.suptitle(an)
   # ax.imshow(dropped_place_ep, aspect='auto',vmin=0,vmax=5)