#%%
"""
"Using a hidden Markov model trained on control trials, we identified a latent neural state that consistently emerged prior to reward-directed licking. This state was significantly reduced in occupancy during optogenetic inhibition of VIP interneurons, suggesting that the perturbation disrupts a neural state critical for initiating reward-directed behavior."

"""
import numpy as np, sys
import scipy.io, scipy.interpolate, scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
import random
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import smooth_lick_rate
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from itertools import combinations
#%%
# ============================== #
# CONFIGURATION
# ============================== #
bins = 90
goal_window_cm = 20
conddf=pd.read_csv(r'Z:\condition_df\conddf_performance_chrimson.csv')
iis = np.arange(len(conddf))  # Animal indices
iis = [ii for ii in iis if ii!=202]
dfs = {}
iis=np.array(iis)
# iis=iis[iis>167]

# ============================== #
# MAIN LOOP OVER ANIMALS
# ============================== #
#%%
# iis=iis[iis>183]
pvalues=[]
for ii in iis:
   # ---------- Load animal info ---------- #
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   in_type = conddf.in_type.values[ii]
   plane = 2 if animal in ['e145', 'e139'] else 0
   params_pth = f"Y:/analysis/fmats/{animal}/days/{animal}_day{day:03d}_plane{plane}_Fall.mat"
   print(params_pth)

   # ---------- Load required variables ---------- #
   keys = ['coms', 'changeRewLoc', 'ybinned', 'VR', 
         'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
         'timedFF', 'stat', 'licks']
   fall = scipy.io.loadmat(params_pth, variable_names=keys)
   VR = fall['VR'][0][0]
   scalingf = VR['scalingFACTOR'][0][0]
   rewsize = VR['settings']['rewardZone'][0][0][0][0] / scalingf if 'rewardZone' in VR['settings'].dtype.names else 10
   track_length=180/scalingf    
   # ---------- Preprocess variables ---------- #
   ybinned = fall['ybinned'][0] / scalingf
   forwardvel = fall['forwardvel'][0]
   trialnum = fall['trialnum'][0]
   rewards = fall['rewards'][0]
   licks = fall['licks'][0]
   time = fall['timedFF'][0]
   if animal == 'e145':  # Trim 1 sample
      trim_len = len(ybinned) - 1
      ybinned = ybinned[:trim_len]
      forwardvel = forwardvel[:trim_len]
      trialnum = trialnum[:trim_len]
      rewards = rewards[:trim_len]
      licks = licks[:trim_len]
      time = time[:trim_len]

   # ---------- Define epochs ---------- #
   changeRewLoc = np.hstack(fall['changeRewLoc'])
   eps = np.where(changeRewLoc > 0)[0]
   rewlocs = changeRewLoc[eps] / scalingf
   eps = np.append(eps, len(changeRewLoc))
   dt=np.nanmedian(np.diff(time))
   # Pick training/testing epochs
   optoep = conddf.optoep.values[ii]
   eptest = optoep if optoep >= 2 else random.randint(2, 3)
   if len(eps) < 4: eptest = 2
   ep_train = eptest - 2

   # ---------- Load fluorescence data ---------- #
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   if in_type=='vip' or animal=='z17':
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   else:
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   skewthres=1.2
   Fc3 = Fc3[:, skew>skewthres] # only keep cells with skew greateer than 2
   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   # tcs
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
   bins=bins_dt,lasttr=8)  
   # far pre
   goal_window = 20*(2*np.pi/track_length) # cm converted to rad
   # change to relative value 
   coms_rewrel = np.array([com-np.pi for com in coms_correct])
   perm = list(combinations(range(len(coms_correct)), 2)) 
   # if 4 ep
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
   bound = 0
   com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
      xx], axis=0)<=bound))] if len(com)>0 else [] for com in com_goal]
   perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
   com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
   print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
   # get goal cells across all epochs        
   if len(com_goal_farrew)>0:
      goal_cells = np.unique(np.concatenate(com_goal_farrew)) 
   else:
      goal_cells=[]

   Xs=[]; Ys=[]
   # per epoch
   for ep in range(len(eps)-1):
      eprng = np.arange(eps[ep],eps[ep+1])
      pos = ybinned[eprng]
      trnm = trialnum[eprng]
      success, fail, str_trials, ftr_trials, ttr, total_trials = get_success_failure_trials(trnm, rewards[eprng])
      # trialstate=np.zeros_like(np.unique(trnm))
      # strind=[np.where(ttr==xx)[0][0] for xx in str_trials]
      # trialstate[strind]=1
      early_window = (pos >= 0) & (pos < 67)
      df_trials = []
      trialstate=[]
      for nm,tr in enumerate(np.unique(trnm)):
         mask = (trnm == tr) & early_window
         if mask.sum() == 0:
            continue
         avg_activity = Fc3[eprng][mask][:,:].mean(axis=0)
         df_trials.append(avg_activity)
         if tr in str_trials:
            trialstate.append(1)
         else:
            trialstate.append(0)
      X = np.vstack(df_trials)  # [n_trials x n_neurons]
      y = np.array(trialstate)  # or y = np.array(correct_trials)
      Xs.append(X)
      Ys.append(y)
   X= np.vstack(Xs)
   X = (X - X.mean(axis=0)) / X.std(axis=0)
   X[np.isnan(X)]=0
   y= np.hstack(Ys)#np.array([np.hstack(Ys)]*X.shape[1]).T
   from sklearn.naive_bayes import GaussianNB
   from sklearn.model_selection import cross_val_score
   clf = GaussianNB()
   scores = cross_val_score(clf, X, y, cv=5)
   print("Cross-validated accuracy: %.3f ± %.3f" % (scores.mean(), scores.std()))
   clf.fit(X, y)
   n_perms = 1000
   real_accuracy=scores.mean()
   shuffled_accs = []
   for _ in range(n_perms):
      y_shuff = np.random.permutation(y)
      shuffled_accs.append(clf.score(X, y_shuff))
   pval = np.mean(np.array(shuffled_accs) >= real_accuracy)
   print(f"p = {pval:.4f}")
   
   # means_0 = np.nanmean(X[y == 0],axis=0)
   # means_1 = np.nanmean(X[y == 1],axis=0)
   # stds = np.nanstd(X,axis=0)

   # effect_size = np.abs(means_1 - means_0) / stds  # Cohen's d

   # # Get top contributing features (cells)
   # top_cells = np.argsort(effect_size)[::-1][:10]

   # # Plot
   # plt.figure(figsize=(6,4))
   # plt.bar(np.arange(len(top_cells)), effect_size[top_cells])
   # plt.xticks(np.arange(len(top_cells)), top_cells)
   # plt.xlabel('Cell Index')
   # plt.ylabel('Effect Size (|d|)')
   # plt.title('Top Predictive Cells')
   # plt.tight_layout()
   # plt.show()
   pvalues.append(pval)