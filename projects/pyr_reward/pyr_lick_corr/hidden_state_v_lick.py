"""
bad fit
"""
#%%
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# ------------------------------
# Simulated Data (replace with real traces)
# ------------------------------
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_time_trial_by_trial
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones, cosine_sim_ignore_nan
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.stats import spearmanr
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'

cm_window=20 # to search for rew cells
ii=0
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if (animal!='e217') & (conddf.optoep.values[ii]<2):
   if animal=='e145' or animal=='e139': pln=2 
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
   # get average success rate
   rates = []
   for ep in range(len(eps)-1):
      eprng = range(eps[ep],eps[ep+1])
      success, fail, str_trials, ftr_trials, ttr, \
      total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
      rates.append(success/total_trials)
   rate=np.nanmean(np.array(rates))
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
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
      rewsize,ybinned,time,lick,
      Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt)  
   bin_size=3
   # abs position
   tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)
   # get cells that maintain their coms across at least 2 epochs
   place_window = 20 # cm converted to rad                
   perm = list(combinations(range(len(coms_correct_abs)), 2))     
   com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
   compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
   # get cells across all epochs that meet crit
   pcs = np.unique(np.concatenate(compc))
   pcs_all = intersect_arrays(*compc)

   lick_correct_abs, _,lick_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick,lick]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
   vel_correct_abs, _,vel_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel,forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
   goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
   # change to relative value 
   coms_rewrel = np.array([com-np.pi for com in coms_correct])
   perm = list(combinations(range(len(coms_correct)), 2)) 
   rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
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
   #only get perms with non zero cells
   perm=[p for ii,p in enumerate(perm) if len(com_goal[ii])>0]
   rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal[ii])>0]
   com_goal=[com for com in com_goal if len(com)>0]
   ######################## near pre reward only
   bound=np.pi/4
   # com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
   # xx], axis=0)<0)&(np.nanmedian(coms_rewrel[:,
   # xx], axis=0)>-bound))] if len(com)>0 else [] for com in com_goal]
   com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
   xx], axis=0)>0)&(np.nanmedian(coms_rewrel[:,
   xx], axis=0)>-bound))] if len(com)>0 else [] for com in com_goal]
   # get goal cells across all epochs        
   if len(com_goal_postrew)>0:
      goal_cells = intersect_arrays(*com_goal_postrew); 
   else:
      goal_cells=[]
   goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)     
#%%
# Simulated features (replace with real aligned data)
calcium = np.nanmean(Fc3[:,goal_cells],axis=1)
t = time
dt = np.nanmedian(np.diff(t))
# manually check 0.7 for smoothing
#correct vs. incorre
calcium_pre = []
lick_rate=[]
vel=[]
rew=[]
pos=[]
for ep in range(len(eps)-1):
   eprng = np.arange(eps[ep],eps[ep+1])
   eprng=[xx for xx in eprng if trialnum[xx]>2]
   ## get pre-reward lick rate
   t = time[eprng][(ybinned[eprng]<rewlocs[ep])]
   dt = np.nanmedian(np.diff(t))
   corr_lr = smooth_lick_rate(lick[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))], dt)       
   lick_rate.append(corr_lr)
   calcium_pre.append(calcium[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))])
   vel.append(forwardvel[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))])
   rew.append(rewards[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))])
   # pos.append(ybinned[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))]-rewlocs[ep])
   pos.append(ybinned[eprng][(ybinned[eprng]<(rewlocs[ep]-(rewsize)))])

#%%
calcium_pre =np.concatenate(calcium_pre)
lick_rate=np.concatenate(lick_rate)
vel=np.concatenate(vel)
rew=np.concatenate(rew)
pos=np.concatenate(pos)
#%%
# Prepare observations: calcium activity (n_samples x 1)
# Stack calcium and lick rate as 2D observations (n_samples, 2)
observations = np.column_stack([calcium_pre, lick_rate])

# Fit HMM with multivariate Gaussian emissions
n_states = 3
model = hmm.GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
model.fit(observations)

# Predict hidden states
hidden_states = model.predict(observations)

# Plot results
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(calcium_pre, label='Calcium')
ax[0].legend()
ax[0].set_ylabel('Calcium')

ax[1].plot(lick_rate, label='Lick Rate', color='orange')
ax[1].legend()
ax[1].set_ylabel('Lick Rate')

ax[2].plot(hidden_states, lw=2)
ax[2].set_ylabel('Hidden State')
ax[2].set_xlabel('Time')
ax[2].set_yticks(np.arange(n_states))
ax[2].set_yticklabels([f'State {i}' for i in range(n_states)])

plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

# Prepare data
y = calcium_pre  # neural activity

# Model A: lick rate only
X_lick = lick_rate.reshape(-1, 1)
model_lick = LinearRegression().fit(X_lick, y)
y_pred_lick = model_lick.predict(X_lick)
r2_lick = r2_score(y, y_pred_lick)

# Model B: hidden state only (one-hot encode states)
hidden_states_reshaped = (hidden_states).reshape(-1, 1).astype(int)
X_state = encoder.fit_transform(hidden_states_reshaped)

model_state = LinearRegression().fit(X_state, y)
y_pred_state = model_state.predict(X_state)
r2_state = r2_score(y, y_pred_state)

# Model C: lick rate + hidden states
X_combined = np.hstack([X_lick, X_state])
model_combined = LinearRegression().fit(X_combined, y)
y_pred_combined = model_combined.predict(X_combined)
r2_combined = r2_score(y, y_pred_combined)

print(f"R² lick rate only: {r2_lick:.3f}")
print(f"R² hidden state only: {r2_state:.3f}")
print(f"R² combined: {r2_combined:.3f}")
