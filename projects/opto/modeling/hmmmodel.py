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
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime

def compute_state_metrics(states, licks, vel, relpos, sr, dff=None):
   """
   states: list of [n_timepoints] arrays per trial
   licks: list of [n_timepoints] arrays per trial
   vel: list of [n_timepoints] arrays per trial
   dff: list of [n_cells x n_timepoints] arrays per trial (optional)
   """
   results = []

   for s, lk, v, relp, df in zip(states, licks, vel, relpos, dff if dff is not None else [None]*len(states)):
      for state in np.unique(s):
         inds = s == state

         out = {
               'State': state,
               'Time_in_state': np.sum(inds) / sr,
               'Mean_lick_rate': np.nanmean(lk[inds]),
               'Mean_velocity': np.nanmean(v[inds]),
               'Mean rel. pos.': np.nanmean(relp[inds])
         }

         if df is not None:
               # df is shape [n_cells x time], take mean across cells and timepoints in state
               out['Mean_dff'] = np.nanmean(df[:, inds])

         results.append(out)

   return pd.DataFrame(results)


def interpolate_behavior_like_dff(epoch_inds, trialnum, licks, velocity):

   """
   Interpolate lick and velocity signals per trial to match neural data interpolation.

   Parameters:
   - epoch_inds: array of indices for the current epoch (e.g. epr_train or epr_test)
   - trialnum: full trial number array (same size as full time series)
   - licks: full lick array (same size as trialnum)
   - velocity: full velocity array (same size as trialnum)

   Returns:
   - lick_trials: array (n_trials, max_len)
   - vel_trials:  array (n_trials, max_len)
   """
   trials_in_epoch = trialnum[epoch_inds]
   licks_epoch = licks[epoch_inds]
   vel_epoch = velocity[epoch_inds]
   unique_trials = np.unique(trials_in_epoch)

   # Get trialwise data
   lick_per_trial = [licks_epoch[trials_in_epoch == tr] for tr in unique_trials]
   vel_per_trial = [vel_epoch[trials_in_epoch == tr] for tr in unique_trials]
   max_len = max(len(tr) for tr in lick_per_trial)
   lick_trials = np.full((len(unique_trials), max_len), np.nan)
   vel_trials = np.full((len(unique_trials), max_len), np.nan)

   for i, (lick_tr, vel_tr) in enumerate(zip(lick_per_trial, vel_per_trial)):
      orig_t = np.linspace(0, 1, len(lick_tr))
      new_t = np.linspace(0, 1, max_len)
      if len(lick_tr) > 1:
         interp = scipy.interpolate.interp1d(orig_t, lick_tr, bounds_error=False, fill_value='extrapolate')
         lick_trials[i] = interp(new_t)
         interp = scipy.interpolate.interp1d(orig_t, vel_tr, bounds_error=False, fill_value='extrapolate')
         vel_trials[i] = interp(new_t)

   return lick_trials, vel_trials

#%%
# ============================== #
# CONFIGURATION
# ============================== #
bins = 90
goal_window_cm = 20
iis = np.arange(len(conddf))  # Animal indices
iis = [ii for ii in iis if ii!=202]
n_states = 4
dfs = {}
iis=np.array(iis)
# iis=iis[iis>167]

# ============================== #
# MAIN LOOP OVER ANIMALS
# ============================== #
#%%
# iis=iis[iis>183]
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
   iscell = fall['iscell'][:, 0].astype(bool)
   border = ~fall['bordercells'][0].astype(bool)
   cell_mask = iscell & border if in_type == 'vip' or animal == 'z17' else iscell
   Fc3 = Fc3[:, cell_mask]
   dFF = dFF[:, cell_mask]
   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   # tcs
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
   bins=bins_dt,lasttr=8)  
   licks = smooth_lick_rate(licks,dt)

   # ---------- Prepare training data ---------- #
   # 8/1/25 - without dark time
   epr_train = np.arange(eps[ep_train], eps[ep_train + 1])
   f_train = Fc3[epr_train, :]
   trials_train = trialnum[epr_train]
   trial_ids = np.unique(trials_train)
   f__train = [f_train[trials_train == tr] for tr in trial_ids]
   max_len = max(len(fr) for fr in f__train)
   dff_train = np.full((len(f__train), f_train.shape[1], max_len), np.nan)

   for i, tr in enumerate(trial_ids):
      f_tr = f_train[trials_train == tr].T
      orig_t = np.linspace(0, 1, f_tr.shape[1])
      new_t = np.linspace(0, 1, max_len)
      for c in range(f_tr.shape[0]):
         if np.sum(~np.isnan(f_tr[c])) > 1:
               interp = scipy.interpolate.interp1d(orig_t, f_tr[c], bounds_error=False, fill_value='extrapolate')
               dff_train[i, c] = interp(new_t)

   # Flatten to 2D and train HMM
   X_train = dff_train.transpose(0, 2, 1).reshape(-1, dff_train.shape[1])
   X_train[np.isnan(X_train)] = 0
   model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=500)
   model.fit(X_train)

   # ---------- Decode held-out trials ---------- #
   heldout = np.random.choice(trial_ids, size=len(trial_ids) // 3, replace=False)
   dff_test_within = dff_train[[i for i, tr in enumerate(trial_ids) if tr in heldout]]
   decoded_within = np.array([model.predict(np.nan_to_num(tr.T)) for tr in dff_test_within])

   # ---------- Decode opto trials ---------- #
   epr_test = np.arange(eps[eptest - 1], eps[eptest])
   f_test = Fc3[epr_test, :]
   trials_test = trialnum[epr_test]
   f__test = [f_test[trials_test == tr] for tr in np.unique(trials_test)]
   max_len = max(len(fr) for fr in f__test)
   dff_test_opto = np.full((len(f__test), f_test.shape[1], max_len), np.nan)

   for i, tr in enumerate(np.unique(trials_test)):
      f_tr = f_test[trials_test == tr].T
      orig_t = np.linspace(0, 1, f_tr.shape[1])
      new_t = np.linspace(0, 1, max_len)
      for c in range(f_tr.shape[0]):
         if np.sum(~np.isnan(f_tr[c])) > 1:
               interp = scipy.interpolate.interp1d(orig_t, f_tr[c], bounds_error=False, fill_value='extrapolate')
               dff_test_opto[i, c] = interp(new_t)

   decoded_opto = np.array([model.predict(np.nan_to_num(trial.T)) for trial in dff_test_opto])
   # ---------- Align to licks & velocity ---------- #
   # You must define this utility yourself
   # Reward-relative position for all trials
   
   lick_train, vel_train = interpolate_behavior_like_dff(epr_train, trialnum, licks, forwardvel)
   relpos_train, _ = interpolate_behavior_like_dff(epr_train, trialnum, np.concatenate(rad), forwardvel)
   # heldout trials
   lick_test = np.array([xx for tr, xx in enumerate(lick_train) if trial_ids[tr] in heldout])
   vel_test = np.array([xx for tr, xx in enumerate(vel_train) if trial_ids[tr] in heldout])
   relpos_test = np.array([xx for tr, xx in enumerate(relpos_train) if trial_ids[tr] in heldout])
   lick_opto, vel_opto = interpolate_behavior_like_dff(epr_test, trialnum, licks, forwardvel)
   relpos_opto, _ = interpolate_behavior_like_dff(epr_test, trialnum, np.concatenate(rad), forwardvel)

   # test
   # fig,ax=plt.subplots(nrows=3)
   # ax[0].imshow(decoded_within,aspect='auto')
   # ax[1].imshow(lick_test,aspect='auto')
   # ax[2].imshow(vel_test,aspect='auto')
   trial = np.random.randint(len(heldout))
   fig,ax=plt.subplots(nrows=2)
   dff_trial = dff_test_within[trial]
   # Normalize each cell (z-score across time)
   dff_norm = (dff_trial - np.nanmean(dff_trial, axis=1, keepdims=True)) / np.nanstd(dff_trial, axis=1, keepdims=True)
   # Get peak time index for sorting (ignoring NaNs)
   dff_norm[np.isnan(dff_norm)]=0
   peak_times = np.nanargmax(dff_norm, axis=1)
   sort_order = np.argsort(peak_times)
   ax[0].imshow(dff_norm[sort_order],aspect='auto')
   ax[1].plot(lick_test[trial])   
   ax[1].plot(vel_test[trial]/np.nanmax(vel_test[trial]))   
   ax[1].plot(decoded_within[trial])  
   ax[1].plot(relpos_test[trial])  
   # ---------- Metrics: Time in state, lick rate, velocity ---------- #
   dt = np.nanmedian(np.diff(time))
   sr = 1 / dt
   df_within = compute_state_metrics(decoded_within, lick_test, vel_test,relpos_test, sr, dff=dff_test_within)
   df_opto = compute_state_metrics(decoded_opto, lick_opto, vel_opto,relpos_opto, sr, dff=dff_test_opto)

   df_within['Condition'] = 'Within'
   df_opto['Condition'] = 'Opto'
   df_all = pd.concat([df_within, df_opto]).reset_index() 
   df_all['animals']=animal
   df_all['days']=day   
   # ---------- Plot ---------- #
   fig,axes = plt.subplots(ncols=5,figsize=(20,5),sharex=True)
   for m,metric in enumerate(['Mean_lick_rate', 'Mean_velocity','Mean rel. pos.', 'Mean_dff','Time_in_state']):      
      ax=axes[m]
      sns.pointplot(data=df_all, x='State', y=metric, hue='Condition', dodge=True, palette='Set2',ax=ax)
      ax.set_title(f'{metric} by State\n{animal}, Day {day}, Opto {optoep}')
      ax.set_ylabel(metric.replace('_', ' '))
   plt.tight_layout()
   plt.show()
   dfs[f'{animal}_{day}'] = df_all

#%%
# analyze
bigdf = pd.concat([v for k,v in dfs.items()])
bigdf=bigdf[(bigdf.animals!='e189') & (bigdf.animals!='e190')]

# remove outlier days
bigdf=bigdf[~((bigdf.animals=='z14')&((bigdf.days<33)))]
bigdf=bigdf[~((bigdf.animals=='z16')&((bigdf.days>13)))]
# bigdf=bigdf[~((bigdf.animals=='z17')&((bigdf.days.isin([3,11]))))]
bigdf=bigdf[~((bigdf.animals=='z15')&((bigdf.days.isin([7,12,16]))))]
bigdf=bigdf[~((bigdf.animals=='e217')&((bigdf.days<9)|(bigdf.days.isin([21,29,30,26,29]))))]
bigdf=bigdf[~((bigdf.animals=='e216')&((bigdf.days<32)))]
bigdf=bigdf[~((bigdf.animals=='e200')&((bigdf.days.isin([67,68,81]))))]


bigdf_merged = pd.merge(bigdf, conddf, on=['animals', 'days'], how='left')
bigdf_merged=bigdf_merged[bigdf_merged.optoep>1]
bigdf_merged=bigdf_merged.groupby(['animals','days','in_type', 'Condition', 'State']).mean(numeric_only=True).reset_index()

# Make a copy to avoid modifying original
bigdf_reordered = bigdf_merged.copy()
# Create a new column to store reordered states
bigdf_reordered['State_reordered'] = np.nan

# Get unique animal-day pairs
pairs = bigdf_reordered[['animals', 'days']].drop_duplicates()

# Loop through each (animal, day) pair
for _, row in pairs.iterrows():
   an, dy = row['animals'], row['days']
   
   mask = (bigdf_reordered['animals'] == an) & (bigdf_reordered['days'] == dy)
   df_sub = bigdf_reordered[mask]
   
   # Skip if not enough states
   if df_sub['State'].nunique() < 2:
      continue
   # Compute reordering
   state_order = df_sub.groupby('State')['Mean rel. pos.'].mean().sort_values().index.tolist()
   state_remap = {orig: new for new, orig in enumerate(state_order)}
   # Apply
   bigdf_reordered.loc[mask, 'State_reordered'] = df_sub['State'].map(state_remap)

# Make sure the column is integer after assignment
bigdf_reordered['State_reordered'] = bigdf_reordered['State_reordered'].astype('Int64')
sns.barplot(y='Mean rel. pos.', x='State_reordered', data=bigdf_reordered)
plt.figure()
sns.barplot(y='Mean_lick_rate', x='State_reordered', data=bigdf_reordered)
plt.figure()
sns.barplot(y='Mean_velocity', x='State_reordered', data=bigdf_reordered)
plt.figure()
sns.barplot(y='Time_in_state', x='State_reordered', data=bigdf_reordered)
#%%
# vip in
fig,axes=plt.subplots(ncols=3,nrows=2,figsize=(8,5),sharex=True,sharey='row')
ax=axes[0,0]
sns.barplot(x='State_reordered',y='Mean rel. pos.',data=bigdf_reordered[bigdf_reordered.in_type=='ctrl'],hue='Condition',ax=ax)
ax=axes[0,1]
sns.barplot(x='State_reordered',y='Mean rel. pos.',data=bigdf_reordered[bigdf_reordered.in_type=='vip'],hue='Condition',ax=ax)
ax=axes[0,2]
sns.barplot(x='State_reordered',y='Mean rel. pos.',data=bigdf_reordered[bigdf_reordered.in_type=='vip_ex'],hue='Condition',ax=ax)

ax=axes[1,0]
sns.barplot(x='State_reordered',y='Time_in_state',data=bigdf_reordered[bigdf_reordered.in_type=='ctrl'],hue='Condition',ax=ax)
ax=axes[1,1]
sns.barplot(x='State_reordered',y='Time_in_state',data=bigdf_reordered[bigdf_reordered.in_type=='vip'],hue='Condition',ax=ax)
ax=axes[1,2]
sns.barplot(x='State_reordered',y='Time_in_state',data=bigdf_reordered[bigdf_reordered.in_type=='vip_ex'],hue='Condition',ax=ax)
#%%
# Add Lick Rate and Velocity to aggregation
grouped = bigdf_reordered.groupby(
    ['animals', 'days', 'in_type', 'State_reordered', 'Condition']
).agg({
    'Mean rel. pos.': 'mean',
    'Time_in_state': 'mean',
    'Mean_lick_rate': 'mean',
    'Mean_velocity': 'mean'
}).reset_index()

# Pivot to compare Within vs. Opto for each variable
pivot = grouped.pivot_table(
    index=['animals', 'days', 'in_type', 'State_reordered'],
    columns='Condition',
    values=['Mean rel. pos.', 'Time_in_state', 'Mean_lick_rate', 'Mean_velocity']
)

# Compute Δ (Opto - Within)
delta = pd.DataFrame({
    'Δ Mean rel. pos.': pivot[('Mean rel. pos.', 'Opto')] - pivot[('Mean rel. pos.', 'Within')],
    'Δ Time in state': pivot[('Time_in_state', 'Opto')] - pivot[('Time_in_state', 'Within')],
    'Δ Lick rate': pivot[('Mean_lick_rate', 'Opto')] - pivot[('Mean_lick_rate', 'Within')],
    'Δ Velocity': pivot[('Mean_velocity', 'Opto')] - pivot[('Mean_velocity', 'Within')]
}).reset_index()

delta=delta[delta['Δ Time in state']<100]
# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10), sharex=True, sharey='row')
types = ['ctrl', 'vip', 'vip_ex']
metrics = ['Δ Mean rel. pos.', 'Δ Time in state', 'Δ Lick rate', 'Δ Velocity']
ylabels = ['Δ Mean rel. pos.', 'Δ Time in state', 'Δ Lick rate', 'Δ Velocity']
pval_dict={}
from scipy.stats import ttest_1samp

# Step 1: Perform t-tests and collect p-values
pval_dict = {}  # (in_type, State_reordered, metric) → p-value

metrics_test = ['Δ Time in state', 'Δ Lick rate', 'Δ Velocity', 'Δ Mean rel. pos.']
for t in types:
   sub = delta[delta['in_type'] == t]
   for s in sorted(sub['State_reordered'].unique()):
      for metric in metrics_test:
         dvals = sub[sub['State_reordered'] == s][metric].dropna()
         if len(dvals) > 1:
               stat, p = ttest_1samp(dvals, 0)
               pval_dict[(t, s, metric)] = p

for i, t in enumerate(types):
   sub = delta[delta['in_type'] == t]
   for j, metric in enumerate(metrics):
      sns.barplot(x='State_reordered', y=metric, data=sub, ax=axes[j, i], fill=False, errorbar='se')
   #   sns.stripplot(x='State_reordered', y=metric, data=sub, ax=axes[j, i])
      if i == 0:
         axes[j, i].set_ylabel(ylabels[j])
      else:
         axes[j, i].set_ylabel('')
      if j == 0:
         axes[j, i].set_title(t)

      # Add significance asterisks
      states = sorted(sub['State_reordered'].unique())
      for k, state in enumerate(states):
         key = (t, state, metric)
         if key in pval_dict:
            p = pval_dict[key]
            if p < 0.001:
               stars = '***'
            elif p < 0.01:
               stars = '**'
            elif p < 0.05:
               stars = '*'
            else:
               continue
            # Estimate y-position for the asterisk
            bar_vals = sub[sub['State_reordered'] == state][metric].dropna()
            y = bar_vals.mean()
            axes[j, i].text(k, y, stars, ha='center', va='bottom', fontsize=30)

plt.tight_layout()
plt.show()

from scipy.stats import ttest_1samp

metrics_test = ['Δ Time in state', 'Δ Lick rate', 'Δ Velocity']
for t in types:
    sub = delta[delta['in_type'] == t]
    print(f"\nGroup: {t}")
    for s in sorted(sub['State_reordered'].unique()):
        print(f"  State {s}:")
        for metric in metrics_test:
            dvals = sub[sub['State_reordered'] == s][metric].dropna()
            stat, p = ttest_1samp(dvals, 0)
            print(f"{metric}: p = {p:.4f}")
