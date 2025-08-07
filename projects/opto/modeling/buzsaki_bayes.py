#%%

import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity

import numpy as np, sys
import scipy.io, scipy.interpolate, scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
import random
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import smooth_lick_rate
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays,make_tuning_curves
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

bins = 90
goal_window_cm = 20
conddf=pd.read_csv(r'Z:\condition_df\conddf_performance_chrimson.csv')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'lick_prediction.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

conddf = conddf[(conddf.in_type!='vip') & (conddf.in_type!='vip_ex')]
iis = np.arange(len(conddf))  # Animal indices
iis = [ii for ii in iis if ii!=202]
dct = {}
iis=np.array(iis)

def get_rewzones(rewlocs, gainf):
   # Initialize the reward zone numbers array with zeros
   rewzonenum = np.zeros(len(rewlocs))
   
   # Iterate over each reward location to determine its reward zone
   for kk, loc in enumerate(rewlocs):
      if 50*gainf <= loc <= 86 * gainf:
         rewzonenum[kk] = 1  # Reward zone 1
      elif 86 * gainf <= loc <= 135 * gainf:
         rewzonenum[kk] = 2  # Reward zone 2
      elif loc >= 135 * gainf:
         rewzonenum[kk] = 3  # Reward zone 3
         
   return rewzonenum

iis=iis[iis>17]
#%%
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
   lick_rate = smooth_lick_rate(licks, np.nanmedian(np.diff(time)))
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
   from sklearn.model_selection import train_test_split
   from sklearn.decomposition import PCA
   import torch
   from torch.utils.data import TensorDataset, DataLoader
   lick_position=np.zeros_like(licks)
   lick_position[licks>0] = ybinned[licks>0]
   ybinned_rel = []
   # Assume these exist:
   # Fc3: shape (time, n_cells)
   # trialnum: shape (time,) indicating trial number
   # lick_position: shape (time,) with lick position at each timepoint
   trial_X =[]
   trial_y =[]
   lick_rel = []
   trial_pos = []
   trial_states = []
   strind = []
   flind = []
   lick_rate_trial = []
   ep_trials = []
   all_trial_num=[]
   # 50 msec time bin
   bin_size = int(0.05 * 1/np.nanmedian(np.diff(time)))  # number of frames in 100ms
   rzs= get_rewzones(rewlocs,1/scalingf)
   for ep in range(len(eps)-1):
      eprng = np.arange(eps[ep],eps[ep+1])
      unique_trials = np.unique(trialnum[eprng])
      success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
      strials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in strials])
      if ep>0: strials_ind=strials_ind+all_trial_num[-1][-1]+1
      ftrials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in ftrials])
      if ep>0: ftrials_ind=ftrials_ind+all_trial_num[-1][-1]+1
      ttr_ind = np.arange(len(ttr))
      if ep>0: ttr_ind = np.arange(len(ttr))+all_trial_num[-1][-1]+1
      strind.append(strials_ind)
      flind.append(ftrials_ind)
      all_trial_num.append(ttr_ind)
      lick_position_rel = lick_position[eprng]-(rewlocs[ep]+rewsize/2)
      ypos = ybinned[eprng]
      lick_position_rel = lick_position_rel.astype(float)
      lick_position_rel[lick_rate[eprng]>9]=np.nan
      lick_position_rel[licks[eprng]==0]=np.nan
      # trial_X = []
      # trial_y = []
      for tr in unique_trials:
         tr_mask = trialnum[eprng] == tr
         fc_trial = Fc3[eprng][tr_mask, :]                  # shape (t, n_cells)
         # remove later activity
         ypos_tr = ypos[tr_mask]
         fc_trial=fc_trial#[(ypos_tr<((rewlocs[ep]-rewsize/2)))]
         lick_trial = lick_rate[eprng][tr_mask]         # shape (t,)
         
         # avg_lick_pre_rew = np.nanmedian(lick_trial)
         avg_lick_pre_rew = np.nanmedian(lick_trial) # early licks
         if fc_trial.shape[0] >= 10 and tr>2:#; dont exclude probes for now
            fc_trial_binned = fc_trial
            trial_X.append(fc_trial_binned)
            trial_y.append(rzs[ep])
            # trial_pos.append(ypos_tr[(ypos_tr<((rewlocs[ep]-rewsize/2)))])
            trial_pos.append(ypos_tr)
            lick_rate_trial.append(lick_trial)
            ep_trials.append(ep)
            
   # trial_y = get_rewzones(trial_y, 1/scalingf)
   max_time = np.nanmax([len(xx) for xx in trial_X])
   strind=np.concatenate(strind)
   flind = np.concatenate(flind)
   trial_fc_org = np.zeros((len(trial_X),max_time,trial_X[0].shape[1]))
   for trind,trx in enumerate(trial_X):
      trial_fc_org[trind,:len(trx)]=trx
      
   X = np.stack(trial_fc_org)   # shape (n_trials, time, n_cells)
   # Reshape to (n_trials*time, n_cells) for cell-wise standardization
   n_trials, t, n_cells = X.shape
   trial_pos_ = np.zeros((len(trial_pos),max_time))
   for trind,trx in enumerate(trial_pos):
      trial_pos_[trind,:len(trx)]=trx

   trial_lick = np.zeros((len(lick_rate_trial),max_time))
   for trind,trx in enumerate(lick_rate_trial):
      trial_lick[trind,:len(trx)]=trx
      
   # Sample data: (trials x time x cells)
   # only correct trials?
   n_trials, T, N = trial_fc_org.shape
   fc3 = trial_fc_org
   ybinned = trial_pos_
   goal_zone = np.array(trial_y)-1

   n_pos_bins = 90
   n_goals = 3
   dt = np.nanmedian(np.diff(time))  # 10 ms per bin
   var_pos = 25  # cm^2
   p_stay_goal = 0.9 ** dt

   def estimate_tuning(fc3, ybinned, goal_zone, cm_per_bin=1, bin_size_cm=3, n_goals=3):
      trials, time, N = fc3.shape
      # Re-bin positions to 3 cm bins
      pos3cm = (ybinned // (bin_size_cm // cm_per_bin)).astype(int)
      n_pos_bins = pos3cm.max() + 1  # auto-detect number of 3 cm bins
      tuning = np.zeros((N, n_pos_bins, n_goals))
      for i in range(N):  # loop over cells
         for g in range(n_goals):  # loop over goals
            mask = np.ravel(goal_zone[:, None] == g)  # trial x time
            spikes = fc3[mask, :,i]
            positions = pos3cm[mask]
            for x in range(n_pos_bins):
                  inds = positions == x
                  if inds.sum() > 0:
                     tuning[i, x, g] = np.nanmean(spikes[inds])
      return tuning

   all_indices=np.arange(fc3.shape[0])
   # Split indices instead of the data directly
   train_idx, test_idx = train_test_split(all_indices, test_size=0.6, random_state=42)
   # Now use the indices to subset your data
   fc3_train, fc3_test = fc3[train_idx], fc3[test_idx]
   ybinned_train, ybinned_test = ybinned[train_idx], ybinned[test_idx]
   goal_zone_train, goal_zone_test = goal_zone[train_idx], goal_zone[test_idx]

   # training ; use held out trials?
   tuning = estimate_tuning(fc3_train, ybinned_train, goal_zone_train)

   # Decoder
   def decode_trial(trial_fc, trial_ybin, goal, tuning):
      T = trial_fc.shape[0]
      log_post = np.full((T, n_pos_bins, n_goals), -np.inf)
      log_prior = np.log(np.ones((n_pos_bins, n_goals)) / (n_pos_bins * n_goals))

      for t in range(T):
         obs = trial_fc[t]
         log_likelihood = np.zeros((n_pos_bins, n_goals))
         for x in range(n_pos_bins):
            for g in range(n_goals):
               lam = tuning[:, x, g]
               lam = np.clip(lam, 1e-3, None)
               log_likelihood[x, g] = np.sum(obs * np.log(lam) - lam)

         if t == 0:
            log_post[t] = log_prior + log_likelihood
         else:
            prev_log_post = log_post[t - 1]
            trans_pos = gaussian_filter1d(np.exp(prev_log_post), sigma=np.sqrt(var_pos), axis=0)
            trans_goal = (1 - p_stay_goal) / (n_goals - 1)
            for g in range(n_goals):
               sticky = p_stay_goal * trans_pos[:, g] + trans_goal * np.sum(trans_pos, axis=1)
               log_post[t, :, g] = np.log(sticky + 1e-10) + log_likelihood[:, g]

         log_post[t] -= logsumexp(log_post[t])

      return np.exp(log_post)
   # trial = np.random.choice(test_idx)
   correct = []
   time_before_change=[]
   predicted=[]
   for trial in test_idx:
      # Run decoder on a trial
      post = decode_trial(fc3[trial], ybinned[trial], goal_zone[trial], tuning)
      post_goal = post.sum(axis=1)  # marginalize over position
      post_pos = post.sum(axis=2)  # marginalize over goal
      # get real rewloc start
      ep = ep_trials[trial]
      rewloc_start = rewlocs[ep-1]-rewsize/2
      ypos_temp = ybinned[trial].copy()
      ypos_temp[ypos_temp==0]=1000000
      rewloc_ind = np.where(ypos_temp<rewloc_start)[0]
      rewloc_ind = rewloc_ind[-1]
      real_ypos = ybinned[trial][rewloc_ind]
      # Plot marginal posterior
      # Change point detection
      # change to rz 1/2/3
      goal_trace = np.argmax(post_goal, axis=1)+1  # MAP goal trace
      model = rpt.Pelt(model="l2").fit(goal_trace)
      bkps = np.array(model.predict(pen=10))
      pred_goal_zone = goal_trace[:rewloc_ind]
      real_goal_zone = goal_zone[trial]+1
      changepoint = bkps[bkps<rewloc_ind]
      if len(changepoint)>0:
         pred_goal_zone_cp = np.nanmedian(goal_trace[changepoint[-1]:rewloc_ind])
      else:
         pred_goal_zone_cp = np.nanmedian(goal_trace[:rewloc_ind])
      predicted.append([pred_goal_zone_cp,real_goal_zone])
      if pred_goal_zone_cp==real_goal_zone:
         correct.append(trial)
         if len(changepoint)>0:
            time_before_change.append((rewloc_ind-changepoint[-1])*np.nanmedian(np.diff(time)))
         else:
            time_before_change.append((rewloc_ind)*np.nanmedian(np.diff(time)))
      # # Plot goal change points
      # plt.figure(figsize=(8, 3))
      # plt.plot(goal_trace, label="Goal (MAP)")
      # for cp in bkps[:-1]:
      #    plt.axvline(cp, color='red', linestyle='--')
      # plt.title("Goal Change Point Detection")
      # plt.xlabel("Time Bin")
      # plt.ylabel("Goal")
      # plt.legend()
      # plt.tight_layout()
      # plt.show()

   plt.figure()
   plt.plot(goal_trace)
   plt.plot(ybinned[trial]/10)
   plt.plot(trial_lick[trial])

   # rate correct
   total_rate = len(correct)/len(test_idx)

   test_s = [xx for xx in test_idx if xx in strind]
   test_f = [xx for xx in test_idx if xx in flind]
   correct_s = [xx for xx in correct if xx in test_s]
   correct_f = [xx for xx in correct if xx in test_f]
   # for correct/incorrect trials
   s_rate = len(correct_s)/len(test_s)
   if len(test_f)>0:
      f_rate = len(correct_f)/len(test_f)
   else: 
      f_rate=np.nan
   
   time_before_predict = np.nanmean(time_before_change)
   time_before_predict_s = np.nanmean([xx for ii,xx in enumerate(time_before_change) if correct[ii] in test_s])
   time_before_predict_f = np.nanmean([xx for ii,xx in enumerate(time_before_change) if correct[ii] in test_f])
   
   print(f'total prediction rate: {total_rate*100:.2g}%')
   print(f'correct prediction rate: {s_rate*100:.2g}%')
   print(f'incorrect prediction rate: {f_rate*100:.2g}%')
   print(f'prediction latency (correct trials): {time_before_predict_s:.2g}s')
   print(f'prediction latency (incorrect trials): {time_before_predict_f:.2g}s')
   dct[f'{animal}_{day}']=[total_rate,s_rate,f_rate,time_before_predict, time_before_predict_s,time_before_predict_f,predicted,rzs,eps]
# %%
df=pd.DataFrame()
# 8-12 = pred
# 13-17 = opto
# Add all the variables
df['prev_s_rate'] = [v[7] for k, v in dct.items()]
df['prev_f_rate'] = [v[8] for k, v in dct.items()]
df['prev_time_before_predict_s'] = [v[9] for k, v in dct.items()]
df['prev_time_before_predict_f'] = [v[10] for k, v in dct.items()]
df['prev_time_to_rew'] = [v[11] for k, v in dct.items()]

df['opto_s_rate'] = [v[12] for k, v in dct.items()]
df['opto_f_rate'] = [v[13] for k, v in dct.items()]
df['opto_time_before_predict_s'] = [v[14] for k, v in dct.items()]
df['opto_time_before_predict_f'] = [v[15] for k, v in dct.items()]
df['opto_time_to_rew'] = [v[16] for k, v in dct.items()]

df['animals'] = [k.split('_')[0] for k, v in dct.items()]
df['days'] = [int(k.split('_')[1]) for k, v in dct.items()]
df_long = pd.DataFrame({
    's_rate': df['prev_s_rate'].tolist() + df['opto_s_rate'].tolist(),
    'f_rate': df['prev_f_rate'].tolist() + df['opto_f_rate'].tolist(),
    'time_before_predict_s': df['prev_time_before_predict_s'].tolist() + df['opto_time_before_predict_s'].tolist(),
    'time_before_predict_f': df['prev_time_before_predict_f'].tolist() + df['opto_time_before_predict_f'].tolist(),
    'time_to_rew': df['prev_time_to_rew'].tolist() + df['opto_time_to_rew'].tolist(),
    'condition': ['prev'] * len(df) + ['opto'] * len(df),
   'animals': df['animals'].tolist() * 2,
    'days': df['days'].tolist() * 2

})
cdf = conddf.copy()
df = pd.merge(df_long, cdf, on=['animals', 'days'], how='inner')
df=df[df.in_type=='vip_ex']

sns.barplot(x='condition',y='s_rate',data=df)
# sns.barplot(x='condition',y='f_rate',data=df)