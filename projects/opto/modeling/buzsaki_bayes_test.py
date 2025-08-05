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

iis = np.arange(len(conddf))  # Animal indices
iis = [ii for ii in iis if ii!=202]
dfs = {}
iis=np.array(iis)
# iis=iis[iis>167]
import torch.nn as nn
import torch.nn.functional as F

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

ii=60
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
ep_trials = []
bin_size = int(0.05 * 1/np.nanmedian(np.diff(time)))  # number of frames in 100ms
rzs= get_rewzones(rewlocs,1/scalingf)
for ep in range(len(eps)-1):
   eprng = np.arange(eps[ep],eps[ep+1])
   unique_trials = np.unique(trialnum[eprng])
   success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
   strials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in strials])
   if ep>0: strials_ind=strials_ind+strind[-1][-1]+1
   ftrials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in ftrials])
   strind.append(strials_ind)
   flind.append(ftrials_ind)
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
      fc_trial=fc_trial#[(ypos_tr<((rewlocs[ep]-rewsize/2)-50))]
      lick_trial = lick_position_rel[tr_mask]         # shape (t,)
      lick_rel.append(lick_trial)
      # avg_lick_pre_rew = np.nanmedian(lick_trial)
      avg_lick_pre_rew = np.nanmedian(lick_trial) # early licks
      if fc_trial.shape[0] >= 10 and not np.isnan(avg_lick_pre_rew) and tr>2:#; dont exclude probes for now
         fc_trial_binned = fc_trial
         trial_X.append(fc_trial_binned)
         trial_y.append(rzs[ep])
         trial_pos.append(ypos_tr)
         ep_trials.append(ep+1)
         
# trial_y = get_rewzones(trial_y, 1/scalingf)
max_time = np.nanmax([len(xx) for xx in trial_X])
strind=np.concatenate(strind)
flind = np.concatenate(flind)
trial_fc_org = np.zeros((len(trial_X),max_time,trial_X[0].shape[1]))
for trind,trx in enumerate(trial_X):
   trial_fc_org[trind,:len(trx)]=trx
   
# prev epoch training
trial_fc = trial_fc_org[np.array(ep_trials) == eptest-1]
X = np.stack(trial_fc)   # shape (n_trials, time, n_cells)
# Reshape to (n_trials*time, n_cells) for cell-wise standardization
n_trials, t, n_cells = X.shape
trial_pos_ = np.zeros((len(trial_pos),max_time))
for trind,trx in enumerate(trial_pos):
   trial_pos_[trind,:len(trx)]=trx

#%%
# Sample data: (trials x time x cells)
n_trials, T, N = trial_fc_org.shape
fc3 = trial_fc_org
ybinned = trial_pos_
goal_zone = np.array(trial_y)

n_pos_bins = 270
n_goals = 3
dt = np.nanmedian(np.diff(time))  # 10 ms per bin
var_pos = 25  # cm^2
p_stay_goal = 0.9 ** dt

# Estimate tuning curves
def estimate_tuning(fc3, ybinned, goal_zone, n_pos_bins, n_goals):
   tuning = np.zeros((N, n_pos_bins, n_goals))
   for i in range(N):
      for g in range(n_goals):
         mask = np.ravel(goal_zone[:, None] == g)
         pos = ybinned[mask]
         spikes = fc3[mask, :, i].flatten()
         positions = pos.flatten()
         for x in range(n_pos_bins):
               inds = positions == x
               if inds.sum() > 0:
                  tuning[i, x, g] = np.mean(spikes[inds])
   return gaussian_filter1d(tuning, sigma=1, axis=1)

tuning = estimate_tuning(fc3, ybinned, goal_zone, n_pos_bins, n_goals)

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
#%%
trial = np.random.randint(fc3.shape[0])
# Run decoder on a trial
post = decode_trial(fc3[trial], ybinned[trial], goal_zone[trial], tuning)
post_goal = post.sum(axis=1)  # marginalize over position
post_pos = post.sum(axis=2)  # marginalize over goal

# Plot marginal posterior
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(post_pos.T, aspect='auto', cmap='viridis')
plt.title("Marginal Posterior over Position")
plt.ylabel("Position Bin")
plt.xlabel("Time")

plt.subplot(1, 2, 2)
plt.imshow(post_goal.T, aspect='auto', cmap='plasma')
plt.title("Marginal Posterior over Goal")
plt.ylabel("Goal")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

# Change point detection
goal_trace = np.argmax(post_goal, axis=1)  # MAP goal trace
model = rpt.Pelt(model="l2").fit(goal_trace)
bkps = model.predict(pen=10)

# Plot goal change points
plt.figure(figsize=(8, 3))
plt.plot(goal_trace, label="Goal (MAP)")
for cp in bkps[:-1]:
   plt.axvline(cp, color='red', linestyle='--')
plt.title("Goal Change Point Detection")
plt.xlabel("Time Bin")
plt.ylabel("Goal")
plt.legend()
plt.tight_layout()
plt.show()
