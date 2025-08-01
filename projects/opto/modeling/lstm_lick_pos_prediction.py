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
import torch.nn as nn
import torch.nn.functional as F

class BNN_Conv1D(nn.Module):
   def __init__(self, input_channels, dropout_rate=0.2):
      super().__init__()
      self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
      self.drop1 = nn.Dropout(dropout_rate)
      self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
      self.drop2 = nn.Dropout(dropout_rate)
      self.pool = nn.AdaptiveAvgPool1d(1)
      self.fc = nn.Linear(64, 1)

   def forward(self, x):
      # Input x shape: (batch, time, cells)
      x = x.permute(0, 2, 1)  # to (batch, channels=n_cells, time)
      x = F.relu(self.conv1(x))
      x = self.drop1(x)
      x = F.relu(self.conv2(x))
      x = self.drop2(x)
      x = self.pool(x).squeeze(-1)
      return self.fc(x)

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
   # Assume these exist:
   # Fc3: shape (time, n_cells)
   # trialnum: shape (time,) indicating trial number
   # lick_position: shape (time,) with lick position at each timepoint
   trial_X =[]
   trial_y =[]
   for ep in range(len(eps)-1):
      eprng = np.arange(eps[ep],eps[ep+1])
      unique_trials = np.unique(trialnum[eprng])
      lick_position_rel = lick_position[eprng]-rewlocs[ep]
      ypos = ybinned[eprng]
      lick_position_rel[(ypos>(rewlocs[ep]-(rewsize/2))) & (ypos<(rewlocs[ep]+(rewsize/2)))]=np.nan
      lick_position_rel[lick_rate[eprng]>8]=np.nan
      lick_position_rel[licks[eprng]==0]=np.nan
      # trial_X = []
      # trial_y = []
      for tr in unique_trials:
         tr_mask = trialnum[eprng] == tr
         fc_trial = Fc3[eprng][tr_mask, :]                  # shape (t, n_cells)
         # remove later activity
         ypos_tr = ypos[tr_mask]
         fc_trial=fc_trial[(ypos_tr>67/scalingf)]
         lick_trial = lick_position_rel[tr_mask]         # shape (t,)
         avg_lick_pre_rew = np.nanmean(lick_trial)
         if fc_trial.shape[0] >= 10 and not np.isnan(avg_lick_pre_rew):
            trial_X.append(fc_trial)
            trial_y.append(avg_lick_pre_rew)
      trial_X_ep.append(trial_X)
      trial_y_ep.append(trial_y)
      
   max_time = np.nanmax([len(xx) for xx in trial_X])
   trial_fc = np.zeros((len(trial_X),max_time,trial_X[0].shape[1]))
   for trind,trx in enumerate(trial_X):
      trial_fc[trind,:len(trx)]=trx
   X = np.stack(trial_fc)   # shape (n_trials, time, n_cells)
   y = np.array(trial_y)  # shape (n_trials,)
   from sklearn.model_selection import train_test_split
   import torch
   from torch.utils.data import TensorDataset, DataLoader

   # Split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Convert to PyTorch tensors
   X_train = torch.tensor(X_train, dtype=torch.float32)
   X_test = torch.tensor(X_test, dtype=torch.float32)
   y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
   y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

   train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
   model = BNN_Conv1D(input_channels=X_train.shape[2], dropout_rate=0.3)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   loss_fn = nn.MSELoss()

   model.train()
   for epoch in range(100):
      for xb, yb in train_loader:
         pred = model(xb)
         loss = loss_fn(pred, yb)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
   def predict_mc(model, X, n_samples=100):
      model.eval()
      preds = []
      with torch.no_grad():
         for _ in range(n_samples):
            preds.append(model(X).cpu().numpy())
      preds = np.stack(preds)
      return preds.mean(0).squeeze(), preds.std(0).squeeze()
   
   y_pred_mean, y_pred_std = predict_mc(model, X_test, n_samples=100)
   from sklearn.metrics import r2_score
   plt.errorbar(y_test.numpy().squeeze(), y_pred_mean, yerr=y_pred_std, fmt='o', alpha=0.5)
   plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
   plt.xlabel("True Lick Position")
   plt.ylabel("Predicted (Mean ± SD)")
   plt.title("BNN: Cell × Time Input")
   plt.grid(True)
   plt.show()

   print("R² score:", r2_score(y_test.numpy().squeeze(), y_pred_mean))
