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
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays,make_tuning_curves
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

from itertools import combinations

def bin_activity(data, bin_size):
   n_bins = data.shape[0] // bin_size
   binned = data[:n_bins * bin_size].reshape(n_bins, bin_size, data.shape[1]).mean(axis=1)
   return binned

def normalize_rows(arr):
   # Normalize each row to [0,1]
   min_vals = arr.min(axis=1, keepdims=True)
   max_vals = arr.max(axis=1, keepdims=True)
   # Avoid division by zero for rows with constant values
   denom = max_vals - min_vals
   denom[denom == 0] = 1
   return (arr - min_vals) / denom

def normalize_1d(arr):
   # Normalize each row to [0,1]
   min_vals = arr.min()
   max_vals = arr.max()
   # Avoid division by zero for rows with constant values
   denom = max_vals - min_vals
   if denom==0:
      denom=1
   return (arr - min_vals) / denom

# ============================== #
# CONFIGURATION
# ============================== #
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

class BNN_Conv1D(nn.Module):
   def __init__(self, input_channels, dropout_rate=0.2):
      super().__init__()
      self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
      self.drop1 = nn.Dropout(dropout_rate)
      self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
      self.drop2 = nn.Dropout(dropout_rate)
      self.pool = nn.AdaptiveAvgPool1d(1)

      self.fc1 = nn.Linear(64, 32)
      self.drop3 = nn.Dropout(dropout_rate)
      self.fc2 = nn.Linear(32, 1)

   def forward(self, x):
      # x: (batch_size, n_cells, time)
      x = self.conv1(x)       # (batch, 32, time)
      x = F.relu(x)
      x = self.drop1(x)

      x = self.conv2(x)       # (batch, 64, time)
      x = F.relu(x)
      x = self.drop2(x)

      x = torch.mean(x, dim=2)  # global average pooling over time -> (batch, 64)

      x = F.relu(self.fc1(x))   # -> (batch, 32)
      x = self.drop3(x)

      return self.fc2(x)    

# ============================== #
# MAIN LOOP OVER ANIMALS
# ============================== #
#%%
# iis=iis[iis>183]
errors=[]
#%%
# iis=iis[iis>79]

for ii in iis:
   if ii!=53 and ii!=202 and ii!=80:
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
      eps_dff = np.insert(eps,0,-34043904)      
      dt=np.nanmedian(np.diff(time))
      timethres = (1/dt)*60*3
      eps = eps[np.diff(eps_dff)>timethres]
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
      ybinned_rel = []
      trial_states = []
      strind = []
      flind = []
      ep_trials = []
      bin_size = int(0.1 * 1/np.nanmedian(np.diff(time)))  # number of frames in 100ms
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
            fc_trial=fc_trial[(ypos_tr<((rewlocs[ep]-rewsize/2)-50))]
            lick_trial = lick_position_rel[tr_mask]         # shape (t,)
            lick_rel.append(lick_trial)
            ybinned_rel.append(ypos_tr-(rewlocs[ep]-(rewsize/2)))
            # avg_lick_pre_rew = np.nanmedian(lick_trial)
            avg_lick_pre_rew = np.nanmedian(lick_trial) # early licks
            if fc_trial.shape[0] >= 10 and not np.isnan(avg_lick_pre_rew) and tr>2:#; dont exclude probes for now
               fc_trial_binned = bin_activity(fc_trial, bin_size)
               trial_X.append(fc_trial_binned)
               trial_y.append(avg_lick_pre_rew)
               ep_trials.append(ep+1)
         # test
         # lick_r = lick_rate[eprng]
         # lick_r[lick_r>5]=0
         # plt.plot(lick_position_rel/50,linewidth=3)
         # plt.plot(lick_r-3,alpha=0.5)
      # split into rz 1,2,3
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
      X_reshaped = X.reshape(-1, n_cells)
      # Fit and transform using StandardScaler
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X_reshaped)
      # Reshape back to original shape
      X = X_scaled.reshape(n_trials, t, n_cells)
      y = np.array(trial_y)[np.array(ep_trials) == eptest-1]  # shape (n_trials,)
      from sklearn.model_selection import train_test_split
      import torch
      from torch.utils.data import TensorDataset, DataLoader
      all_indices=np.arange(X.shape[0])
      # Split indices instead of the data directly
      train_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=42)
      # Now use the indices to subset your data
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      
      # Convert to PyTorch tensors
      X_train = torch.tensor(X_train, dtype=torch.float32)
      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
      y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
      X_train = X_train.permute(0, 2, 1)  # shape: (n_trials, n_cells, time)
      X_test = X_test.permute(0, 2, 1)    # shape: (n_trials, n_cells, time)

      train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
      model = BNN_Conv1D(input_channels=X_train.shape[1], dropout_rate=0.3)
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
      loss_fn = nn.MSELoss()
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

      model.train()
      train_losses=[]
      for epoch in range(1200):
         total_loss=0
         for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if epoch%100==0: print(loss.item())
         train_losses.append(total_loss)

      def predict_mc(model, X, n_samples=100):
         model.train()  
         preds = []
         with torch.no_grad():
            for _ in range(n_samples):
               preds.append(model(X).cpu().numpy())
         preds = np.stack(preds)
         return preds.mean(0).squeeze(), preds.std(0).squeeze()

      y_pred_mean, y_pred_std = predict_mc(model, X_test, n_samples=100)
      from sklearn.metrics import r2_score
      test_st_id = np.array([ii for ii,xx in enumerate(test_idx) if xx in strind])
      test_flind = np.array([ii for ii,xx in enumerate(test_idx) if xx in flind])
      test_probe = np.array([ii for ii,xx in enumerate(test_idx) if xx not in strind and xx not in flind])
      from sklearn.metrics import mean_squared_error, mean_absolute_error
      # Assuming y_true and y_pred are your arrays
      rmse = mean_squared_error(y_test, y_pred_mean)
      mae = mean_absolute_error(y_test, y_pred_mean)
      print(f"RMSE: {rmse:.2f}")
      print(f"MAE: {mae:.2f}")
      
      # apply to opto epoch  
      # prev epoch training
      trial_fc = trial_fc_org[np.array(ep_trials) == eptest]
      X = np.stack(trial_fc)   # shape (n_trials, time, n_cells)
      # Reshape to (n_trials*time, n_cells) for cell-wise standardization
      n_trials, t, n_cells = X.shape
      X_reshaped = X.reshape(-1, n_cells)
      # Fit and transform using StandardScaler
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X_reshaped)
      # Reshape back to original shape
      X = X_scaled.reshape(n_trials, t, n_cells)
      y = np.array(trial_y)[np.array(ep_trials) == eptest]  # shape (n_trials,)
      all_indices=np.arange(X.shape[0])
      # Convert to PyTorch ensors
      X_test = torch.tensor(X, dtype=torch.float32)
      y_test = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
      X_test = X_test.permute(0, 2, 1)    # shape: (n_trials, n_cells, time)

      y_pred_mean, y_pred_std = predict_mc(model, X_test, n_samples=100)
      test_idx = np.where(np.array(ep_trials)==eptest)[0]
      test_st_id = np.array([ii for ii,xx in enumerate(test_idx) if xx in strind])
      test_flind = np.array([ii for ii,xx in enumerate(test_idx) if xx in flind])
      from sklearn.metrics import mean_squared_error, mean_absolute_error
      # Assuming y_true and y_pred are your arrays
      rmse = mean_squared_error(y_test, y_pred_mean)
      mae = mean_absolute_error(y_test, y_pred_mean)
      print(f"RMSE: {rmse:.2f}")
      print(f"MAE: {mae:.2f}")

      fig,ax = plt.subplots()
      if len(test_st_id)>0:
         ax.scatter(y_test.numpy().squeeze()[test_st_id], y_pred_mean[test_st_id], alpha=0.5,color='seagreen',label='correct')
      if len(test_flind)>0:
         ax.scatter(y_test.numpy().squeeze()[test_flind], y_pred_mean[test_flind], alpha=0.5,color='firebrick',label='incorrect')
      vmin = np.nanmin(np.concatenate([y, y_pred_mean]))
      vmax = np.nanmax(np.concatenate([y, y_pred_mean]))
      ax.plot([vmin,vmax], [vmin,vmax], 'k--')
      ax.axvline(-50,color='grey')
      ax.axvline(0,color='grey',linestyle='--')
      ax.axhline(0,color='grey',linestyle='--')
      ax.axhline(-50,color='grey')
      ax.set_xlabel("True Lick Position (cm)")
      ax.set_ylabel("Predicted Lick Position (cm)")
      r2 = r2_score(y_test, y_pred_mean)
      ax.set_title(f'opto epoch, {animal}, {day}, {eptest}, r2: {r2:.3g}, mae: {mae:.2g} cm')
      ax.legend()
      pdf.savefig(fig)
      plt.show() 

      # tc w/ dark time
      print('making tuning curves...\n')
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 
      goal_window = 20*(2*np.pi/track_length)
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      perm = list(combinations(range(len(coms_correct)), 2)) 
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
      com_goal=np.unique(np.concatenate(com_goal))
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8)
      lick_tcs_correct_abs, lick_coms_correct_abs, lick_tcs_fail_abs, lick_coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8)

      figmul=1.5
      # After training:
      with torch.no_grad():
         weights = model.conv1.weight.cpu().numpy()  # shape: (32 filters, input_channels, kernel_size)
         # Aggregate across filters and kernel positions
         cell_importance = np.abs(weights).mean(axis=(0, 2))  # shape: (input_channels,)
         top_cells = np.argsort(cell_importance)[::-1]  # descending
         print("Top contributing cells (by average absolute weight):", top_cells[:10])
      if len(top_cells)>200:
         top_p = int(0.2*len(top_cells))
         topp=20
      else:
         top_p = int(0.4*len(top_cells))
         topp=40
      top_cells_that_are_rew_cells = [xx for xx in top_cells[:top_p] if xx in com_goal]
      top_cells_that_are_not_rew_cells = [xx for xx in top_cells[:top_p] if xx not in com_goal]
      # needs to last more than 4 min 
      fr = np.nanmedian(np.diff(time))
      thres = (1/fr)*60*4
      epochs = len(eps[np.insert(np.diff(eps),0,10000)>thres])-1
      fig, axes = plt.subplots(ncols=2,nrows = epochs,sharex=True, sharey='col')
      for r in range(epochs):
         norm_tcs_nonrew = normalize_rows(tcs_correct_abs[r, top_cells_that_are_not_rew_cells])[np.argsort(coms_correct_abs[0, top_cells_that_are_not_rew_cells])]
         norm_tcs_rew = normalize_rows(tcs_correct_abs[r, top_cells_that_are_rew_cells])[np.argsort(coms_correct_abs[0, top_cells_that_are_rew_cells])]
         axes[r,0].imshow(norm_tcs_nonrew, aspect='auto')
         axes[r,1].imshow(norm_tcs_rew, aspect='auto')
         axes[0,0].set_title(f'top {topp}% contributing not rew cells')
         axes[r,0].axvline(rewlocs[r]/3,color='w')
         axes[r,1].axvline(rewlocs[r]/3,color='w')
         axes[0,1].set_title(f'top {topp}% contributing rew cells')
         lick_rate_map = normalize_1d(lick_tcs_correct_abs[r][0])
         axes[r,0].plot(
            np.arange(lick_rate_map.shape[-1]),  # x-axis (time)
            norm_tcs_nonrew.shape[0]-1 - lick_rate_map.T * norm_tcs_nonrew.shape[0]/figmul,  # inverted y
            color='w'
         )
         axes[r,1].plot(
            np.arange(lick_rate_map.shape[-1]),  # x-axis (time)
            norm_tcs_rew.shape[0]-1 - lick_rate_map.T * norm_tcs_rew.shape[0]/figmul,  # inverted y
            color='w'
         )
         axes[0,0].text(45, 0.5, 'lick rate', color='white',
         fontsize=12, ha='center', va='bottom', clip_on=True)

      fig.suptitle(f'Correct, last 8 trials, {animal}, {day}')
      # incorrects?
      plt.show()
      pdf.savefig(fig)

      fig, axes = plt.subplots(ncols=2,nrows = epochs,sharex=True, sharey='col')
      for r in range(epochs):
         norm_tcs_nonrew = normalize_rows(tcs_fail_abs[r, top_cells_that_are_not_rew_cells])[np.argsort(coms_fail_abs[0, top_cells_that_are_not_rew_cells])]
         norm_tcs_rew = normalize_rows(tcs_fail_abs[r, top_cells_that_are_rew_cells])[np.argsort(coms_fail_abs[0, top_cells_that_are_rew_cells])]
         axes[r,0].imshow(norm_tcs_nonrew, aspect='auto')
         axes[r,1].imshow(norm_tcs_rew, aspect='auto')
         axes[0,0].set_title(f'top {topp}% contributing not rew cells')
         axes[r,0].axvline(rewlocs[r]/3,color='w')
         axes[r,1].axvline(rewlocs[r]/3,color='w')
         axes[0,1].set_title(f'top {topp}% contributing rew cells')
         lick_rate_map = normalize_1d(lick_tcs_fail_abs[r][0])
         axes[r,0].plot(
            np.arange(lick_rate_map.shape[-1]),  # x-axis (time)
            norm_tcs_nonrew.shape[0]-1 - lick_rate_map.T * norm_tcs_nonrew.shape[0]/figmul,  # inverted y
            color='w'
         )
         axes[r,1].plot(
            np.arange(lick_rate_map.shape[-1]),  # x-axis (time)
            norm_tcs_rew.shape[0]-1 - lick_rate_map.T * norm_tcs_rew.shape[0]/figmul,  # inverted y
            color='w'
         )
         axes[0,0].text(45, 0.5, 'lick rate', color='white',
               fontsize=12, ha='center', va='bottom', clip_on=True)


      fig.suptitle(f'Incorrects, {animal}, {day}')
      # incorrects?
      plt.show()
      pdf.savefig(fig)
      fig,ax=plt.subplots()
      for i in range(epochs):
         if sum(np.isnan(coms_rewrel[i,top_cells[:top_p]]))<len(coms_rewrel[i,top_cells[:top_p]]):
            ax.hist(coms_rewrel[i,top_cells[:top_p]],alpha=0.4)
      ax.set_ylabel('# top contributing cells')
      ax.set_xlabel('reward-centric distance')
      plt.show()
      fig.suptitle(f'{animal}, {day}')
      pdf.savefig(fig)
      errors.append([y_test.numpy().squeeze(),y_pred_mean,r2,test_st_id,test_flind,mae,coms_rewrel,top_cells_that_are_not_rew_cells,top_cells_that_are_rew_cells])
pdf.close()
#%%
df=pd.DataFrame()
df['real_lick_dist']=np.concatenate([xx[0] for xx in errors])
df['pred_lick_dist']=np.concatenate([xx[1] for xx in errors])
df['mae']=np.concatenate([[xx[5]]*len(xx[0]) for xx in errors])
df['epoch_trials']=np.concatenate([xx[9] for xx in errors])