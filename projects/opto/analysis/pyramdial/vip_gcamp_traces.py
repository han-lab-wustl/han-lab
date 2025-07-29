#%%
from scipy.io import loadmat
import os, scipy, sys
import glob, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl, pandas as pd, seaborn as sns
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

#%%
# Definitions and setups
mice = ["z14",'z15','z17']
dys_s = [[17,28,29,33,34,36],[6,9,11],[18,19]]
# dys_s = [[43,44,45,48]]
opto_ep_s = [[2,2,2,2,2,2],[2,2,2],[2,2]]
# cells_to_plot_s = [[466,159,423,200,299]]#, [16,6,9], 
cells_to_plot_s = [[0,5,9,3,7,1],[[66,55],[136,165],[93]],[2,83]]

src = r"Y:\analysis\fmats"
dffs_cp_dys = []
mind = 0
# Define the number of bins and the size of each bin
nbins = 90
bin_size = 3                
binsize = 0.3 # s
range_val = 8
# Processing loop
for m, mouse_name in enumerate(mice):
   days = dys_s[m]
   cells_to_plot = cells_to_plot_s[m]
   opto_ep = opto_ep_s[m]
   dyind = 0
   for dy in days:
      daypath = os.path.join(src, mouse_name,'days',
      f"{mouse_name}_day{dy:03d}_plane0_Fall.mat")
      data = loadmat(daypath, variable_names=['dFF', 'forwardvel', 'ybinned', 'VR',
      'timedFF', 'changeRewLoc', 'rewards', 'licks', 'trialnum'])
      dFF = data['dFF']
      changeRewLoc = data['changeRewLoc'].flatten()
      VR = data['VR']
      ybinned = data['ybinned'].flatten()
      forwardvel = data['forwardvel'].flatten()
      timedFF = data['timedFF'].flatten()
      rewards = np.hstack(data['rewards'])==.5
      rewards[:2000]=0 # remove first outliers?
      licks = data['licks']
      trialnum = data['trialnum'].flatten()
      print(daypath)

      # Additional processing
      eps = np.where(changeRewLoc > 0)[0]
      eps = np.append(eps, len(changeRewLoc))
      gainf = 1 / VR['scalingFACTOR'].item()
      rewloc = np.hstack(changeRewLoc[changeRewLoc > 0] * gainf)
      rewsize = VR['settings'][0][0][0][0][4] * gainf # reward zone is 5th element
      ypos = np.hstack(ybinned * gainf)
      velocity = forwardvel
      dffs_cp = []
      indtemp = 0

      rngopto = range(eps[opto_ep[dyind] - 1], eps[opto_ep[dyind]])
      rngpreopto = range(eps[opto_ep[dyind] - 2], eps[opto_ep[dyind] - 1])
      # normalize to rew loc?
      yposopto = ypos[rngopto]
      ypospreopto = ypos[rngpreopto]
      # mask for activity before reward loc
      # yposoptomask = np.hstack(yposopto < rewloc[opto_ep[dyind] - 1] - rewsize-10)
      # ypospreoptomask = np.hstack(ypospreopto < rewloc[opto_ep[dyind] - 2] - rewsize-10)
      # yposoptomask = np.hstack(yposopto > rewloc[opto_ep[dyind] - 1] - rewsize-10)
      # ypospreoptomask = np.hstack(ypospreopto > rewloc[opto_ep[dyind] - 2] - rewsize-10)
      # get entire tuning curve
      yposoptomask = (np.ones_like(np.hstack(yposopto))*True).astype(bool)
      ypospreoptomask = (np.ones_like(np.hstack(ypospreopto))*True).astype(bool)
      trialoptomask = trialnum[rngopto] > 10
      trialpreoptomask = trialnum[rngpreopto] > 10
      cp = cells_to_plot[dyind] # just get 1 cell        
      # try:
      #     if len(cp)>1:
      #         cp = cp[0]
      # except:
      #     print('e')
      dffopto = dFF[rngopto, :]
      dffpreopto = dFF[rngpreopto, :]
      dffs_cp.append([np.nanmean(dffopto[:, [cp]],axis=1), 
                     np.nanmean(dffpreopto[:, [cp]],axis=1)])
      # Initialize arrays for tuning curves
      # opto_tuning = np.ones(nbins)*np.nan
      # prevep_tuning = np.ones(nbins)*np.nan

      # Extract dFF arrays for the corresponding conditions
      optodff = np.nanmean(dffopto[:, [cp]],axis=1)
      prevepdff = np.nanmean(dffpreopto[:, [cp]],axis=1)
      # get tuning curve
      # # Process for 'opto' condition
      # # Create an index array from 0 to len(timedFF(rngopto)) - 1
      # time_moving = np.arange(len(timedFF[rngopto]))
      # ypos_mov = yposopto[time_moving]
      # # Filter by the mask conditions (y position and trial number)
      # time_moving = time_moving[yposoptomask & trialoptomask]
      # ypos_mov = ypos_mov[yposoptomask & trialoptomask]

      # # Bin the data
      # time_in_bin_opto = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

      # # Process for 'pre opto' condition
      # time_moving = np.arange(len(timedFF[rngpreopto]))
      # ypos_mov = ypospreopto[time_moving]
      # time_moving = time_moving[ypospreoptomask & trialpreoptomask]
      # ypos_mov = ypos_mov[ypospreoptomask & trialpreoptomask]

      # # Bin the data
      # time_in_bin_pre = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

      # # Compute the mean for each bin and populate the tuning curves
      # for bin_ in range(nbins):
      #     if len(time_in_bin_opto[bin_]) > 0:
      #         opto_tuning[bin_] = np.nanmean(optodff[time_in_bin_opto[bin_]])
      #     if len(time_in_bin_pre[bin_]) > 0:
      #         prevep_tuning[bin_] = np.nanmean(prevepdff[time_in_bin_pre[bin_]])
      # peri reward activity
      normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
      rewdFF_opto = eye.perireward_binned_activity(optodff, rewards[rngopto], 
               timedFF[rngopto], range_val, binsize)
      normmeanrewdFF, meanrewdFF_ctrl, normrewdFF, \
      rewdFF_ctrl = eye.perireward_binned_activity(prevepdff, rewards[rngpreopto], 
               timedFF[rngpreopto], range_val, binsize)
      #also get vel
      _, meanvel_ctrl, __, \
      velall_ctrl = eye.perireward_binned_activity(forwardvel, rewards[rngpreopto], 
               timedFF[rngpreopto], range_val, binsize)

      dffs_cp_dys.append([meanrewdFF_opto, meanrewdFF_ctrl,rewdFF_opto,rewdFF_ctrl, velall_ctrl])
      indtemp += 1
      dyind += 1
#%%
# per trial activity during ctrl?
trial_ctrl = [xx[1].T for xx in dffs_cp_dys[:6]]
velall_ctrl = [np.nanmean(xx[4],axis=1).T for xx in dffs_cp_dys[:6]]
fig,axes=plt.subplots(nrows=2,figsize=(6,3),sharex=True)
ax=axes[0]
im=ax.imshow(trial_ctrl,aspect='auto',cmap='magma')
ax.axvline(range_val/binsize,color='w',linestyle='--',linewidth=3)
ax.set_ylabel('Days')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.set_xlabel('Time from reward (s)')
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
fig.colorbar(im, ax=ax, label='VIP $\Delta F/F$')
ax=axes[1]
im=ax.imshow(velall_ctrl,aspect='auto',cmap='Greys')
ax.axvline(range_val/binsize,color='k',linestyle='--',linewidth=3)
ax.set_ylabel('')
ax.set_xlabel('Time from CS (s)')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
ax.set_xticks([0, int(range_val/binsize), (int(range_val/binsize)*2)])
ax.set_xticklabels([-8, 0, 8])
# ax.set_xticklabels(np.arange(-range_val, range_val+1, 1))
fig.colorbar(im, ax=ax, label='Velocity (cm/s)')
axes[0].set_title('VIP+ neuron activity')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'

plt.savefig(os.path.join(savedst,'vip_ctrl_activity.svg'),bbox_inches='tight')

#%%
# plot mean activity between diff epochs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
import os

s = 12
a = 0.7
fs = 46
ii = 0.5
y = 1.5
pshift = 0.2

# Simulate input arrays (replace with your actual data)
opto = [np.nanmax(xx[0][20:int(range_val/binsize)]) for xx in dffs_cp_dys][:10]
ctrl = [np.nanmax(xx[1][20:int(range_val/binsize)]) for xx in dffs_cp_dys][:10]

# Create dataframe
df = pd.DataFrame()
df['average_dff'] = np.concatenate([ctrl, opto])
df['condition'] = ['LED off'] * len(ctrl) + ['LED on'] * len(opto)

# Plot
fig, ax = plt.subplots(figsize=(2.5, 5))
sns.barplot(
    data=df, x='condition', y='average_dff', hue='condition', ax=ax, fill=False,
    palette={'LED off': "k", 'LED on': "darkgoldenrod"}, alpha=a
)
sns.stripplot(
    data=df, x='condition', y='average_dff', hue='condition', ax=ax, s=s,
    palette={'LED off': "k", 'LED on': "darkgoldenrod"}, alpha=a, dodge=False
)

# Add connecting lines
for i in range(len(opto)):
    ax.plot([0, 1], [ctrl[i], opto[i]], color='gray', alpha=0.5, linewidth=1.5)

# Clean up axes
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('VIP Pre-Reward $\Delta$F/F')

# Wilcoxon test and annotation
t, pval = scipy.stats.wilcoxon(opto, ctrl)
if pval < 0.001:
    ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
    ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
    ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii - 0.5, y + pshift, f'p={pval:.3g}', fontsize=12)
ax.set_title('Excitation')

# Save figure
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
plt.savefig(os.path.join(savedst, 'vip_pre_reward_dff_chrimson.svg'), bbox_inches='tight')

#%% 
from matplotlib.patches import Rectangle

def add_reward_zone_patches(ax, ypos_segment, start_idx, 
        rewloc, color='lightcoral', alpha=0.3):
    in_patch = False
    start = None
    for i, y in enumerate(ypos_segment):
        if 0 <= y < rewloc:
            if not in_patch:
                start = i
                in_patch = True
        else:
            if in_patch:
                ax.add_patch(Rectangle((start, 0), i - start, rewloc - 0,
                                       color=color, alpha=alpha, zorder=0))
                in_patch = False
    if in_patch:  # close final patch if still open
        ax.add_patch(Rectangle((start, 0), len(ypos_segment) - start, rewloc - 0,
                               color=color, alpha=alpha, zorder=0))
# z17, day 18
# plot individual traces
fig,axes = plt.subplots(nrows=2,ncols=1, figsize=(6,6),sharey=True,sharex=True)
x1,x2 = 1500,2200
tropto=trialnum[rngopto][x1:x2]
trprev=trialnum[rngpreopto][x1:x2]
yposopto = ypos[rngopto][x1:x2]
ypospreopto = ypos[rngpreopto][x1:x2]
# get rew loc
rl=rewloc[[0,1]]
axes[0].plot(prevepdff[x1:x2], 'k')
axes[1].plot(optodff[x1:x2], 'darkgoldenrod')
ax2_0 = axes[0].twinx()
ax2_0.plot(ypospreopto, color='gray')
ax2_0.set_ylabel('', color='gray')
ax2_0.tick_params(axis='y', labelcolor='gray')
ax2_0.spines[['top','bottom']].set_visible(False)
axes[0].set_title('LED off', color='k')
axes[1].set_title('LED on', color='darkgoldenrod')
axes[1].set_ylim([-0.2,4])
ax2_0 = axes[1].twinx()
ax2_0.plot(yposopto, color='gray')
ax2_0.set_ylabel('Position (cm)', color='gray')
ax2_0.tick_params(axis='y', labelcolor='gray')
add_reward_zone_patches(ax2_0, yposopto, x1, rewloc[1]-8)
ax2_0.spines[['top']].set_visible(False)
axes[1].set_ylabel('$\Delta$ F/F')
axes[0].set_ylabel('$\Delta$ F/F')
time = (x2-x1)/31.25
axes[1].set_xticks([0,x2-x1])
axes[1].set_xticklabels([0,int(time)])
axes[1].set_xlabel('Time (s)')
axes[0].spines[['top','bottom']].set_visible(False)
axes[1].spines[['top','right']].set_visible(False)

fig.suptitle('VIP+ neuron, Excitation')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
plt.savefig(os.path.join(savedst,'vip_traces_chrimson.svg'),bbox_inches='tight')
