"""get lick rate in old reward zone 
make lick tuning curve
get licks in old vs. new reward zone pos
trial by trial tuning curve
current vs previous licks per trial
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# plt.rc('font', size=20)          # controls default text sizes
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

def get_success_failure_trials(trialnum, reward):
   """
   Counts the number of success and failure trials.

   Parameters:
   trialnum : array-like, list of trial numbers
   reward : array-like, list indicating whether a reward was found (1) or not (0) for each trial

   Returns:
   success : int, number of successful trials
   fail : int, number of failed trials
   str : list, successful trial numbers
   ftr : list, failed trial numbers
   ttr : list, trial numbers excluding probes
   total_trials : int, total number of trials excluding probes
   """
   trialnum = np.array(trialnum)
   reward = np.array(reward)
   unique_trials = np.unique(trialnum)
   
   success = 0
   fail = 0
   str_trials = []  # success trials
   ftr_trials = []  # failure trials
   probe_trials = []

   for trial in unique_trials:
      if trial >= 3:  # Exclude probe trials
         trial_indices = trialnum == trial
         if np.any(reward[trial_indices] == 1):
               success += 1
               str_trials.append(trial)
         else:
               fail += 1
               ftr_trials.append(trial)
      else:
         probe_trials.append(trial)
   
   total_trials = np.sum(unique_trials)
   ttr = unique_trials  # trials excluding probes

   return success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials


from behavior import get_rewzones, smooth_lick_rate,get_behavior_tuning_curve

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
# exclude some animals and sessions
conddf=conddf[(conddf.animals!='e189') & (conddf.animals!='e186')]
# conddf=conddf[~((conddf.animals=='e218')&(conddf.days==55))]

# conddf=conddf[~((conddf.animals=='e201')&((conddf.days>62)))]
conddf=conddf[~((conddf.animals=='z14')&((conddf.days>33)&(conddf.days.isin([54]))))]
# conddf=conddf[~((conddf.animals=='e200')&((conddf.days<75)))]
conddf=conddf[~((conddf.animals=='z15')&((conddf.days.isin([15,16]))))]
conddf=conddf[~((conddf.animals=='z17')&(conddf.days.isin([3,4,22,])))]

bins=135
lick_selectivity = {} # collecting
for _,ii in enumerate(range(len(conddf))):
   animal = conddf.animals.values[ii]
   in_type = conddf.in_type.values[ii] if 'vip' in conddf.in_type.values[ii] else 'ctrl'             
   day = conddf.days.values[ii]
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
   fall = scipy.io.loadmat(params_pth, variable_names=['VR'])
   VR = fall['VR'][0][0][()]
   eps = np.where(np.hstack(VR['changeRewLoc']>0))[0]
   eps = np.append(eps, len(np.hstack(VR['changeRewLoc'])))
   scalingf = VR['scalingFACTOR'][0][0]
   track_length = 180/scalingf
   ybinned = np.hstack(VR['ypos']/scalingf)
   rewlocs = np.ceil(np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf).astype(int)
   rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
   trialnum = np.hstack(VR['trialNum'])
   rewards = np.hstack(VR['reward'])
   forwardvel = np.hstack(VR['ROE']); time =np.hstack(VR['time'])
   forwardvel=-0.013*forwardvel[1:]/np.diff(time) # make same size
   forwardvel = np.append(forwardvel, np.interp(len(forwardvel)+1, np.arange(len(forwardvel)),forwardvel))
   licks = np.hstack(VR['lick'])
   reward=(np.hstack(VR['reward'])==1).astype(int)
   eptest = conddf.optoep.values[ii]    
   if (eptest>1) or (in_type=='vip' and eptest==0): # ONLY OPTO SESSIONS
      print(conddf.optoep.values[ii])
      if conddf.optoep.values[ii]<2: 
         eptest = random.randint(2,3)   
         if len(eps)<4: eptest = 2 # if no 3 epochs 
      opto_ep = eptest
      # lick rate in reward zone (exclude consumption licks)
      time = np.hstack(VR['time'])
      dt = np.nanmedian(np.diff(time))
      lick_rate = smooth_lick_rate(licks,dt)
      lick_rate[forwardvel<2]=np.nan
      # lick_rate[lick_rate>6]=np.nan
      rzs = get_rewzones(rewlocs, 1/scalingf)
      # lick rate +/- 20 cm near new vs. old rew zone
      bound=40
      eprng = np.arange(eps[eptest-1],eps[eptest])
      trials = trialnum[eprng]
      trial_max = np.nanmax(trials)
      success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials=get_success_failure_trials(trials, reward[eprng])
         # last few trials
      if True:
         trials_keep = trials<8#(trial_max/2)
         # FIRST FEW TRIALS
         # trials_keep = trials<int(trial_max/3)
         #a ll trials
         trials_keep = np.ones_like(trials).astype(bool)
         # only incorrects
         # only first probe
         # probe_trials=[0,1]
         # ftr_trials=ftr_trials[:10]
         # trials_keep = np.array([True if xx in ftr_trials else False for xx in trials])
         lick_tc_per_trial=[]
         trial_state_per_trial=[]
         if np.sum(trials_keep)>0: # only if incorrect exists
            for tr in np.unique(trials[trials_keep]):
               mask = trials[trials_keep]==tr
               _,lick_tc = get_behavior_tuning_curve(ybinned[eprng][trials_keep][mask], lick_rate[eprng][trials_keep][mask], bins=bins)
               ypos=ybinned[eprng][trials_keep][mask]
               lick_tc_pad = np.zeros(bins)
               lick_tc_pad[:len(lick_tc)]=lick_tc.values
               # test
               # if ii%20==0:
               #     plt.figure()
               #     plt.plot(lick_tc.values)
               #     plt.axvline(rewlocs[eptest-1],color='b')
               #     plt.axvline(rewlocs[eptest-2],color='r')
               #     plt.title(f'{animal},{day}')
               # get lick tc before old v new
               lick_rate_old_all_tr=np.nanmean(lick_tc.values[int(rewlocs[eptest-2]-bound):int(rewlocs[eptest-2])])
               lick_rate_new_all_tr=np.nanmean(lick_tc.values[int(rewlocs[eptest-1]-bound):int(rewlocs[eptest-1])])
               lick_tc_per_trial.append(lick_tc_pad)
               if tr in str_trials: trial_state=np.ones_like(lick_tc_pad)
               elif tr in ftr_trials: trial_state=np.zeros_like(lick_tc_pad)
               else:
                  trial_state=np.ones_like(lick_tc_pad)*-1
               trial_state_per_trial.append(trial_state)
            lick_selectivity[f'{animal}_{day:03d}_{in_type}'] = [lick_rate_old_all_tr,lick_rate_new_all_tr,eptest, rzs,lick_tc_per_trial,rewlocs,trial_state_per_trial] 

#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

plt.rc('font', size=16)

transitions = [[2,3]]
vip_an = ['e217','e216','e218']
vip_ex = ['z15','z14','z17']
binsz = 2
ranges = [[80/binsz,140/binsz],[129/binsz,178/binsz],[180/binsz,240/binsz]]

# Trial type colormap
trial_colors = {1: 'green', 0: 'grey', -1: 'yellow'}
trial_labels = {1: 'Correct', 0: 'Incorrect', -1: 'Probe'}
trial_cmap = mcolors.ListedColormap([trial_colors[-1], trial_colors[0], trial_colors[1]])
trial_norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], trial_cmap.N)

for kk, tr in enumerate(transitions):
   rewloc_from, rewloc_to = tr

   lick_tcs = {
      'Inhibition': [(v[4], v[6]) for k,v in lick_selectivity.items()
                     if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and k.split('_')[0] in vip_an],
      'Excitation': [(v[4], v[6]) for k,v in lick_selectivity.items()
                     if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and k.split('_')[0] in vip_ex],
      'Control': [(v[4], v[6]) for k,v in lick_selectivity.items()
                  if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                  and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]
   }

   for cond, tcs in lick_tcs.items():
      for i_sess, (Z, trial_state) in enumerate(tcs):
         Z = np.array(Z, dtype=np.float64)
         Z = np.nan_to_num(Z, nan=0.0)[:-1,:]
         trial_state = np.array(trial_state, dtype=np.float64)[:-1,:]
         n_trials, n_bins = Z.shape
         fig, ax = plt.subplots(figsize=(4,3.5))
         # Lick rate heatmap
         im = ax.imshow(Z, aspect='auto', cmap='Blues')
         # Trial state heatmap overlay
         ax.imshow(trial_state, aspect='auto', cmap=trial_cmap, norm=trial_norm, alpha=0.5)
         # Overlay reward zones
         for zone_idx, color in zip([rewloc_from-1, rewloc_to-1], ['grey','black']):
               x_start, x_end = ranges[zone_idx]
               rect = patches.Rectangle(
                  (x_start, -0.5),
                  x_end-x_start,
                  n_trials,
                  linewidth=3,
                  edgecolor=color,
                  facecolor='none',
                  linestyle='--'
               )
               ax.add_patch(rect)

         ax.set_xlabel('Track position (cm)')
         ax.set_ylabel('Trial')
         ax.set_xticks([0, n_bins-1])
         ax.set_xticklabels([0, 270])
         ax.set_yticks([0, n_trials-1])
         ax.set_yticklabels([1, n_trials])
         ax.set_title(rf'{cond}, Area {rewloc_from}$\rightarrow${rewloc_to}')
         # Lick rate colorbar
         cbar = fig.colorbar(im, ax=ax, fraction=0.05)
         cbar.set_label('Lick rate (licks/s)')
         # Trial type legend
         from matplotlib.patches import Patch
         legend_elements = [Patch(facecolor=color, edgecolor='k', label=label) 
                              for label, color in trial_colors.items()]
         # ax.legend(handles=legend_elements, title="Trial type", loc='upper right')
         plt.tight_layout()
         plt.savefig(os.path.join(savedst, f'lick_rate_trial_by_trial_heatmap_trialstate_{cond}_{i_sess}_{tr[0]}_to_{tr[1]}.svg'),
                     bbox_inches='tight')
         plt.close()