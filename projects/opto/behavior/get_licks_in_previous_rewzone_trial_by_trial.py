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
            lick_selectivity[f'{animal}_{day:03d}_{in_type}'] = [lick_rate_old_all_tr,lick_rate_new_all_tr,eptest, rzs,lick_tc_per_trial,rewlocs] 

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rc('font', size=14)

transitions = [[3,1]]
vip_an = ['e217','e216','e218']
vip_ex = ['z15','z14','z17']
binsz = 2
ranges = [[80/binsz,129/binsz],[129/binsz,178/binsz],[180/binsz,231/binsz]]

for kk, tr in enumerate(transitions):
    rewloc_from, rewloc_to = tr

    lick_tcs = {
        'Inhibition': [v[4] for k,v in lick_selectivity.items()
                       if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                       and k.split('_')[0] in vip_an],
        'Excitation': [v[4] for k,v in lick_selectivity.items()
                       if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                       and k.split('_')[0] in vip_ex],
        'Control': [v[4] for k,v in lick_selectivity.items()
                    if (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                    and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]
    }

    for cond, tcs in lick_tcs.items():
        for i_sess, Z in enumerate(tcs):
            Z = np.array(Z, dtype=np.float64)
            if np.isnan(Z).any():
               Z = np.nan_to_num(Z, nan=0.0)  # replace NaN with 0
            else:
               Z = Z
            n_trials, n_bins = Z.shape
            X, Y = np.meshgrid(np.arange(n_bins), np.arange(n_trials))

            fig = plt.figure(figsize=(4,5))
            ax = fig.add_subplot(111, projection='3d')

            # Surface plot
            cmap = {'Inhibition':'Reds', 'Excitation':'YlOrBr', 'Control':'Greys'}
            surf = ax.plot_surface(X, Y, Z, cmap=cmap[cond], edgecolor='k', linewidth=0.3, zorder=0)

            # Set limits to make sure rectangles are visible
            ax.set_xlim(0, n_bins-1)
            ax.set_ylim(0, n_trials-1)
            ax.set_zlim(Z.min(), Z.max()+0.5)

            # Add reward zone rectangles
            for zone_idx, color in zip([rewloc_from-1, rewloc_to-1], ['blue','black']):
                x_start, x_end = ranges[zone_idx]
                y_start, y_end = 0, n_trials-1
                z_start, z_end = Z.min(), Z.max() + 0.5
                # 6 faces of a box
                verts = [
                    [[x_start, y_start, z_start],[x_end, y_start, z_start],[x_end, y_end, z_start],[x_start, y_end, z_start]], # bottom
                    [[x_start, y_start, z_end],[x_end, y_start, z_end],[x_end, y_end, z_end],[x_start, y_end, z_end]], # top
                    [[x_start, y_start, z_start],[x_start, y_start, z_end],[x_start, y_end, z_end],[x_start, y_end, z_start]], # left
                    [[x_end, y_start, z_start],[x_end, y_start, z_end],[x_end, y_end, z_end],[x_end, y_end, z_start]], # right
                    [[x_start, y_start, z_start],[x_start, y_start, z_end],[x_end, y_start, z_end],[x_end, y_start, z_start]], # front
                    [[x_start, y_end, z_start],[x_start, y_end, z_end],[x_end, y_end, z_end],[x_end, y_end, z_start]] # back
                ]
                for v in verts:
                    poly = Poly3DCollection([v], facecolor=color, alpha=0.2, zorder=1)
                    ax.add_collection3d(poly)

            ax.set_xlabel('Track position (cm)')
            ax.set_xticks([0,135])
            ax.set_xticklabels([0,270])
            ax.set_yticks([0,n_trials-1])
            ax.set_yticklabels([1,n_trials])
            ax.set_ylabel('Trial')
            ax.set_zlabel('Lick rate')
            ax.set_title(f'{cond} - Transition {rewloc_from}->{rewloc_to} - Session {i_sess+1}')
            plt.tight_layout()
            plt.savefig(os.path.join(savedst, f'lick_rate_trial_by_trial_{cond}_{i_sess}_{tr[0]}_to_{tr[1]}.svg'), bbox_inches='tight')
#%%
