"""
lick selectivity across trials
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
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


from behavior import get_rewzones, smooth_lick_rate,get_behavior_tuning_curve,get_lick_selectivity

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
# exclude some animals and sessions
conddf=conddf[(conddf.animals!='e189') & (conddf.animals!='e190')]
conddf=conddf[~((conddf.animals=='e201')&((conddf.days>62)))]

lick_selectivity = {} # collecting
for _,ii in enumerate(range(len(conddf))):
   animal = conddf.animals.values[ii]
   in_type = conddf.in_type.values[ii] if 'vip' in conddf.in_type.values[ii] else 'ctrl'             
   day = conddf.days.values[ii]
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
   print(params_pth)
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
      if True:#abs(rzs[eptest-1]-rzs[eptest-2])==2: # only far to near/near to far conditions
         eprng = np.arange(eps[eptest-1],eps[eptest])
         trials = trialnum[eprng]
         trial_max = np.nanmax(trials)
         success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials=get_success_failure_trials(trials, reward[eprng])
         # last few trials
         trials_keep = trials>(trial_max-8)
         # only incorrects
         # only first probe
         # probe_trials=[0,1]
         # ftr_trials=ftr_trials[:5]
         trials_keep = np.array([True if xx in str_trials else False for xx in trials])
         if np.sum(trials_keep)>0: # only if incorrect exists
               ls_per_trial_opto = get_lick_selectivity(ybinned[eprng][trials_keep], trials[trials_keep], licks[eprng][trials_keep], rewlocs[eptest-1], rewsize)
         else:
               ls_per_trial_opto=[np.nan]
               
         
         eprng = np.arange(eps[eptest-2],eps[eptest-1])
         trials = trialnum[eprng]
         trial_max = np.nanmax(trials)
         success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials=get_success_failure_trials(trials, reward[eprng])
         # last few trials
         trials_keep = trials>(trial_max-8)
         # only incorrects
         # only first probe
         # probe_trials=[0,1]
         # ftr_trials=ftr_trials[:5]
         trials_keep = np.array([True if xx in str_trials else False for xx in trials])
         if np.sum(trials_keep)>0: # only if incorrect exists
               ls_per_trial_prev = get_lick_selectivity(ybinned[eprng][trials_keep], trials[trials_keep], licks[eprng][trials_keep], rewlocs[eptest-2], rewsize)
         else:
               ls_per_trial_prev=[np.nan]
         lick_selectivity[f'{animal}_{day:03d}_{in_type}'] = [ls_per_trial_prev,ls_per_trial_opto]

#%%
# plot
plt.rc('font', size=16)          # controls default text sizes
lick_tcs_lst = [v for k,v in lick_selectivity.items()]
# max 21 correct trials
vip_an=['e217','e216','e218']
vip_ex = ['z15','z14','z17']
cond=['off','on']
fig,axes=plt.subplots(nrows=2,figsize=(3.5,5.5),sharex=True,sharey=True)
for i in range(2):
    ax=axes[i]
    lick_tcs_inhib_ = [v[i] for k,v in lick_selectivity.items() if k.split('_')[0] in vip_an]
    lick_tcs_excit_ = [v[i] for k,v in lick_selectivity.items() if k.split('_')[0] in vip_ex]
    lick_tcs_ctrl_ = [v[i] for k,v in lick_selectivity.items() if (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]

    lick_tcs_inhib=np.ones((len(lick_tcs_inhib_),21))*np.nan
    for ll,ls in enumerate(lick_tcs_inhib_):
        lick_tcs_inhib[ll,:len(ls)]=ls
    lick_tcs_excit=np.ones((len(lick_tcs_excit_),21))*np.nan
    for ll,ls in enumerate(lick_tcs_excit_):
        lick_tcs_excit[ll,:len(ls)]=ls
    lick_tcs_ctrl=np.ones((len(lick_tcs_ctrl_),21))*np.nan
    for ll,ls in enumerate(lick_tcs_ctrl_):
        lick_tcs_ctrl[ll,:len(ls)]=ls

    ax.plot(np.nanmean(lick_tcs_ctrl,axis=0),color='slategray',label='Control')
    m=np.nanmean(lick_tcs_ctrl,axis=0)
    ax.fill_between(range(0,21), 
        m-scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'), alpha=0.2,color='slategray')
    ax.plot(np.nanmean(lick_tcs_inhib,axis=0),color='red',label='VIP Inhibition')
    m=np.nanmean(lick_tcs_inhib,axis=0)
    ax.fill_between(range(0,21), 
        m-scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'), alpha=0.2,color='r')
    ax.plot(np.nanmean(lick_tcs_excit,axis=0),color='darkgoldenrod',label='VIP Excitation')
    m=np.nanmean(lick_tcs_excit,axis=0)
    ax.fill_between(range(0,21), 
        m-scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'), alpha=0.2,color='darkgoldenrod')
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([1,11,21])
    ax.spines[['top','right']].set_visible(False)    
    ax.axhline(0.8,color='k',linestyle='--')
    if i==0: 
        ax.set_ylabel('Lick selectivity')
        ax.set_xlabel('# correct trials across epoch')
    ax.set_title(f'LED {cond[i]} epoch')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'lick_selectivity_across_epoch.svg'), bbox_inches='tight')
#%%
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy import stats

plt.rc('font', size=16)

vip_an = ['e217','e216','e218']
vip_ex = ['z15','z14','z17']
cond = ['off','on']
all_dfs = []
triallim=8
for i in range(2):
    # Extract per-condition
    lick_tcs_inhib_ = [v[i] for k,v in lick_selectivity.items() if k.split('_')[0] in vip_an]
    lick_tcs_excit_ = [v[i] for k,v in lick_selectivity.items() if k.split('_')[0] in vip_ex]
    lick_tcs_ctrl_  = [v[i] for k,v in lick_selectivity.items() if (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]

    # Pad to 21 trials
    def pad(arrs):
        out = np.ones((len(arrs),21))*np.nan
        for ll,ls in enumerate(arrs):
            out[ll,:len(ls)] = ls
        return out
    
    lick_tcs_inhib = pad(lick_tcs_inhib_)
    lick_tcs_excit = pad(lick_tcs_excit_)
    lick_tcs_ctrl  = pad(lick_tcs_ctrl_)

    # Take average of last 10 trials
    inhib_last10 = np.nanmean(lick_tcs_inhib[:, -triallim:], axis=1)
    excit_last10 = np.nanmean(lick_tcs_excit[:, -triallim:], axis=1)
    ctrl_last10  = np.nanmean(lick_tcs_ctrl[:, -triallim:], axis=1)

    # Build dataframe
    df = pd.DataFrame({
        "val": np.concatenate([ctrl_last10, inhib_last10, excit_last10]),
        "condition": (["Control"]*len(ctrl_last10) +
                      ["VIP Inhibition"]*len(inhib_last10) +
                      ["VIP Excitation"]*len(excit_last10)),
        "epoch": cond[i]
    })
    all_dfs.append(df)

# Concatenate across LED off/on
df_all = pd.concat(all_dfs, ignore_index=True)

# ---- PLOT ----
order = ["Control","VIP Inhibition","VIP Excitation"]
palette = {"Control":"slategray","VIP Inhibition":"red","VIP Excitation":"darkgoldenrod"}

fig, axes = plt.subplots(2,1, figsize=(3,6), sharey=True,sharex=True)

for i, epoch in enumerate(cond):
    ax = axes[i]
    sub = df_all[df_all["epoch"]==epoch]
    sns.barplot(data=sub, x="condition", y="val", order=order,
                palette=palette, errorbar="se", ax=ax, fill=False)
    sns.stripplot(data=sub, x="condition", y="val", order=order,s=4,
                  palette=palette, ax=ax, alpha=0.7, jitter=True)
    # Annotate n above each bar
    n_sessions = sub.groupby("condition").size()
    ymax = sub["val"].max()
    for j, cond_name in enumerate(order):
        if cond_name in n_sessions:
            n = n_sessions[cond_name]
            y = sub[sub["condition"]==cond_name]["val"].mean()
            ax.text(j, y + 0.05*ymax, f"(n={n})",
                    ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Lick selectivity")
    ax.set_xlabel("")
    ax.set_title(f"LED {epoch} epoch")
    ax.spines[['top','right']].set_visible(False)
    ax.set_xticklabels(["Control","VIP\nInhibition","VIP\nExcitation"],rotation=20)
    # Stats only for LED on
    if epoch == "on":
        pairs = [("Control","VIP Inhibition"),
                 ("Control","VIP Excitation")]
        annot = Annotator(ax, pairs, data=sub, x="condition", y="val", order=order)
        annot.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction="fdr_bh")
        annot.apply_and_annotate()
fig.suptitle('Last 8 trials')
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'lick_selectivity_across_epoch_quant.svg'), bbox_inches='tight')
