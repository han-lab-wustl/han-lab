"""get lick rate in old reward zone 
make lick tuning curve
get licks in old vs. new reward zone pos
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
            bound=40
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
            # trials_keep = np.array([True if xx in ftr_trials else False for xx in trials])
            if np.sum(trials_keep)>0: # only if incorrect exists
                _,lick_tc = get_behavior_tuning_curve(ybinned[eprng][trials_keep], lick_rate[eprng][trials_keep], bins=270)
                ypos=ybinned[eprng][trials_keep]
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
                lick_selectivity[f'{animal}_{day:03d}_{in_type}'] = [lick_rate_old_all_tr,lick_rate_new_all_tr,eptest, rzs,lick_tc.values,rewlocs] 

#%%
# plot

# all tcs 
plt.rc('font', size=12)          # controls default text sizes
transitions = [[1,2],[1,3],[2,1],[2,3],[3,1],[3,2]]
fig,axes=plt.subplots(ncols=3,nrows=2,sharey=True,sharex=True,
        figsize=(6,4.5))
axes=axes.flatten()
for kk,tr in enumerate(transitions):
    rewloc_from=tr[0]
    rewloc_to=tr[1]
    lick_tcs = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)])
    # separate
    vip_an=['e217','e216','e218']
    vip_ex = ['z15','z14','z17']
    lick_tcs_inhib = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and k.split('_')[0] in vip_an])
    lick_tcs_excit = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and k.split('_')[0] in vip_ex])

    lick_tcs_ctrl = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)])
    ax=axes[kk]
    ax.plot(np.nanmean(lick_tcs_ctrl,axis=0),color='slategray',label=f'Control (n={lick_tcs_ctrl.shape[0]})')
    m=np.nanmean(lick_tcs_ctrl,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'), alpha=0.2,color='slategray')
    ax.plot(np.nanmean(lick_tcs_excit,axis=0),color='darkgoldenrod',label=f'VIP Excitation (n={lick_tcs_excit.shape[0]})')
    m=np.nanmean(lick_tcs_excit,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'), alpha=0.2,color='darkgoldenrod')
    ax.plot(np.nanmean(lick_tcs_inhib,axis=0),color='red',label=f'VIP Inhibition (n={lick_tcs_inhib.shape[0]})')
    m=np.nanmean(lick_tcs_inhib,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'), alpha=0.2,color='red')
    ax.spines[['top','right']].set_visible(False)

    import matplotlib.patches as patches
    # Example range: x=80 to x=120, full y-range of the plot
    ranges=[[80,129],[129,178],[180,231]]
    x_start, x_end = ranges[rewloc_to-1]
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle(
        (x_start, ymin),             # (x,y) lower-left corner
        x_end - x_start,             # width
        ymax - ymin,                 # height
        linewidth=1.5, edgecolor='black',
        facecolor='none', linestyle='--',label='Current reward area'
    )
    ax.add_patch(rect)
    x_start, x_end = ranges[rewloc_from-1]
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle(
        (x_start, ymin),             # (x,y) lower-left corner
        x_end - x_start,             # width
        ymax - ymin,                 # height
        linewidth=1.5, edgecolor='blue',
        facecolor='none', linestyle='--',label='Previous reward area'
    )
    ax.add_patch(rect)
    # ax.legend(fontsize=8)
    ax.set_title(rf'Reward area {rewloc_from}$\rightarrow${rewloc_to}')
    if kk==0: ax.set_ylabel('Lick rate (licks/s)')
    if kk==4: ax.set_xlabel('Track position (cm)')
    ax.set_xticks([0,270])
    # if kk==5: ax.axis('off')
fig.suptitle('Lick tuning, last 8 trials\nLED on epoch')
plt.tight_layout()
plt.savefig(os.path.join(savedst, f'lick_tuning_all_transitions.svg'), bbox_inches='tight')
#%%
# reward area ranges in bins
# ranges = [[80,129],[129,170],[180,231]]

def avg_lick_in_range(arr, start, end, sess_ids):
    """
    Compute average lick rate within [start:end] per trial,
    then average per session.
    arr: trials x bins
    sess_ids: list of session IDs matching arr rows
    """
    if arr.size == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        "sess": sess_ids,
        "val": np.nanmean(arr[:, start-20:start], axis=1)
    })
    return df.groupby("sess")["val"].mean().reset_index()

# storage
all_dfs = []

for kk, tr in enumerate(transitions):
    rewloc_from, rewloc_to = tr
    curr_start, curr_end = ranges[rewloc_to-1]
    prev_start, prev_end = ranges[rewloc_from-1]

    # ----- Control -----
    sess_ids_ctrl = [k.split('_')[0] for k,v in lick_selectivity.items()
                     if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]
    lick_tcs_ctrl = np.array([v[4] for k,v in lick_selectivity.items()
                     if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)])

    df_prev = avg_lick_in_range(lick_tcs_ctrl, prev_start, prev_end, sess_ids_ctrl)
    df_prev["condition"] = "Control"; df_prev["transition"] = f"{rewloc_from}->{rewloc_to}"; df_prev["zone"] = "Previous"
    df_curr = avg_lick_in_range(lick_tcs_ctrl, curr_start, curr_end, sess_ids_ctrl)
    df_curr["condition"] = "Control"; df_curr["transition"] = f"{rewloc_from}->{rewloc_to}"; df_curr["zone"] = "Current"
    all_dfs.extend([df_prev, df_curr])

    # ----- Excit -----
    sess_ids_ex = [k.split('_')[0] for k,v in lick_selectivity.items()
                   if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                   and k.split('_')[0] in vip_ex]
    lick_tcs_excit = np.array([v[4] for k,v in lick_selectivity.items()
                   if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                   and k.split('_')[0] in vip_ex])

    df_prev = avg_lick_in_range(lick_tcs_excit, prev_start, prev_end, sess_ids_ex)
    df_prev["condition"] = "VIP Excitation"; df_prev["transition"] = f"{rewloc_from}->{rewloc_to}"; df_prev["zone"] = "Previous"
    df_curr = avg_lick_in_range(lick_tcs_excit, curr_start, curr_end, sess_ids_ex)
    df_curr["condition"] = "VIP Excitation"; df_curr["transition"] = f"{rewloc_from}->{rewloc_to}"; df_curr["zone"] = "Current"
    all_dfs.extend([df_prev, df_curr])

    # ----- Inhib -----
    sess_ids_inhib = [k.split('_')[0] for k,v in lick_selectivity.items()
                      if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                      and k.split('_')[0] in vip_an]
    lick_tcs_inhib = np.array([v[4] for k,v in lick_selectivity.items()
                      if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                      and k.split('_')[0] in vip_an])

    df_prev = avg_lick_in_range(lick_tcs_inhib, prev_start, prev_end, sess_ids_inhib)
    df_prev["condition"] = "VIP Inhibition"; df_prev["transition"] = f"{rewloc_from}->{rewloc_to}"; df_prev["zone"] = "Previous"
    df_curr = avg_lick_in_range(lick_tcs_inhib, curr_start, curr_end, sess_ids_inhib)
    df_curr["condition"] = "VIP Inhibition"; df_curr["transition"] = f"{rewloc_from}->{rewloc_to}"; df_curr["zone"] = "Current"
    all_dfs.extend([df_prev, df_curr])
plt.rc('font', size=14)          # controls default text sizes

from scipy import stats
from statannotations.Annotator import Annotator

# Define group order and palette
order = ['Control','VIP Inhibition','VIP Excitation']
pl = {'Control': "slategray", 'VIP Inhibition': "red", 'VIP Excitation': 'darkgoldenrod'}

# Make figure
fig, axes = plt.subplots(1, 2, figsize=(4,3), sharey=True)

for i, zone in enumerate(['Current','Previous']):
    ax = axes[i]
    sub = df_all[df_all["zone"]==zone]

    sns.barplot(
        data=sub, x="zone", y="val", hue="condition",
        errorbar="se", palette=pl, hue_order=order, ax=ax, fill=False,legend=False
    )    
    sns.stripplot(
        data=sub, x="zone", y="val", hue="condition",dodge=True,
        palette=pl, hue_order=order, ax=ax, alpha=0.7
    )

    # Count unique sessions per group
    n_sessions = sub.groupby("condition")["val"].nunique()

    # Get actual bar positions from the containers
    for patch, cond in zip(ax.patches, order*1):  # one group of patches per hue
        # Only annotate first set (since 'zone' is constant)
        height = patch.get_height()
        xpos = patch.get_x() + patch.get_width()/2
        n = n_sessions.get(cond, 0)
        ax.text(xpos, height + 0.05, f"(n={n})",
                ha="center", va="bottom", fontsize=9)

    # Comparisons
    pairs = [
        (("Current", "Control"), ("Current", "VIP Inhibition")),
        (("Current", "Control"), ("Current", "VIP Excitation")),
    ]
    if zone=="Previous":
        pairs = [
            (("Previous", "Control"), ("Previous", "VIP Inhibition")),
            (("Previous", "Control"), ("Previous", "VIP Excitation")),
        ]

    annot = Annotator(ax, pairs, data=sub, x="zone", y="val", hue="condition", hue_order=order)
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction="fdr_bh")
    annot.apply_and_annotate()

    # ax.set_title(f"{zone} reward area")
    ax.set_ylabel("Lick rate (licks/s)" if i==0 else "")
    ax.set_xlabel("")
    ax.spines[['top','right']].set_visible(False)
    ax.get_legend().remove()

fig.tight_layout()
ax.legend()

plt.savefig(os.path.join(savedst, f'lick_tuning_all_transitions_quant.svg'), bbox_inches='tight')

#%%
#%%
# all tcs 
# PROBE
plt.rc('font', size=12)          # controls default text sizes
transitions = [[1,2],[1,3],[2,3],[3,2],[3,1]]
fig,axes=plt.subplots(ncols=2,nrows=3,sharey=True,sharex=True,figsize=(5,6))
axes=axes.flatten()
for kk,tr in enumerate(transitions):
    rewloc_from=tr[0]
    rewloc_to=tr[1]
    lick_tcs = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-2]==rewloc_to and v[3][eptest-3]==rewloc_from)])
    # separate
    vip_an=['e217','e216','e218']
    vip_ex = ['z15','z14','z17']
    lick_tcs_inhib = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-2]==rewloc_to and v[3][eptest-3]==rewloc_from) and k.split('_')[0] in vip_an])
    lick_tcs_excit = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-2]==rewloc_to and v[3][eptest-3]==rewloc_from) and k.split('_')[0] in vip_ex])

    lick_tcs_ctrl = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-2]==rewloc_to and v[3][eptest-3]==rewloc_from) and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)])
    ax=axes[kk]
    ax.plot(np.nanmean(lick_tcs_ctrl,axis=0),color='slategray',label=f'Control (n={lick_tcs_ctrl.shape[0]})')
    m=np.nanmean(lick_tcs_ctrl,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'), alpha=0.2,color='slategray')
    ax.plot(np.nanmean(lick_tcs_excit,axis=0),color='darkgoldenrod',label=f'VIP Excitation (n={lick_tcs_excit.shape[0]})')
    m=np.nanmean(lick_tcs_excit,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'), alpha=0.2,color='darkgoldenrod')
    ax.plot(np.nanmean(lick_tcs_inhib,axis=0),color='red',label=f'VIP Inhibition (n={lick_tcs_inhib.shape[0]})')
    m=np.nanmean(lick_tcs_inhib,axis=0)
    ax.fill_between(range(0,270), 
        m-scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'), alpha=0.2,color='red')
    ax.spines[['top','right']].set_visible(False)

    import matplotlib.patches as patches
    # Example range: x=80 to x=120, full y-range of the plot
    ranges=[[80,129],[129,170],[180,231]]
    x_start, x_end = ranges[rewloc_to-1]
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle(
        (x_start, ymin),             # (x,y) lower-left corner
        x_end - x_start,             # width
        ymax - ymin,                 # height
        linewidth=1.5, edgecolor='black',
        facecolor='none', linestyle='--',label='Current reward area'
    )
    ax.add_patch(rect)
    x_start, x_end = ranges[rewloc_from-1]
    ymin, ymax = ax.get_ylim()
    rect = patches.Rectangle(
        (x_start, ymin),             # (x,y) lower-left corner
        x_end - x_start,             # width
        ymax - ymin,                 # height
        linewidth=1.5, edgecolor='blue',
        facecolor='none', linestyle='--',label='Previous reward area'
    )
    ax.add_patch(rect)
    ax.legend(fontsize=8)
    ax.set_title(rf'Reward area {rewloc_from} $ \rightarrow$ {rewloc_to}')
    if kk==0: ax.set_ylabel('Lick rate (licks/s)')
    if kk==4: ax.set_xlabel('Track position (cm)')
    ax.set_xticks([0,270])
    # if kk==5: ax.axis('off')
fig.suptitle('Probe trials')
plt.tight_layout()
plt.savefig(os.path.join(savedst, f'probe_lick_tuning_all_transitions.svg'), bbox_inches='tight')

#%%
df = pd.DataFrame()

df['lick_rate_old_rew_zone'] = np.array([v[0] for k,v in lick_selectivity.items()])
df['lick_rate_new_rew_zone'] = np.array([v[1] for k,v in lick_selectivity.items()])
df['rz_transition']= np.array([f'{int(v[3][v[2]-2])}_{int(v[3][v[2]-1])}' for k,v in lick_selectivity.items()])
df['animals'] = [k.split('_')[0] for k,v in lick_selectivity.items()]
df['days'] = [int(k.split('_')[1]) for k,v in lick_selectivity.items()]
df = pd.merge(df, conddf, on=['animals', 'days'], how='inner')
df = df[(df.animals!='e189')&(df.animals!='e190')]
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
df['opto']=df.optoep>1
# ratio
df['lick_rate_old_new'] = df['lick_rate_old_rew_zone']/df['lick_rate_new_rew_zone']

fig,ax=plt.subplots(figsize=(5,6))
# df=df[df.opto==True]
# df=df[(df.rz_transition=='3_1')|(df.rz_transition=='2_1')]
var= 'lick_rate_old_new'
dfagg = df#.groupby(['animals', 'opto', 'condition']).mean(numeric_only = True)
dfagg=dfagg[dfagg[var]<15]

# drop 1st row
dfagg = dfagg.sort_values('opto')
order=['ctrl','vip','vip_ex']
pl = {'ctrl': "slategray", 'vip': "red", 'vip_ex': 'darkgoldenrod'}
sns.barplot(x='opto', y=var, hue='condition', data=dfagg, fill=False,hue_order=order,ax=ax,errorbar='se',
                palette=pl)
# sns.stripplot(x='opto', y=var, hue='condition', data=dfagg,dodge=True,hue_order=order,ax=ax,palette=pl,s=5)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)

# var= 'lick_rate_new_rew_zone'
# # dfagg = df#.groupby(['animals', 'opto', 'condition']).mean(numeric_only = True)
# # dfagg=dfagg[dfagg[var]<15]
# ax=axes[0]
# sns.barplot(x='opto', y=var, hue='condition', data=dfagg, fill=False,hue_order=order,ax=ax,errorbar='se',palette=pl)
# # sns.stripplot(x='opto', y=var, hue='condition', data=dfagg,dodge=True,hue_order=order,ax=ax,palette=pl,s=5)
# ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
                            
# plt.savefig(os.path.join(savedst, 'lick_rate_first_5_trials.svg'), bbox_inches='tight')


