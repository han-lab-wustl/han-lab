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
            # FIRST FEW TRIALS
            # trials_keep = (trials<11) & (trials>2)
            # LAST FEW TRIALS
            trials_keep = (trials<trial_max-10)
            #a ll trials
            # trials_keep = np.ones_like(trials).astype(bool)
            # only incorrects
            # only first probe
            # probe_trials=[0,1]
            # ftr_trials=ftr_trials[:10]
            # trials_keep = np.array([True if xx in ftr_trials else False for xx in trials])
            if np.sum(trials_keep)>0: # only if incorrect exists
                _,lick_tc = get_behavior_tuning_curve(ybinned[eprng][trials_keep], lick_rate[eprng][trials_keep], bins=bins)
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
# all trials
plt.rc('font', size=12)          # controls default text sizes
transitions = [[1,2],[1,3],[2,3],[3,1]]
fig,axes=plt.subplots(ncols=2,nrows=2,sharey=True,sharex=True,
        figsize=(5,4))
axes=axes.flatten()
for kk,tr in enumerate(transitions):
    rewloc_from=tr[0]
    rewloc_to=tr[1]
    lick_tcs = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==270 and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)])
    # separate
    vip_an=['e217','e216','e218']
    vip_ex = ['z15','z14','z17']
    lick_tcs_inhib = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and k.split('_')[0] in vip_an])
    lick_tcs_excit = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and k.split('_')[0] in vip_ex])

    lick_tcs_ctrl = np.array([v[4] for k,v in lick_selectivity.items() if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from) and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)])
    ax=axes[kk]
    ax.plot(np.nanmean(lick_tcs_ctrl,axis=0),color='slategray',label=f'Control (n={lick_tcs_ctrl.shape[0]})')
    m=np.nanmean(lick_tcs_ctrl,axis=0)
    ax.fill_between(range(0,bins), 
        m-scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_ctrl,axis=0,nan_policy='omit'), alpha=0.2,color='slategray')
    ax.plot(np.nanmean(lick_tcs_excit,axis=0),color='darkgoldenrod',label=f'VIP Excitation (n={lick_tcs_excit.shape[0]})')
    m=np.nanmean(lick_tcs_excit,axis=0)
    ax.fill_between(range(0,bins), 
        m-scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_excit,axis=0,nan_policy='omit'), alpha=0.2,color='darkgoldenrod')
    ax.plot(np.nanmean(lick_tcs_inhib,axis=0),color='red',label=f'VIP Inhibition (n={lick_tcs_inhib.shape[0]})')
    m=np.nanmean(lick_tcs_inhib,axis=0)
    ax.fill_between(range(0,bins), 
        m-scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'),
        m+scipy.stats.sem(lick_tcs_inhib,axis=0,nan_policy='omit'), alpha=0.2,color='red')
    ax.spines[['top','right']].set_visible(False)

    import matplotlib.patches as patches
    # Example range: x=80 to x=120, full y-range of the plot
    binsz=2
    ranges=[[80/binsz,129/binsz],[129/binsz,178/binsz],[180/binsz,231/binsz]]
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
    ax.set_title(rf'Area {rewloc_from}$\rightarrow${rewloc_to}')
    if kk==0: ax.set_ylabel('Lick rate (licks/s)')
    if kk==4: ax.set_xlabel('Track position (cm)')
    # ax.set_xticks([0,bins])
    # if kk==5: ax.axis('off')
# ax.legend(fontsize=8)
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
    start, end=int(start), int(end)
    if arr.size == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        "sess": sess_ids,
        "val": np.nanmean(arr[:, start-10:start], axis=1)
    })
    return df.groupby("sess")["val"].mean().reset_index()

# storage
all_dfs = []

for kk, tr in enumerate(transitions):
    rewloc_from, rewloc_to = tr
    curr_start, curr_end = ranges[rewloc_to-1]
    prev_start, prev_end = ranges[rewloc_from-1]

    # ----- Control -----
    sess_ids_ctrl = [k for k,v in lick_selectivity.items()
                     if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)]
    lick_tcs_ctrl = np.array([v[4] for k,v in lick_selectivity.items()
                     if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                     and (k.split('_')[0] not in vip_ex) and (k.split('_')[0] not in vip_an)])

    df_prev = avg_lick_in_range(lick_tcs_ctrl, prev_start, prev_end, sess_ids_ctrl)
    df_prev["condition"] = "Control"; df_prev["transition"] = f"{rewloc_from}->{rewloc_to}"; df_prev["zone"] = "Previous"
    df_curr = avg_lick_in_range(lick_tcs_ctrl, curr_start, curr_end, sess_ids_ctrl)
    df_curr["condition"] = "Control"; df_curr["transition"] = f"{rewloc_from}->{rewloc_to}"; df_curr["zone"] = "Current"
    all_dfs.extend([df_prev, df_curr])

    # ----- Excit -----
    sess_ids_ex = [k for k,v in lick_selectivity.items()
                   if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                   and k.split('_')[0] in vip_ex]
    lick_tcs_excit = np.array([v[4] for k,v in lick_selectivity.items()
                   if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                   and k.split('_')[0] in vip_ex])

    df_prev = avg_lick_in_range(lick_tcs_excit, prev_start, prev_end, sess_ids_ex)
    df_prev["condition"] = "VIP Excitation"; df_prev["transition"] = f"{rewloc_from}->{rewloc_to}"; df_prev["zone"] = "Previous"
    df_curr = avg_lick_in_range(lick_tcs_excit, curr_start, curr_end, sess_ids_ex)
    df_curr["condition"] = "VIP Excitation"; df_curr["transition"] = f"{rewloc_from}->{rewloc_to}"; df_curr["zone"] = "Current"
    all_dfs.extend([df_prev, df_curr])

    # ----- Inhib -----
    sess_ids_inhib = [k for k,v in lick_selectivity.items()
                      if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
                      and k.split('_')[0] in vip_an]
    lick_tcs_inhib = np.array([v[4] for k,v in lick_selectivity.items()
                      if len(v[4])==bins and (v[3][eptest-1]==rewloc_to and v[3][eptest-2]==rewloc_from)
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
df_all=pd.concat(all_dfs)

# Merge with original
df_plot = pd.concat([df_all])
df_plot=df_plot[df_plot.val<10]
# Plot all three

# Get hue offsets (where seaborn puts the stripplot points)
n_hue = df_plot["zone"].nunique()
dodge_amount = 0.4  # adjust if needed

from statsmodels.stats.multitest import multipletests

# Store stats
stats_list = []

# Comparisons: Previous vs Current per condition
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    # Pivot to have Previous and Current in columns per session
    pivot = sub.pivot(index="sess", columns="zone", values="val").dropna()
    if pivot.shape[0] == 0:
        continue
    # Paired Wilcoxon test
    stat, p = stats.wilcoxon(pivot["Previous"], pivot["Current"])
    stats_list.append({
        "Condition": cond,
        "Test": "Wilcoxon (paired)",
        "Statistic": stat,
        "p-value": p
    })

# Correct for multiple comparisons
pvals = [d['p-value'] for d in stats_list]
rej, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
for i, d in enumerate(stats_list):
    d['p-corrected'] = pvals_corrected[i]
    d['Significant'] = rej[i]

# Convert to DataFrame
df_stats = pd.DataFrame(stats_list)
print(df_stats)

# ----------------- Plot with bars -----------------
fig, ax = plt.subplots(figsize=(3.5,3))
sns.barplot(
    data=df_plot, x="condition", y="val", hue="zone",
    order=order, errorbar='se', palette=['k','royalblue'],
    ax=ax, fill=False
)

# Add connecting lines per session
dodge_amount = 0.4
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    for sess in sub["sess"].unique():
        sdat = sub[sub["sess"] == sess]
        if {"Previous", "Current"} <= set(sdat["zone"]):
            prev_y = sdat[sdat["zone"]=="Previous"]["val"].values[0]
            curr_y = sdat[sdat["zone"]=="Current"]["val"].values[0]

            xpos = order.index(cond)
            prev_x = xpos - dodge_amount/2
            curr_x = xpos + dodge_amount/2
            ax.plot([prev_x, curr_x], [prev_y, curr_y],
                    color="gray", alpha=0.5, linewidth=.8, zorder=0)

# Add significance asterisks
for i, row in df_stats.iterrows():
    xpos = order.index(row['Condition'])
    y_max = df_plot[df_plot["condition"]==row['Condition']]['val'].max()-1
    if row['Significant']:
        ax.text(xpos, y_max, "*", ha='center', va='bottom', fontsize=30)


########
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import numpy as np

# --- Settings ---
order = ['Control', 'VIP Inhibition', 'VIP Excitation']
zone_colors = ['k', 'royalblue']
dodge_amount = 0.4
bar_height_offset = 0.3  # offset above bars for comparison lines

combs=list(combinations(order, 2))[:2]
# --- Compute pairwise comparisons per zone ---
stats_list = []
for zone in ['Previous', 'Current']:
    sub_zone = df_plot[df_plot['zone'] == zone]
    for cond1, cond2 in combs:
        df1 = sub_zone[sub_zone['condition'] == cond1].set_index('sess')['val']
        df2 = sub_zone[sub_zone['condition'] == cond2].set_index('sess')['val']
        stat, p = stats.ranksums(df1, df2)
        stats_list.append({
            "Zone": zone,
            "Comparison": f"{cond1} vs {cond2}",
            "Statistic": stat,
            "p-value": p,
            "cond1": cond1,
            "cond2": cond2
        })

# FDR correction across all comparisons
pvals = [d['p-value'] for d in stats_list]
rej, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
for i, d in enumerate(stats_list):
    d['p-corrected'] = pvals_corrected[i]
    d['Significant'] = rej[i]

df_stats = pd.DataFrame(stats_list)
print(df_stats)

# --- Add comparison bars with asterisks ---
y_offsets = {}  # track top of each zone/cond for stacking multiple bars
for idx, row in df_stats.iterrows():
    cond1 = row['cond1']
    cond2 = row['cond2']
    zone = row['Zone']

    # X positions for bars
    x1 = order.index(cond1)
    x2 = order.index(cond2)

    # Y positions: get max bar height in this zone/cond
    y1 = df_plot[(df_plot['condition']==cond1) & (df_plot['zone']==zone)]['val'].max()
    y2 = df_plot[(df_plot['condition']==cond2) & (df_plot['zone']==zone)]['val'].max()
    y_base = max(y1, y2)
    bracket_height = 0.5  # vertical size of the bracket
    line_width = 1

    # Coordinates for the bracket
    x_coords = [x1, x1, x2, x2]
    y_coords = [y_bar, y_bar+bracket_height, y_bar+bracket_height, y_bar]

    # Apply stacking offset if multiple bars for same zone
    key = (zone, x1, x2)
    if key in y_offsets:
        y_bar = y_offsets[key] + 0.15
    else:
        y_bar = y_base + bar_height_offset
    y_offsets[key] = y_bar

    if row['Significant']:
        # Horizontal line
        # Draw the bracket
        ax.plot(x_coords, y_coords, color='k', lw=line_width)
        # Asterisk
        ax.text((x1+x2)/2, y_bar + 0.05, "*", ha='center', va='bottom', fontsize=30)


# Clean up
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel("Lick rate (licks/s)")
ax.set_xlabel("")
ax.legend(title="Condition")
fig.suptitle('Last 8 trials')
plt.savefig(os.path.join(savedst, f'lick_tuning_all_transitions_quant_last8.svg'), bbox_inches='tight')
#%%
