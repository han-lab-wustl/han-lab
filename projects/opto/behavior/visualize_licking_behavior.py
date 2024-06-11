""""use binned licks to visualize average licking behavior 
align all rew locations to a common coordinate
"""
#%%
    
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
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

from behavior import get_success_failure_trials, get_performance, get_rewzones, get_behavior_tuning_curve, \
    get_lick_tuning_curves_per_trial

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior_licks.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing

lick_tc_vip_opto = {}
lick_tc_vip_ledoff = {}
lick_tc_ctrl_opto = {}
# com shift analysis
dcts = []
for dd,day in enumerate(conddf.days.values):
    if (conddf.in_type.values[dd]=='vip') and (conddf.optoep.values[dd]>1):
        animal = conddf.animals.values[dd]
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep = get_lick_tuning_curves_per_trial(params_pth, conddf, dd)
        lick_tc_vip_opto[f'rz{int(rewzone)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
    elif (conddf.in_type.values[dd]!='vip') and (conddf.optoep.values[dd]>1):
        animal = conddf.animals.values[dd]
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep = get_lick_tuning_curves_per_trial(params_pth, conddf, dd)
        lick_tc_ctrl_opto[f'rz{int(rewzone)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
    elif (conddf.in_type.values[dd]=='vip') and (conddf.optoep.values[dd]==-1):
        animal = conddf.animals.values[dd]
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep = get_lick_tuning_curves_per_trial(params_pth, conddf, dd)
        lick_tc_vip_ledoff[f'rz{int(rewzone)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded,trialstate_per_ep]
        
#%%
# trial by trial heatmap

def plot_lick_vis(dct_to_use, condition):
    rz1 = [67,86]
    rz2 = [101,120]
    rz3 = [135,155]
    bin_size = 3
    scalingf = 2/3
    fig, axes = plt.subplots(nrows=1,ncols=3,sharex=True)
    licks_opto_rz1 = [v[0] for k,v in dct_to_use.items() if k[:3] == 'rz1']
    licks_opto_rz1 = np.concatenate(licks_opto_rz1)
    trialstate_rz1 = [v[1] for k,v in dct_to_use.items() if k[:3] == 'rz1']
    trialstate_rz1 = np.concatenate(trialstate_rz1).astype(bool)
    df = pd.DataFrame(licks_opto_rz1)
    mask = np.array([[trial]*df.shape[1] for trial in trialstate_rz1])
    sns.heatmap(df, mask=mask, cmap='Reds',cbar=False, ax=axes[0])
    sns.heatmap(df, mask=~mask, cmap='Greys',cbar=False, ax=axes[0])
    axes[0].set_title('Reward zone 1')
    axes[0].set_ylabel('Trials')
    axes[0].set_xlabel('Position bin (3cm)')
    axes[0].add_patch(patches.Rectangle(
        xy=((rz1[0]/scalingf)/bin_size,0),  # point of origin.
        width=((rz1[1]/scalingf)-(rz1[0]/scalingf))/bin_size, 
        height=licks_opto_rz1.shape[0], linewidth=1, # width is s
        color='slategray', alpha=0.3))
    licks_opto_rz2 = [v[0] for k,v in dct_to_use.items() if k[:3] == 'rz2']
    licks_opto_rz2 = np.concatenate(licks_opto_rz2)
    trialstate_rz2 = [v[1] for k,v in dct_to_use.items() if k[:3] == 'rz2']
    trialstate_rz2 = np.concatenate(trialstate_rz2)
    df = pd.DataFrame(licks_opto_rz2)
    mask = np.array([[trial]*df.shape[1] for trial in trialstate_rz2])
    sns.heatmap(df, mask=mask, cmap='Reds',cbar=False, ax=axes[1])
    sns.heatmap(df, mask=~mask, cmap='Greys',cbar=False, ax=axes[1])
    axes[1].set_title('Reward zone 2')
    axes[1].add_patch(patches.Rectangle(
        xy=((rz2[0]/scalingf)/bin_size,0),  # point of origin.
        width=((rz2[1]/scalingf)-(rz2[0]/scalingf))/bin_size, 
        height=licks_opto_rz2.shape[0], linewidth=1, # width is s
        color='slategray', alpha=0.3))

    licks_opto_rz3 = [v[0] for k,v in dct_to_use.items() if k[:3] == 'rz3']
    licks_opto_rz3 = np.concatenate(licks_opto_rz3)
    trialstate_rz3 = [v[1] for k,v in dct_to_use.items() if k[:3] == 'rz3']
    trialstate_rz3 = np.concatenate(trialstate_rz3)
    df = pd.DataFrame(licks_opto_rz3)
    mask = np.array([[trial]*df.shape[1] for trial in trialstate_rz3])
    sns.heatmap(df, mask=mask, cmap='Reds',cbar=False, ax=axes[2])
    sns.heatmap(df, mask=~mask, cmap='Greys',cbar=False, ax=axes[2])
    axes[2].set_title(f'{condition}\nReward zone 3')
    axes[2].add_patch(patches.Rectangle(
        xy=((rz3[0]/scalingf)/bin_size,0),  # point of origin.
        width=((rz3[1]/scalingf)-(rz3[0]/scalingf))/bin_size, 
        height=licks_opto_rz3.shape[0], linewidth=1, # width is s
        color='slategray', alpha=0.3))


dct_to_use = lick_tc_ctrl_opto
condition = 'Control LED on'
plot_lick_vis(dct_to_use, condition)
dct_to_use = lick_tc_vip_opto
condition = 'VIP LED on'
plot_lick_vis(dct_to_use, condition)
dct_to_use = lick_tc_vip_ledoff
condition = 'VIP LED off'
plot_lick_vis(dct_to_use, condition)
