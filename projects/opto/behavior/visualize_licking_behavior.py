""""use binned licks to visualize average licking behavior 
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] =10
mpl.rcParams["ytick.major.size"] =10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import get_success_failure_trials, get_performance, get_rewzones, get_behavior_tuning_curve, \
    get_lick_tuning_curves_per_trial

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
bin_size = 1
lick_tc_vip_opto = {} # collecting
lick_tc_vip_ledoff = {}
lick_tc_ctrl_opto = {}

probe_lick_tc_vip_opto = {} # collecting
probe_lick_tc_vip_ledoff = {}
probe_lick_tc_ctrl_opto = {}
# conddf=conddf[conddf.animals.isin(['e201','z8'])]
conddf=conddf[~((conddf.animals=='z14')&(conddf.days<15))]
conddf=conddf[~((conddf.animals=='z17')&(conddf.days<13))]
conddf=conddf[~((conddf.animals=='z15')&(conddf.days<8))]
for dd,_ in enumerate(conddf.days.values):
    animal = conddf.animals.values[dd]
    day = conddf.days.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    if (conddf.in_type.values[dd]=='vip_ex') and (conddf.optoep.values[dd]>1):
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size)
        lick_tc_vip_opto[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size,probes=True)
        probe_lick_tc_vip_opto[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
    elif (conddf.in_type.values[dd]!='vip') and (conddf.optoep.values[dd]>1):
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size)
        lick_tc_ctrl_opto[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size,probes=True)
        probe_lick_tc_ctrl_opto[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
    elif (conddf.in_type.values[dd]=='vip') and (conddf.optoep.values[dd]==-1):
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size)
        lick_tc_vip_ledoff[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded,trialstate_per_ep]
        lick_tuning_curves_per_trial_per_ep_padded, rewzone, trialstate_per_ep, rewzone_prev = get_lick_tuning_curves_per_trial(params_pth, conddf, dd,
            bin_size=bin_size,probes=True)
        probe_lick_tc_vip_ledoff[f'rz{int(rewzone)}_{int(rewzone_prev)}_{dd}'] = [lick_tuning_curves_per_trial_per_ep_padded, trialstate_per_ep]
        
#%%
plt.rc('font', size=10)
# Function to plot and calculate average licks
def plot_lick_vis(dct_to_use, condition, savedst, probes=False,
            plot_all_probes=False, probe2plot=1,save=False,figsize=(10,10)):
    expandrzwindow = 0
    scalingf = 2/3
    rz1 = np.ceil(np.array([67-expandrzwindow,86+expandrzwindow])/scalingf)
    rz2 = np.ceil(np.array([101-expandrzwindow,120+expandrzwindow])/scalingf)
    rz3 = np.ceil(np.array([135-expandrzwindow,154+expandrzwindow])/scalingf)
    rzs = np.array([rz1,rz2,rz3]).astype(int)
    
    fig, axes = plt.subplots(nrows=1,ncols=3,sharex=True,figsize=figsize)
    
    av_licks_in_rewzone = {}
    av_licks_in_prevrewzone = {}
    
    for i, (zone, rz) in enumerate(zip(range(1, 4), rzs)):
        licks_opto = [v[0] for k,v in dct_to_use.items() if k[:3] == f'rz{zone}']
        prev_rz = [[int(k[4])]*len(v[0]) for k,v in dct_to_use.items() if k[:3] == f'rz{zone}']
        licks_opto = np.concatenate(licks_opto)
        trialstate = [v[1] for k,v in dct_to_use.items() if k[:3] == f'rz{zone}']
        trialstate = np.concatenate(trialstate).astype(bool)
        df = pd.DataFrame(licks_opto)
        mask = np.array([[trial]*df.shape[1] for trial in trialstate])
        # Normalize each row by its sum
        row_sums = np.nanmax(df, axis=1)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        df = df.div(row_sums, axis=0)

        # Calculate the endpoints for each block
        block_indices = []
        counter = 0
        for k, v in dct_to_use.items():
            if k[:3] == f'rz{zone}':
                block_indices.append(counter)
                counter += len(v[0])
        # Now you can draw lines at those borders in your plot
        ax = axes[i]  # for the corresponding zone
        for idx in block_indices[1:]:  # we omit the first, that's the start
            ax.axhline(y=idx, color='black', linestyle='--', linewidth=.5)
        axes[i].set_title(f'Reward zone {zone}')
        # Create a colormap starting with white and then blending into reds
        colors = ['white'] + list(sns.color_palette("Reds", 256))
        from matplotlib.colors import LinearSegmentedColormap
        new_cmap = LinearSegmentedColormap.from_list('Reds_mod', colors)
        if not probes:
            # Plot each row as a histogram
            sns.heatmap(df, mask=mask, cmap=new_cmap, cbar=False, ax=axes[i],rasterized=True)
            sns.heatmap(df, mask=~mask, cmap='Greys', cbar=False, ax=axes[i],rasterized=True)
            # Add the legend to the first axis
            grey_correct = patches.Patch(color='grey', label='Correct licks')
            red_incorrect = patches.Patch(color='red', label='Incorrect licks')
            green_rewzone = patches.Patch(color='mediumspringgreen', alpha=0.3, label='Current Reward Zone')
            grey_prevzone = patches.Patch(color='slategray', alpha=0.3, label='Previous Reward Zone')
            # Now add the legend to the first axis
            if i == 0:
                axes[0].legend(handles=[grey_correct, red_incorrect, green_rewzone, grey_prevzone],
                            loc='upper right')
        elif plot_all_probes:
            mask = np.ones_like(df);mask[::3] = 0 # first probe
            sns.heatmap(df, mask=mask, cmap='Greys', cbar=False, ax=axes[i],rasterized=True)
            mask = np.ones_like(df);mask[1::3] = 0 
            sns.heatmap(df, mask=mask, cmap='Blues', cbar=False, ax=axes[i],rasterized=True)
            mask = np.ones_like(df);mask[2::3] = 0 
            sns.heatmap(df, mask=mask, cmap='Purples', cbar=False, ax=axes[i],rasterized=True)
                # Add the legend to the first axis
            grey_correct = patches.Patch(color='k', label='Probe 1')
            red_incorrect = patches.Patch(color='blue', label='Probe 2')
            probe3 = patches.Patch(color='purple', label='Probe 3')
            green_rewzone = patches.Patch(color='mediumspringgreen', alpha=0.3, label='Current Reward Zone')
            # Now add the legend to the first axis
            if i == 0:
                axes[0].legend(handles=[grey_correct, red_incorrect, probe3,green_rewzone],
                            loc='upper right')

        elif (plot_all_probes==False) and (probes==True):
            df_ = df[(probe2plot-1)::3]
            sns.heatmap(df_, cmap='Greys', 
                    cbar=False, ax=axes[i],rasterized=True)
        if not probes:
            axes[i].add_patch(patches.Rectangle(
                xy=((rz[0])/bin_size,0),  # point of origin.
                width=((rz[1])/bin_size)-((rz[0])/bin_size),
                height=licks_opto.shape[0], linewidth=1, 
                color='mediumspringgreen', alpha=0.3))
            
        if i == 0:
            axes[i].set_ylabel('Trials')
        axes[i].set_xlabel('Position (cm)')
        axes[i].set_xticks([0, 135, 270])
        axes[i].set_yticks([0, len(df)//2, len(df)])
        axes[i].set_xticklabels([0, 135, 270])
        axes[i].set_yticklabels([0, len(df)//2,len(df)])

        av_licks_in_rewzones = []
        av_licks_in_prevrewzones = []
        for jj, przs in enumerate(prev_rz):
            if jj > 0:
                ystart = len(np.concatenate(prev_rz[:jj]))
            else:
                ystart = 0
            if not probes:
                axes[i].add_patch(patches.Rectangle(
                xy=((rzs[przs[0]-1][0])/bin_size,ystart),
                width=((rzs[przs[0]-1][1])/bin_size)-((rzs[przs[0]-1][0])/bin_size),
                height=len(przs), linewidth=1,
                color='slategray', alpha=0.3))
            else:
                axes[i].add_patch(patches.Rectangle(
                xy=((rzs[przs[0]-1][0])/bin_size,ystart),
                width=((rzs[przs[0]-1][1])/bin_size)-((rzs[przs[0]-1][0])/bin_size),
                height=len(przs), linewidth=1,
                color='mediumspringgreen', alpha=0.3))

                
            arr = np.array(df)[ystart:(ystart+len(przs)), (rz[0]//bin_size):(rz[1]//bin_size)]
            # mask, only fails
            # arr = arr[trialstate[ystart:(ystart+len(przs))],:]
            licks_in_rewzone = np.nansum(arr,axis=1)
            total_licks = np.nansum(np.array(df)[ystart:(ystart+len(przs)), :], axis=1)
            # total_licks = total_licks[trialstate[ystart:(ystart+len(przs))]]
            av_licks_in_rewzones.append(np.nanmean(licks_in_rewzone/total_licks))            
            # -1 for py indexing of rzs
            # mask, only fails
            arr = np.array(df)[ystart:(ystart+len(przs)), int(rzs[prev_rz[jj][0]-1][0]/bin_size):int(rzs[prev_rz[jj][0]-1][1]//bin_size)]
            # arr = arr[trialstate[ystart:(ystart+len(przs))],:]
            licks_in_prevrewzone = np.nansum(arr,axis=1)
            av_licks_in_prevrewzones.append(np.nanmean(licks_in_prevrewzone/total_licks))

        av_licks_in_rewzone[zone] = av_licks_in_rewzones
        av_licks_in_prevrewzone[zone] = av_licks_in_prevrewzones
    
    plt.savefig(os.path.join(savedst, condition+'.svg'),bbox_inches='tight')        
    if (plot_all_probes==False) and (probes==True):
        plt.suptitle(f'Probe {probe2plot}, {condition}')
    else:
        plt.suptitle(f'{condition}')
    plt.show()
    
    return av_licks_in_rewzone, av_licks_in_prevrewzone

#%%
# Plot and compute licks for each condition
dct_to_use = lick_tc_ctrl_opto
condition = 'Control LED on'

av_licks_in_rewzone_ctrl, av_licks_in_prevrewzone_ctrl = plot_lick_vis(dct_to_use, condition,savedst, plot_all_probes=True,save=True,figsize=(12,10))
#%%
dct_to_use = lick_tc_vip_opto
condition = 'VIP Ex LED on'
av_licks_in_rewzone_vip, av_licks_in_prevrewzone_vip = plot_lick_vis(dct_to_use,condition,savedst, save=True,fignsize=(12,10))

#%%
dct_to_use = probe_lick_tc_ctrl_opto
condition = 'Control LED on'
av_licks_in_rewzone_ctrl, av_licks_in_prevrewzone_ctrl = plot_lick_vis(dct_to_use, condition, savedst, probes=True, plot_all_probes=True, 
                probe2plot=3)
#%%
dct_to_use = probe_lick_tc_vip_opto
condition = 'VIP LED on'
av_licks_in_rewzone_vip, av_licks_in_prevrewzone_vip = plot_lick_vis(dct_to_use, condition,savedst,probes=True, plot_all_probes=True,figsize=(4,3))

#%%
prevrz1_ctrl = av_licks_in_prevrewzone_ctrl[3]
prevrz1_vip = av_licks_in_prevrewzone_vip[3]
# prevrz1_ctrl = av_licks_in_rewzone_ctrl[1]
# prevrz1_vip = av_licks_in_rewzone_vip[1]

df = pd.DataFrame(np.concatenate([prevrz1_ctrl,prevrz1_vip]), columns = ['prevrz_lick_prob'])
df['condition'] = np.concatenate([['Control']*len(prevrz1_ctrl),['VIP']*len(prevrz1_vip)])
dfagg = df
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='prevrz_lick_prob', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'Control': "slategray", 'VIP': "red"})
ax = sns.stripplot(x='condition', y='prevrz_lick_prob', hue='condition', data=dfagg,
                palette={'Control': "slategray", 'VIP': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
#%%
dct_to_use = lick_tc_vip_ledoff
condition = 'VIP LED off'
plot_lick_vis(dct_to_use, condition,savedst)

