"""
zahra
get trial by trial heatmap of rew cells
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['gray', 'red', 'lime'])  # -1 = gray, 0 = red, 1 = green
bounds = [-1.5, -0.5, 0.5, 1.5]  # boundaries between categories
norm = BoundaryNorm(bounds, cmap.N)
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'trial_by_trial_tuning_w_com.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
bins = 150
goal_cm_window=20
dfs = []; lick_dfs = []
lowcountan=['e217','z17','z14','e200']
# iis=[130,166,49] # control v inhib x ex
for ii in range(len(conddf)):
        # ii=34 # z17
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        in_type = conddf.in_type.values[ii]
        optoep = conddf.optoep.values[ii]
        if 'vip' in in_type and ii!=202:
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                print(params_pth)
                fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
                        'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                        'stat', 'licks'])
                VR = fall['VR'][0][0][()]
                scalingf = VR['scalingFACTOR'][0][0]
                try:
                        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
                except:
                        rewsize = 10
                ybinned = fall['ybinned'][0]/scalingf
                track_length=180/scalingf    
                forwardvel = fall['forwardvel'][0]    
                changeRewLoc = np.hstack(fall['changeRewLoc'])
                trialnum=fall['trialnum'][0]
                rewards = fall['rewards'][0]
                lick=fall['licks'][0]
                time=fall['timedFF'][0]
                if animal=='e145':
                        ybinned=ybinned[:-1]
                        forwardvel=forwardvel[:-1]
                        changeRewLoc=changeRewLoc[:-1]
                        trialnum=trialnum[:-1]
                        rewards=rewards[:-1]
                        lick=lick[:-1]
                        time=time[:-1]
                # set vars
                eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
                rz = get_rewzones(rewlocs,1/scalingf)       
                # get average success rate
                rates = []
                for ep in range(len(eps)-1):
                        eprng = range(eps[ep],eps[ep+1])
                        success, fail, str_trials, ftr_trials, ttr, \
                        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                        rates.append(success/total_trials)
                rate=np.nanmean(np.array(rates))
                # dark time params
                track_length_dt = 550 # cm estimate based on 99.9% of ypos
                track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
                bins_dt=150 
                bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
                # added to get anatomical info
                # takes time
                fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
                Fc3 = fall_fc3['Fc3']
                dFF = fall_fc3['dFF']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
                dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
                skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)                
                if animal in lowcountan: skewthres=1.2
                else: skewthres=2
                Fc3 = Fc3[:, skew>skewthres] # only keep cells with skew greateer than 2
                tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
                        rewsize,ybinned,time,lick,
                        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
                        bins=bins_dt)  
                bin_size=3
                tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=16,velocity_filter=True)
                lick_correct_abs, _,_,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick,lick]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=16,velocity_filter=True)
                vel_correct_abs, _,_,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel,forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=16,velocity_filter=True)

                # trial by trial raster
                # abs distance
                # trialstates, licks_all, tcs_all, coms_all, ypos_max_all_ep, vels_all =make_tuning_curves_time_trial_by_trial_w_darktime(eps, rewlocs, rewsize, lick, ybinned, time, Fc3, trialnum, rewards, forwardvel, scalingf,bins=150)

                def moving_average(x, window_size=3):
                        return np.convolve(x, np.ones(window_size)/window_size, mode='same')

                # tcs_correct_abs=tcs_correct_abs[:3]
                # cells raster
                # panel for fig 4
                fig, axes = plt.subplots(ncols=len(tcs_correct_abs),nrows = 3,figsize=(9,6),height_ratios=[3,1,1])
                axes=axes.flatten()
                goal_cells=np.unique(np.concatenate(com_goal))
                for ep in range(len(tcs_correct_abs)):
                        ax=axes[ep]
                        m=np.array([moving_average(tcs_correct_abs[ep, gc].T, window_size=3) for gc in range(Fc3.shape[1])])
                        m=(m.T/np.nanmax(m,axis=1)).T
                        # sort by ctrl ep
                        m=m[np.argsort(coms_correct_abs[optoep-2,:])]
                        m = m[~np.isnan(m).any(axis=1)]
                        im=ax.imshow(m,aspect='auto')
                        ax.axvline(rewlocs[ep]/bin_size,color='w',linestyle='--')
                        ax.set_title(f'Epoch {ep+1}\n Reward @ {int(rewlocs[ep]-5)} cm',fontsize=14)
                        ax.set_xticks([0,90])
                        ax.set_xticklabels([])
                        if ep==0: ax.set_ylabel('All cells (sorted)')
                        if not ep==0: ax.set_yticklabels([])

                for ep in range(len(tcs_correct_abs)):
                        ax=axes[ep+len(tcs_correct_abs)]
                        m=moving_average(lick_correct_abs[ep][0], window_size=3)
                        m=m/np.nanmax(m)
                        ax.plot(m,color=colors[ep])
                        ax.axvline(rewlocs[ep]/bin_size,color=colors[ep],linestyle='--')
                        ax.spines[['top', 'right']].set_visible(False)
                        ax.set_xticks([0,90])
                        ax.set_xticklabels([])
                        if ep==0: ax.set_ylabel('Norm. licks', fontsize=14)
                for ep in range(len(tcs_correct_abs)):
                        ax=axes[ep+len(tcs_correct_abs)*2]
                        m=moving_average(vel_correct_abs[ep][0], window_size=3)
                        m=m/np.nanmax(m)
                        ax.plot(m,color=colors[ep])
                        ax.axvline(rewlocs[ep]/bin_size,color=colors[ep],linestyle='--')
                        ax.spines[['top', 'right']].set_visible(False)
                        ax.set_xticks([0,90])
                        ax.set_xticklabels([0,270])
                        if ep==0: ax.set_ylabel('Norm. Velocity', fontsize=14)

                        # tr_by_tr = vel_correct_abs[ep][0]
                        # sem = np.array([moving_average(s,window_size=5) for s in tr_by_tr]).T
                        # m = np.nanmean(sem)
                        # ax.fill_between(
                        #     range(0, bins_dt),
                        #     m - scipy.stats.sem(sem, axis=1, nan_policy='omit'),
                        #     m + scipy.stats.sem(sem, axis=1, nan_policy='omit'),
                        #     alpha=0.5
                        #     )  
                # Add colorbar to the right
                # Create a new axis for the colorbar without affecting subplot layout
                cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Norm. $\Delta F/F$', fontsize=12)
                ax.set_xlabel('Track position (cm)')
                fig.suptitle(f'{animal},{day}, {in_type}, opto ep {optoep}')
                plt.savefig(os.path.join(savedst, f'{animal}_{day}_allcell_raster.svg'))

#%%
# trial by trial raster
tcs_dist=tcs_all
licks_all_dist=licks_all
coms_dist=coms_all
ypos_max = ypos_max_all_ep
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']        
clls = [50]  # subset of goal_cells to plot
for cll in clls:
        fig, axes_all = plt.subplots(nrows=len(tcs_dist), ncols=4, figsize=(15, 10), sharex=True, width_ratios=[1.5, 1, 1, 1])
        vmin = 0
        vmax = np.nanmax(tcs_correct[:, cll])

        for ep in range(len(tcs_dist)):
                axes = axes_all[ep, :]
                # --- Load data ---
                raw_data = tcs_dist[ep][cll]
                lick_data = licks_all_dist[ep]
                vel_data = vels_all[ep]
                trial_mask = np.array(trialstates[ep])
                coms = np.array(coms_dist[ep][cll])

                # --- Filter valid trials ---
                valid_trials = [
                i for i in range(raw_data.shape[0])
                if trial_mask[i] != -1 and
                not np.all(np.isnan(raw_data[i])) and
                not np.all(np.isnan(lick_data[i])) and
                not np.all(np.isnan(vel_data[i]))
                ]
                data = raw_data[valid_trials]
                lick_data = lick_data[valid_trials]
                vel_data = vel_data[valid_trials]
                trial_mask_filtered = trial_mask[valid_trials]
                coms_filtered = coms[valid_trials]

                # --- Trial mask overlay ---
                trial_mask_2d = trial_mask_filtered[:, np.newaxis] * np.ones((1, data.shape[1]))

                # --- dF/F plot ---
                norm_data = (data - np.nanmin(data, axis=1, keepdims=True)) / \
                        (np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True) + 1e-10)
                divider = make_axes_locatable(axes[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = axes[0].imshow(norm_data, aspect='auto', cmap='Greys')
                fig.colorbar(im, cax=cax, orientation='vertical', label='Norm. $\Delta F/F$')

                a = 0.4
                axes[0].imshow(trial_mask_2d, cmap=cmap, norm=norm, aspect='auto', alpha=a)

                bin_size_dt = [ypos / bins_dt for ypos in ypos_max[ep]]
                axes[0].scatter(coms_filtered / np.nanmean(bin_size_dt), np.arange(len(coms_filtered)), color='w', marker='|')

                # --- Trial-average trace ---
                avg_trace = moving_average(tcs_correct[ep, goal_cells[cll]].T, window_size=5)
                axes[3].plot(avg_trace, color=colors[ep],linewidth=2)
                axes[3].legend()
                axes[3].set_title(f'Reward @ {int(rewlocs[ep])}')
                axes[3].set_ylabel('$\Delta F/F$')

                # --- Lick COM ---
                com_trial_lick = []
                for arr in lick_data:
                        arr_clean = np.nan_to_num(arr, nan=0.0)
                        bins = np.arange(len(arr_clean))
                        total = np.sum(arr_clean)
                        if total > 0:
                                com = np.sum(bins * arr_clean) / total
                        else:
                                com = np.nan
                        com_trial_lick.append(com)

                norm_licks = (lick_data - np.nanmin(lick_data, axis=1, keepdims=True)) / \
                        (np.nanmax(lick_data, axis=1, keepdims=True) - np.nanmin(lick_data, axis=1, keepdims=True) + 1e-10)
                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im=axes[1].imshow(norm_licks, aspect='auto', cmap='Blues')
                fig.colorbar(im, cax=cax, orientation='vertical', label='Norm. licks')
                axes[1].scatter(com_trial_lick, np.arange(len(com_trial_lick)), color='k', marker='|')
                axes[1].imshow(trial_mask_2d, cmap=cmap, norm=norm, aspect='auto', alpha=a)

                # --- Velocity COM ---
                com_trial_vel = []
                for arr in vel_data:
                        arr = np.nan_to_num(arr, nan=0.0)
                        bins = np.arange(len(arr))
                        total = np.sum(arr)
                        com = np.sum(bins * arr) / total if total > 0 else np.nan
                        com_trial_vel.append(com)

                norm_vels = (vel_data - np.nanmin(vel_data, axis=1, keepdims=True)) / \
                        (np.nanmax(vel_data, axis=1, keepdims=True) - np.nanmin(vel_data, axis=1, keepdims=True) + 1e-10)
                divider = make_axes_locatable(axes[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im=axes[2].imshow(norm_vels, aspect='auto', cmap='Purples')
                fig.colorbar(im, cax=cax, orientation='vertical', label='Norm. velocity')
                axes[2].scatter(com_trial_vel, np.arange(len(com_trial_vel)), color='k', marker='|')
                axes[2].imshow(trial_mask_2d, cmap=cmap, norm=norm, aspect='auto', alpha=a)

                # --- Formatting ---
                center_bin = bins_dt / 2
                for ax in axes:
                        ax.axvline(center_bin, color='k')
                axes[0].set_title(f'{animal}, {day}, cell: {goal_cells[cll]}')

                ticks = [0, bins_dt/2, bins_dt - 1]
                axes[3].set_xticks(ticks)
                axes[3].set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
                axes[3].set_xlabel('Reward-centric distance ($\Theta$)')
                axes[0].set_ylabel('Trial #')
                axes[3].spines[['top', 'right']].set_visible(False)

                plt.tight_layout()
                # plt.savefig(os.path.join(savedst, f'{animal}_{day}_rewcell{cll}_trialbytrial.svg'))

                #%% 
                plt.close(fig)