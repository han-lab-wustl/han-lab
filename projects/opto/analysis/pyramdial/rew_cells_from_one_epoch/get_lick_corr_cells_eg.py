#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
from projects.pyr_reward.placecell import make_tuning_curves_time_trial_by_trial
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones, cosine_sim_ignore_nan
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.stats import spearmanr

conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'lickcorr.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
def normalize_rows(mat):
    mat = np.array(mat)
    row_min = np.nanmin(mat, axis=1, keepdims=True)
    row_max = np.nanmax(mat, axis=1, keepdims=True)
    return (mat - row_min) / (row_max - row_min + 1e-9)

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
cm_window=20
datadct={}
plot=False
iis = [49]
plot=True
plot2=False
plt.rc('font', size=20)
for ii in iis:
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   optoep=conddf.optoep.values[ii]
   in_type=conddf.in_type.values[ii]
   if ii!=202 and optoep>1:
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
      if in_type=='vip' or animal=='z17':
         Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
         dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      else:
         Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
         dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
      dt = np.nanmedian(np.diff(time))
      lick_rate=smooth_lick_rate(lick,dt)
      
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
         rewsize,ybinned,time,lick,
         Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8)  
      bin_size=3
      # abs position
      # all trials
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,velocity_filter=True)
      # get cells that maintain their coms across at least 2 epochs
      # only get opto ad prev epoch
      coms_correct_abs = coms_correct_abs[[optoep-2,optoep-1]]
      tcs_correct_abs = tcs_correct_abs[[optoep-2,optoep-1]]
      place_window = 20 # cm converted to rad                
      perm = list(combinations(range(len(coms_correct_abs)), 2))     
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      pcs_all = intersect_arrays(*compc)
      lick_correct_abs, _,lick_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,velocity_filter=True)
      lick_correct_abs = lick_correct_abs[[optoep-2,optoep-1]]
      vel_correct_abs, _,vel_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,velocity_filter=True)
      # beh rew aligned
      lick_correct, lick_coms_correct, lick_fail, lick_coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
      rewsize,ybinned,time,lick,
      np.array([lick_rate]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt)  
      vel_correct, vel_coms_correct, vel_fail, vel_coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
         rewsize,ybinned,time,lick,
         np.array([forwardvel]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt)  
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      # get cells in epoch 1
      coms_ep1 = coms_rewrel[0]
      bound=np.pi/2
      ep1_rew_cells = np.where(((coms_ep1>-bound) & (coms_ep1<bound)))[0]
      ########## correct trials      
      lick_tc_cs = np.array([spearmanr(tcs_correct_abs[0,cll,:], lick_correct_abs[0][0])[0] for cll in ep1_rew_cells])
      # get high correlated cells in ep1, plot frames in ep 2
      lick_corr_cells = ep1_rew_cells[lick_tc_cs>np.nanmean(lick_tc_cs)]
      # plot
      from mpl_toolkits.axes_grid1.inset_locator import inset_axes
      fig,axes=plt.subplots(nrows=3,ncols=2,sharey='row',sharex=True, height_ratios=[3,1,1],figsize=(6,7))
      lbls=['LED off', 'LED on']
      for ep in range(2):
         tc_abs_norm = normalize_rows(tcs_correct_abs[ep, lick_corr_cells][np.argsort(coms_correct_abs[0,lick_corr_cells])])
         tc_th_norm  = normalize_rows(tcs_correct[ep, lick_corr_cells][np.argsort(coms_correct[0,lick_corr_cells])])
         # Plot normalized heatmaps
         im0 = axes[0,ep].imshow(tc_abs_norm, aspect='auto', vmin=0, vmax=1)
         axes[0,ep].axvline(rewlocs[ep]/3, color='w', linestyle='--', linewidth=3)
         axes[0,ep].set_yticks([0,tc_abs_norm.shape[0]-1])
         axes[1,ep].plot(lick_correct_abs[ep][0],color='k')
         axes[1,ep].axvline(rewlocs[ep]/3,color='k',linestyle='--',linewidth=3)
         axes[1,0].set_ylabel('Lick rate (licks/s)')
         axes[2,ep].plot(vel_correct_abs[ep][0],color='grey')
         axes[2,ep].axvline(rewlocs[ep]/3,color='k',linestyle='--',linewidth=3)
         axes[2,0].set_ylabel('Velocity (cm/s)')
         axes[2,ep].spines[['top','right']].set_visible(False)
         axes[1,ep].spines[['top','right']].set_visible(False)
         if ep == 1:
            # Inset colorbar for im0 (track aligned)
            cax0 = inset_axes(axes[0, ep],width="5%", height="50%",loc='right',borderpad=-1)
            fig.colorbar(im0, cax=cax0,label=f'Norm. $\Delta F/F$')
         from matplotlib.patches import Rectangle
         if ep == 1:
            rew_x = rewlocs[ep] / 3  # convert reward location to bin units (same as axvline)
            patch = Rectangle(
               (0, -5),             # x=0 (start), y=0 (bottom of heatmap)
               width=rew_x,        # up to reward
               height=30,  # full height of heatmap
               color='red',
               alpha=0.2
            )
            axes[0, ep].add_patch(patch)
         axes[2,ep].set_xticks([0,90])
         axes[2,ep].set_xticklabels([0,270])
         axes[2,0].set_xlabel('Track position (cm)')
      axes[0,0].set_ylabel('Lick correlated cell # (sorted)')
      fig.suptitle(f'{animal}, {day}, {in_type}, optoep {optoep-1}')
      plt.savefig(os.path.join(savedst, f'fig5_{animal}_{day}_lickcorrcells.svg'),bbox_inches='tight')
      
