"""
zahra
circularly shuffle dff
95% spatial info cells
split into place and rew
june 25
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves_by_trialtype_w_darktime_early, make_tuning_curves, make_tuning_curves_early
from projects.opto.analysis.pyramdial.placecell import process_goal_cell_proportions
import numpy as np
from scipy.ndimage import gaussian_filter1d
from projects.pyr_reward.rewardcell import get_rewzones, intersect_arrays
def compute_spatial_info(p_i, f_i):
    f = np.sum(f_i * p_i)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = f_i / f
        log_term = np.where(ratio > 0, np.log2(ratio), 0)
        si = np.nansum(p_i * f_i * log_term)
    return si

def bin_activity_per_trial(position, activity, n_bins=3, track_length=270):
    trial_activity = np.zeros((len(position), n_bins))
    occupancy = np.zeros((len(position), n_bins))
    bin_edges = np.linspace(0, track_length, n_bins + 1)
    for i, (pos, act) in enumerate(zip(position, activity)):
        for b in range(n_bins):
            in_bin = (pos >= bin_edges[b]) & (pos < bin_edges[b+1])
            occupancy[i, b] = np.sum(in_bin)
            if np.any(in_bin):
                trial_activity[i, b] = np.mean(act[in_bin])
    return trial_activity, occupancy

def compute_trial_avg_si(activity_matrix, occupancy_matrix):
    mean_activity = np.nanmean(activity_matrix, axis=0)
    mean_occupancy = np.nansum(occupancy_matrix, axis=0)
    p_i = mean_occupancy / np.nansum(mean_occupancy)
    return compute_spatial_info(p_i, mean_activity)

def shuffle_positions(position_trials, frame_rate):
    """
    Circularly permute position data by a random shift of at least ~1s.
    Skips trials shorter than 1s.
    """
    shuffled = []
    for pos in position_trials:
        n = len(pos)
        if n <= int(frame_rate):
            shuffled.append(pos.copy())  # skip shuffle if trial too short
            continue
        shift = np.random.randint(int(frame_rate), n)
        shuffled_pos = np.roll(pos, shift)
        shuffled.append(shuffled_pos)
    return shuffled

def compute_shuffle_distribution(position_trials, activity_trials,  frame_rate, n_shuffles=100):
    shuffle_SIs = []
    for _ in range(n_shuffles):
        permuted_pos = shuffle_positions(position_trials,frame_rate)
        binned, occ = bin_activity_per_trial(permuted_pos, activity_trials)
        si = compute_trial_avg_si(binned, occ)
        shuffle_SIs.append(si)
    return np.array(shuffle_SIs)

def is_place_cell(position_trials, activity_trials, frame_rate,n_shuffles=100, alpha=0.05):
    binned, occ = bin_activity_per_trial(position_trials, activity_trials)
    real_si = compute_trial_avg_si(binned, occ)
    shuffled_sis = compute_shuffle_distribution(position_trials, activity_trials,frame_rate, n_shuffles=n_shuffles)
    p_val = np.mean(real_si <= shuffled_sis)
    return real_si > np.percentile(shuffled_sis, 95), real_si, shuffled_sis, p_val

#%%
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_rew.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# initialize var
datadct = {} # overwrite
place_window = 20
num_iterations=1000
bin_size=3 # cm
lasttr=8 # last trials
bins=90
#%%
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   # check if its the last 3 days of animal behavior
   andf = conddf[(conddf.animals==animal) &( conddf.optoep<2)]
   lastdays = andf.days.values[-3:]
   if (day in lastdays):
      if animal=='e145': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF','licks',
         'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
         'stat'])
      pcs = np.vstack(np.array(fall['putative_pcs'][0]))
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
      time = fall['timedFF'][0]
      lick = fall['licks'][0]
      if animal=='e145':
         ybinned=ybinned[:-1]
         forwardvel=forwardvel[:-1]
         changeRewLoc=changeRewLoc[:-1]
         trialnum=trialnum[:-1]
         rewards=rewards[:-1]
         time=time[:-1]
         lick=lick[:-1]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      # only test opto vs. ctrl
      eptest = conddf.optoep.values[ii]
      if conddf.optoep.values[ii]<2: 
         eptest = random.randint(2,3)   
         if len(eps)<4: eptest = 2 # if no 3 epochs    
      fr=31.25
      if animal=='z9' or animal=='e190':
         fr=fr/2
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3_org = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      dFF_org = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      skew = scipy.stats.skew(dFF_org, nan_policy='omit', axis=0)
      dFF=dFF_org[:, skew>2]
      Fc3=Fc3_org[:, skew>2]
      # low cells
      if animal=='e217' or animal=='z17' or animal=='z14' or animal=='e200':
         dFF=dFF_org[:, skew>1]
         Fc3=Fc3_org[:, skew>1]
      # per epoch si
      # nshuffles=100   
      rz = get_rewzones(rewlocs,1/scalingf)
      pcs_ep=[]; si_ep=[]
      for ep in range(len(eps)-1):
         eprng = np.arange(eps[ep], eps[ep+1])
         trials=trialnum[eprng]
         activity_trials=dFF[eprng,:]
         pcs = []; si=[]
         for cll in range(activity_trials.shape[1]):
            is_sig, real_si, shuffled_sis, p_val = is_place_cell([ybinned[eprng][trials==tr] for tr in np.unique(trials)], [activity_trials[trials==tr][:,cll] for tr in np.unique(trials)], fr)
            pcs.append(is_sig); si.append(real_si)
         pcs_ep.append(pcs); si_ep.append(si)
      spatially_tuned = np.sum(pcs_ep,axis=0)>0 # if tuned in any epoch
      dFF=dFF[:,spatially_tuned]
      Fc3=Fc3[:,spatially_tuned] # replace to make easier
      if conddf.optoep.values[ii]<2: 
         eptest = random.randint(2,3)      
      if len(eps)<4: eptest = 2 # if no 3 epochs
      comp = [eptest-2,eptest-1] # eps to compare, python indexing   
      cm_window=20
      # TODO:
      # get rew cells activity and %
      # get place dff and %
      # get other spatial tuned cells activity and %
      # tc w/ dark time
      print('making tuning curves...\n')
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,relpos = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 
      # early tc
      tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)        
      goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad

      results_pre_early = process_goal_cell_proportions(eptest, 
      cell_type='pre',
      coms_correct=coms_correct_early,
      tcs_correct=tcs_correct_early,
      rewlocs=rewlocs,
      animal=animal,
      day=day,
      pdf=pdf,
      rz=rz,
      scalingf=scalingf,
      bins=bins,
      goal_window=goal_window
      )

      results_post_early = process_goal_cell_proportions(eptest, 
         cell_type='post',
         coms_correct=coms_correct_early,
         tcs_correct=tcs_correct_early,
         rewlocs=rewlocs,
         animal=animal,
         day=day,
         pdf=pdf,
         rz=rz,
         scalingf=scalingf,
         bins=bins,
         goal_window=goal_window
      )
      results_pre = process_goal_cell_proportions(eptest, 
         cell_type='pre',
         coms_correct=coms_correct,
         tcs_correct=tcs_correct,
         rewlocs=rewlocs,
         animal=animal,
         day=day,
         pdf=pdf,
         rz=rz,
         scalingf=scalingf,
         bins=bins,
         goal_window=goal_window
      )
      results_post = process_goal_cell_proportions(eptest, 
         cell_type='post',
         coms_correct=coms_correct,
         tcs_correct=tcs_correct,
         rewlocs=rewlocs,
         animal=animal,
         day=day,
         pdf=pdf,
         rz=rz,
         scalingf=scalingf,
         bins=bins,
         goal_window=goal_window
      )
      print('#############making place tcs#############\n')
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
      Fc3,trialnum,rewards,forwardvel,
      rewsize,bin_size) # last 5 trials
      tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early = make_tuning_curves_early(eps,rewlocs,ybinned, Fc3,trialnum,rewards,forwardvel,
      rewsize,bin_size) # last 5 trials
      # all goal
      goal_cells = np.unique(np.concatenate([xx['goal_id'] for xx in [results_pre, results_post, results_pre_early, results_post_early]]))
      perm = [(eptest-2, eptest-1)]   
      print(eptest, perm)            
      # get cells that maintain their coms across at least 2 epochs
      place_window = 20 # cm converted to rad                
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      compc=[xx for xx in compc if len(xx)>0]
      if len(compc)>0:
         pcs_all = intersect_arrays(*compc)
         # exclude goal cells
         pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
      else:
         pcs_all=[]      
      pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
      pc_p=len(pcs_all)/len(coms_correct_abs[0])
      #early
      com_per_ep = np.array([(coms_correct_abs_early[perm[jj][0]]-coms_correct_abs_early[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      compc=[xx for xx in compc if len(xx)>0]
      if len(compc)>0:
         pcs_all_early = intersect_arrays(*compc)
         # exclude goal cells
         pcs_all_early=[xx for xx in pcs_all_early if xx not in goal_cells]
      else:
         pcs_all_early=[]      
      pc_p_early=len(pcs_all_early)/len(coms_correct_abs[0])

      # get % of other spatially tuned cells
      spatially_tuned_not_rew_place = [xx for xx in range(Fc3.shape[1]) if xx not in pcs_all and xx not in pcs_all_early and xx not in goal_cells]
      spatially_tuned_not_rew_place_p=len(spatially_tuned_not_rew_place)/len(coms_correct[0])
      print(spatially_tuned_not_rew_place_p,pc_p_early,pc_p)
      datadct[f'{animal}_{day:03d}'] = [spatially_tuned_not_rew_place_p,pc_p_early,pc_p,results_pre, results_post, results_pre_early, results_post_early]
#%%
spatially_tuned_not_rew_place=[v[0] for k,v in datadct.items()]
placecell_p=[v[2] for k,v in datadct.items()]
pre_p=[v[3]['goal_cell_prop'] for k,v in datadct.items()]
post_p=[v[4]['goal_cell_prop'] for k,v in datadct.items()]

df=pd.DataFrame()
df['proportions']=np.concatenate([spatially_tuned_not_rew_place,placecell_p,pre_p,post_p])
allty=[spatially_tuned_not_rew_place,placecell_p,pre_p,post_p]
lbl=['other_spatially_tuned','place','pre','post']
df['type']=np.concatenate([[lbl[i]]*len(allty[i]) for i in range(len(lbl))])

#%%
df=df[(df.proportions<1) & (df.proportions>0)]
sns.barplot(x='proportions',y='type',data=df)
sns.stripplot(x='proportions',y='type',data=df)