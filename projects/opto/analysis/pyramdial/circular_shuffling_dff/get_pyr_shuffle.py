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
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

savepickle=r'Z:\condition_df\circ_shuffle_vip_opto.p'
with open(savepickle, "rb") as fp: #unpickle
   datadct = pickle.load(fp)
# initialize var

def compute_spatial_info(p_i, f_i):
    f = np.sum(f_i * p_i)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = f_i / f
        log_term = np.where(ratio > 0, np.log2(ratio), 0)
        si = np.nansum(p_i * f_i * log_term)
    return si

def bin_activity_per_trial(position_trials, activity_trials, n_bins=90, track_length=270):
    n_trials = len(position_trials)
    trial_activity = np.zeros((n_trials, n_bins), dtype=np.float32)
    occupancy = np.zeros((n_trials, n_bins), dtype=np.float32)
    bin_edges = np.linspace(0, track_length, n_bins + 1, dtype=np.float32)

    for i in range(n_trials):
        pos = np.asarray(position_trials[i], dtype=np.float32)
        act = np.asarray(activity_trials[i], dtype=np.float32)
        bin_ids = np.digitize(pos, bin_edges, right=False) - 1

        for b in range(n_bins):
            in_bin = (bin_ids == b)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                occupancy[i, b] = n_in_bin
                trial_activity[i, b] = np.mean(act[in_bin], dtype=np.float32)

    return trial_activity, occupancy


def compute_trial_avg_si(activity_matrix, occupancy_matrix):
    mean_activity = np.nanmean(activity_matrix, axis=0)
    mean_occupancy = np.nansum(occupancy_matrix, axis=0)
    p_i = mean_occupancy / np.nansum(mean_occupancy)
    return compute_spatial_info(p_i, mean_activity)

def shuffle_positions(position_trials, frame_rate):
    for pos in position_trials:
        n = len(pos)
        if n <= int(frame_rate):
            yield np.asarray(pos, dtype=np.float32)
        else:
            shift = np.random.randint(int(frame_rate), n)
            yield np.roll(np.asarray(pos, dtype=np.float32), shift)

def compute_shuffle_si_once(position_trials, activity_trials, frame_rate):
    permuted_pos = list(shuffle_positions(position_trials, frame_rate))
    binned, occ = bin_activity_per_trial(permuted_pos, activity_trials)
    return compute_trial_avg_si(binned, occ)

def compute_shuffle_distribution(position_trials, activity_trials, frame_rate, n_shuffles=100, n_jobs=1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(compute_shuffle_si_once)(position_trials, activity_trials, frame_rate)
            for _ in range(n_shuffles)
        ),
        dtype=np.float32
    )

def is_place_cell(position_trials, activity_trials, frame_rate, n_shuffles=100, p=95, n_jobs=-1):
    # print(activity_trials)
    binned, occ = bin_activity_per_trial(list(position_trials), list(activity_trials))
    real_si = compute_trial_avg_si(binned, occ)
    shuffled_sis = compute_shuffle_distribution(position_trials, activity_trials, frame_rate, n_shuffles, n_jobs)
    p_val = np.mean(real_si <= shuffled_sis)
    return real_si > np.percentile(shuffled_sis, p), real_si, shuffled_sis, p_val
from joblib import Parallel, delayed

def run_place_cell_batch(position_trials, activity_trials_list, frame_rate, n_shuffles=100, n_jobs=-1):
    """
    Run place cell analysis on a batch of cells in parallel.

    Parameters
    ----------
    position_trials : list of np.ndarray
        Position arrays for each trial (same for all cells)
    activity_trials_list : list of list of np.ndarray
        One list per cell, containing trial-wise activity arrays
    frame_rate : float
        Sampling rate (e.g. frames/sec)
    n_shuffles : int
        Number of shuffles per cell
    n_jobs : int
        Number of parallel jobs (-1 = use all cores)

    Returns
    -------
    results : list of tuples
        Each tuple is (is_place_cell, real_si, shuffled_sis, p_val)
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(is_place_cell)(
            position_trials,
            activity_trials,
            frame_rate,
            n_shuffles=n_shuffles,
            n_jobs=1  # Important! Don’t nest parallel jobs
        )
        for activity_trials in activity_trials_list
    )
    return results

#%%
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_rew.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# initialize var
# datadct = {} # overwrite
save_shuf_info=[]
place_window = 20
num_iterations=50
bin_size=3 # cm
bins=90
a=0.05 # threshold for si detection
#%%
# iterate through all animals
for ii in range(159,len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   # check if its the last 3 days of animal behavior
   andf = conddf[(conddf.animals==animal)]
   lastdays = andf.days.values#[-3:]
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
      if animal=='z9' or animal=='e190' or animal=='z14':
         fr=fr/2
      if animal=='z17':
         fr=fr/3
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3_org = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF_org = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF_org, nan_policy='omit', axis=0)
      dFF=dFF_org[:, skew>1.2]
      Fc3=Fc3_org[:, skew>1.2]
      # per epoch si
      # nshuffles=100   
      rz = get_rewzones(rewlocs,1/scalingf)
      pcs_ep=[]; si_ep=[]
      for ep in range(len(eps)-1):
         eprng = np.arange(eps[ep], eps[ep+1])
         trials=trialnum[eprng]
         activity_trials=dFF[eprng,:]
         position_trials = ybinned[eprng]
         pcs = []; si=[]
         # position_trials = [np.array(...), ...]      # shared position per trial
         # activity_trials_list = [[trial1_cell1, ...], [trial1_cell2, ...], ...]  # each sublist = one cell
         # for cells > 700 (e216)
         # only do on 700 cells
         activity_trials_list = [[activity_trials[trials==tr][:,cll] for tr in np.unique(trials)] for cll in np.arange(activity_trials.shape[1])]
         pos = [position_trials[trials==tr] for tr in np.unique(trials)]
         # make first pos 1.5 again
         pos_corr=[]
         for xx in pos:
            xx[0]=1.5
            pos_corr.append(xx)
         # fast shuf
         results = run_place_cell_batch(
            position_trials=pos_corr,
            activity_trials_list=activity_trials_list,
            frame_rate=fr,            # or whatever your frame rate is
            n_shuffles=num_iterations,
            n_jobs=4
         )
         place_cell_flags = [r[0] for r in results]
         real_sis = [r[1] for r in results]
         shuffle_sis = [r[2] for r in results]
         p_values = [r[3] for r in results]
         pcs_ep.append(np.array(p_values)<a); si_ep.append(real_sis)
      spatially_tuned = np.sum(np.array(pcs_ep),axis=0)>0 # if tuned in any epoch
      save_shuf_info.append([pcs_ep,si_ep,spatially_tuned])
      dFF_si=dFF[:,spatially_tuned]
      Fc3_si=Fc3[:,spatially_tuned] # replace to make easier
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
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,relpos = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt) 
      # early tc
      goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
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
      # tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early = make_tuning_curves_early(eps,rewlocs,ybinned, Fc3,trialnum,rewards,forwardvel,
      # rewsize,bin_size) # last 5 trials
      # # all goal
      goal_cells = np.unique(np.concatenate([xx['goal_id'] for xx in [results_pre, results_post]]))
      print(f'\n pre si restriction: {len(goal_cells)} rew cells')
      # remove goal cells that do not pass spatial info metric!!!
      not_sp_tuned = np.arange(Fc3.shape[1])[~spatially_tuned]
      goal_cells = [xx for xx in goal_cells if xx not in not_sp_tuned]
      print(f'\n post si restriction: {len(goal_cells)} rew cells')
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
         # exclude no sp cells
         pcs_all=[xx for xx in pcs_all if xx not in not_sp_tuned]
         # exclude goal cells
         pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
      else:
         pcs_all=[]      
      pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
      pc_p=len(pcs_all)/len(coms_correct_abs[0])
      # get % of other spatially tuned cells
      spatially_tuned_not_rew_place = [xx for xx in range(Fc3.shape[1]) if xx not in not_sp_tuned and xx not in pcs_all and xx not in goal_cells]
      spatially_tuned_not_rew_place_p=len(spatially_tuned_not_rew_place)/len(coms_correct_abs[0])
      print(spatially_tuned_not_rew_place_p,pc_p,results_pre['goal_cell_prop'],results_post['goal_cell_prop'])
      datadct[f'{animal}_{day:03d}'] = [spatially_tuned_not_rew_place_p,pc_p,results_pre, results_post,spatially_tuned]
#%%
with open(savepickle, "wb") as fp: 
   pickle.dump(datadct, fp) 
#%%
# per cell prop comparison
spatially_tuned_not_rew_place=[v[0] for k,v in datadct.items()]
placecell_p=[v[1] for k,v in datadct.items()]
pre_p=[v[2]['goal_cell_prop'] for k,v in datadct.items()]
post_p=[v[3]['goal_cell_prop'] for k,v in datadct.items()]

df=pd.DataFrame()
df['proportions']=np.concatenate([spatially_tuned_not_rew_place,placecell_p,pre_p,post_p])
allty=[spatially_tuned_not_rew_place,placecell_p,pre_p,post_p]
lbl=['other_spatially_tuned','place','pre','post']
df['type']=np.concatenate([[lbl[i]]*len(allty[i]) for i in range(len(lbl))])
df['animals']=[k.split('_')[0] for k,v in datadct.items()]*len(allty)
df['days']=[int(k.split('_')[1]) for k,v in datadct.items()]*len(allty)
df = df.merge(conddf[['animals','days', 'optoep', 'in_type']], on=['animals', 'days'], how='left')
df['opto']=df['optoep']>1
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type]
keep = ~((df.animals == 'z14') & (df.days < 15))
keep &= ~((df.animals == 'z15') & (df.days < 8))
keep &= ~((df.animals == 'e217') &((df.days < 9) | (df.days == 26)))
keep &= ~((df.animals == 'e216') & (df.days < 32))
keep &= ~((df.animals=='e200')&((df.days.isin([67]))))
keep &= ~((df.animals=='z16') | (df.animals=='e200'))
# keep &= ~((df.animals=='e218')&(df.days>44))
df = df[keep].reset_index(drop=True)

# Get non-opto averages to subtract
non_opto_means = (
    df[df.opto == False]
    .groupby(['animals','type', 'condition'])['proportions']
    .mean()
    .reset_index()
    .rename(columns={'proportions': 'baseline'})
)
# Merge with opto trials
df_opto = df[df.opto == True].copy()
df_opto = df_opto.merge(non_opto_means, on=['animals','type', 'condition'], how='left')
# Compute normalized proportions
df_opto['norm_proportions'] = df_opto['proportions']-df_opto['baseline']
df_opto = df_opto.groupby(['animals', 'type', 'condition']).mean(numeric_only=True).reset_index()
# Define comparison groups and types
conditions_to_compare = df_opto['condition'].unique()
cell_types = df_opto['type'].unique()

# Set up plot
plt.figure(figsize=(5, 6))
ax = sns.barplot(y='norm_proportions', x='type', hue='condition', data=df_opto, errorbar='se',fill=False)
sns.stripplot(y='norm_proportions', x='type', hue='condition', data=df_opto,dodge=True)

ax.set_xlabel('Opto - No-Opto Δ (Proportion)')
ax.set_ylabel('Cell Type')
ax.set_title('Optogenetic Modulation of Goal/Place Cell Proportions')
ax.spines[['top', 'right']].set_visible(False)

# Perform tests and annotate
y_max = df_opto['norm_proportions'].max()
y_offset = 0 * y_max if y_max > 0 else 0

for t in cell_types:
    data = df_opto[df_opto['type'] == t]
    ctrl_vals = data[data['condition'] == 'ctrl']['norm_proportions']

    xpos = {}
    for i, cond in enumerate(conditions_to_compare):
        if cond == 'ctrl':
            continue
        opto_vals = data[data['condition'] == cond]['norm_proportions']
        if len(ctrl_vals) > 0 and len(opto_vals) > 0:
            stat, pval = scipy.stats.ranksums(opto_vals, ctrl_vals, alternative='two-sided')
            print(f'{t}, {cond} vs ctrl: p = {pval:.3g}')

            # Set y-position for annotation
            bar_heights = data.groupby('condition')['norm_proportions'].mean()
            max_height = max(bar_heights.get(cond, 0), bar_heights.get('ctrl', 0))
            y = max_height + y_offset

            # Significance annotation
            if pval < 0.001:
                star = '***'
            elif pval < 0.01:
                star = '**'
            elif pval < 0.05:
                star = '*'
            else:
                star = 'n.s.'

            # Get bar position
            group_order = df_opto['condition'].unique()
            xpos = np.where((df_opto['type'].unique() == t))[0][0]
            xctrl = xpos
            xopto = xpos

            ax.plot([xctrl, xopto], [y, y], lw=1.2, c='k')
            ax.text((xctrl + xopto)/2, y + y_offset/2, star, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()