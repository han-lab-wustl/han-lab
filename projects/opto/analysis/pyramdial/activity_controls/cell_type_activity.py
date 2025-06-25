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
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

#%%
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_rew.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# initialize var
datadct = {} # overwrite
save_shuf_info=[]
place_window = 20
cm_window = 20
num_iterations=100
bin_size=3 # cm
bins=90

# iterate through all animals
for ii in range(len(conddf)):
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
      dFF=dFF_org[:, skew>2]
      Fc3=Fc3_org[:, skew>2]
      # low cells
      if animal=='e217' or animal=='z17' or animal=='z14' or animal=='e200':
         dFF=dFF_org[:, skew>1]
         Fc3=Fc3_org[:, skew>1]
      # per epoch si
      # nshuffles=100   
      rz = get_rewzones(rewlocs,1/scalingf)
      comp = [eptest-2,eptest-1] # eps to compare, python indexing   
      # tc w/ dark time
      print('making tuning curves...\n')
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,relpos = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt) 
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
      goal_cells = np.unique(np.concatenate([xx['goal_id'] for xx in [results_pre, results_post]])).astype(int)
      print(f'\n pre si restriction: {len(goal_cells)} rew cells')
      perm = [(eptest-2, eptest-1)]   
      print(eptest, perm)            
      # get cells that maintain their coms across at least 2 epochs
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      compc=[xx for xx in compc if len(xx)>0]
      if len(compc)>0:
         pcs_all = intersect_arrays(*compc)
         # exclude no sp cells
         # pcs_all=[xx for xx in pcs_all if xx not in not_sp_tuned]
         # exclude goal cells
         pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
      else:
         pcs_all=[]      
      pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
      pc_p=len(pcs_all)/len(coms_correct_abs[0])
      # get % of other spatially tuned cells
      spatially_tuned_not_rew_place = [xx for xx in range(Fc3.shape[1]) if xx not in pcs_all and xx not in goal_cells]
      spatially_tuned_not_rew_place_p=len(spatially_tuned_not_rew_place)/len(coms_correct_abs[0])
      print(spatially_tuned_not_rew_place_p,pc_p,results_pre['goal_cell_prop'],results_post['goal_cell_prop'])
      # get activity, pre vs. post
      for ep in range(len(eps)-1):
         eprng = np.arange(eps[ep], eps[ep+1])
         ypos = ybinned[eprng]
         # leading up to and in rew zone
         spatially_tuned_not_rew_place_act_pre = np.nanmean(dFF[eprng,:][:,spatially_tuned_not_rew_place][(ypos<rewlocs[ep]+rewsize/2),:],axis=0)
         # post
         spatially_tuned_not_rew_place_act_post = np.nanmean(dFF[eprng,:][:,spatially_tuned_not_rew_place][(ypos>rewlocs[ep]+rewsize/2),:],axis=0)
         place_pre = np.nanmean(dFF[eprng,:][:,pcs_all][(ypos<rewlocs[ep]+rewsize/2),:],axis=0)
         place_post = np.nanmean(dFF[eprng,:][:,pcs_all][(ypos>rewlocs[ep]+rewsize/2),:],axis=0)
         rew_pre = np.nanmean(dFF[eprng,:][:,np.array(results_pre['goal_id']).astype(int)][(ypos<rewlocs[ep]+rewsize/2),:],axis=0)
         rew_post = np.nanmean(dFF[eprng,:][:,np.array(results_post['goal_id']).astype(int)][(ypos>rewlocs[ep]+rewsize/2),:],axis=0)
         prerew_post = np.nanmean(dFF[eprng,:][:,np.array(results_pre['goal_id']).astype(int)][(ypos>rewlocs[ep]+rewsize/2),:],axis=0)

      datadct[f'{animal}_{day:03d}'] = [spatially_tuned_not_rew_place_p,pc_p,results_pre, results_post, comp,spatially_tuned_not_rew_place_act_pre, spatially_tuned_not_rew_place_act_post,place_pre,place_post,rew_pre,rew_post,prerew_post]
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
df = df.merge(conddf[['animals', 'days', 'optoep', 'in_type']], on=['animals', 'days'], how='left')
df['opto']=df['optoep']>-1
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type]
keep = ~((df.animals == 'z14') & (df.days < 15))
keep &= ~((df.animals == 'z15') & (df.days < 8))
keep &= ~((df.animals == 'e217') &((df.days < 9) | (df.days == 26)))
keep &= ~((df.animals == 'e216') & (df.days < 32))
keep &= ~((df.animals=='e200')&((df.days.isin([67]))))
# keep &= ~((df.animals=='e218')&(df.days>44))
df = df[keep].reset_index(drop=True)

# Get non-opto averages to subtract
non_opto_means = (
    df[df.opto == False]
    .groupby(['animals', 'type', 'condition'])['proportions']
    .mean()
    .reset_index()
    .rename(columns={'proportions': 'baseline'})
)
# Merge with opto trials
df_opto = df[df.opto == True].copy()
df_opto = df_opto.merge(non_opto_means, on=['animals', 'type', 'condition'], how='left')
# Compute normalized proportions
df_opto['norm_proportions'] = df_opto['proportions']-df_opto['baseline']
df_opto = df_opto.groupby(['animals', 'type', 'condition']).mean(numeric_only=True).reset_index()
# Define comparison groups and types
conditions_to_compare = df_opto['condition'].unique()
cell_types = df_opto['type'].unique()

# Set up plot
plt.figure(figsize=(5, 6))
ax = sns.barplot(y='norm_proportions', x='type', hue='condition', data=df_opto, errorbar='se')
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