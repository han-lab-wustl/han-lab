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
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
cm_window=20
datadct={}
plot=False
# cm_window = [10,20,30,40,50,
# 60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
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
      Fc3 = Fc3[:, skew>1.2] # only keep cells with skew greateer than 2
      dt = np.nanmedian(np.diff(time))
      lick_rate=smooth_lick_rate(lick,dt)
      
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
         rewsize,ybinned,time,lick,
         Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8)  
      bin_size=3
      # abs position
      # all trials
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=150)
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
      lick_correct_abs, _,lick_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=150)
      lick_correct_abs = lick_correct_abs[[optoep-2,optoep-1]]
      vel_correct_abs, _,vel_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=150)
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
      bound=np.pi/4
      ep1_rew_cells = np.where(((coms_ep1>-bound) & (coms_ep1<bound)))[0]
      # test
      # fig,axes=plt.subplots(nrows=2,ncols=len(tcs_correct)-1,sharey=True,sharex='row')
      # for ep in range(tcs_correct.shape[0]-1):
      #    axes[0,ep].imshow(tcs_correct_abs[ep, ep1_rew_cells][np.argsort(coms_correct_abs[0,ep1_rew_cells])],aspect='auto')
      #    axes[0,ep].axvline(rewlocs[ep]/3,color='w',linestyle='--',linewidth=2)
      #    axes[1,ep].imshow(tcs_correct[ep, ep1_rew_cells][np.argsort(coms_correct[0,ep1_rew_cells])],aspect='auto')
      #    axes[1,ep].axvline(75,color='w',linestyle='--',linewidth=2)
      ########## correct trials      
      lick_tc_cs = np.array([spearmanr(tcs_correct_abs[0,cll,:], lick_correct_abs[0][0])[0] for cll in ep1_rew_cells])
      # get high correlated cells
      lick_corr_cells_ep1 = ep1_rew_cells[lick_tc_cs>np.nanmean(lick_tc_cs)]
      # compare to rew cells
      goal_window = 20*(2*np.pi/track_length) # cm converted to rad
      # change to relative value 
      coms_correct = coms_correct[[optoep-2,optoep-1]]
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      perm = list(combinations(range(len(coms_correct)), 2)) 
      rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
      # account for cells that move to the end/front
      # Define a small window around pi (e.g., epsilon)
      epsilon = .7 # 20 cm
      # Find COMs near pi and shift to -pi
      com_loop_w_in_window = []
      for pi,p in enumerate(perm):
         for cll in range(coms_rewrel.shape[1]):
               com1_rel = coms_rewrel[p[0],cll]
               com2_rel = coms_rewrel[p[1],cll]
               # print(com1_rel,com2_rel,com_diff)
               if ((abs(com1_rel - np.pi) < epsilon) and 
               (abs(com2_rel + np.pi) < epsilon)):
                     com_loop_w_in_window.append(cll)
      # get abs value instead
      coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
      com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
      com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
      lowerbound = -np.pi/4 # updated 4/21/25
      # only pre cells
      # com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
      #    xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
      #    xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
      # com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
      # if len(com_goal_farrew)>0:
      #       goal_cells = np.unique(np.concatenate(com_goal_farrew))
      #       goal_cells = intersect_arrays(*com_goal_farrew)
      # else:
      #       goal_cells=[]    

      # get all goal
      com_goal_farrew = com_goal
      com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
      if len(com_goal_farrew)>0:
            goal_cells = np.unique(np.concatenate(com_goal_farrew))
      else:
            goal_cells=[]    
      # get lick correlated cells per epoch
      coms_ep1 = coms_rewrel[0]
      bound=np.pi/4 # no bound of near/far cells
      ep_nearrew_cells = [np.where(((coms_rewrel[ep]>-bound) & (coms_rewrel[ep]<bound)))[0] for ep in range(len(coms_rewrel))]      
      # num of near rew cells
      num_nearrew=[len(xx)/len(coms_correct[0]) for xx in ep_nearrew_cells]
      lick_tc_cs = [[spearmanr(tcs_correct_abs[ep,cll,:], lick_correct_abs[ep][0])[0] for cll in per_ep_cll] for ep,per_ep_cll in enumerate(ep_nearrew_cells)]
      # get high correlated cells
      lick_corr_cells = [ep_nearrew_cell[lick_tc_cs[epep]>np.nanmean(lick_tc_cs[epep])] for epep, ep_nearrew_cell in enumerate(ep_nearrew_cells)]
      # overlap of lick corr and pre-reward cells
      # overlap_cells = [xx for xx in goal_cells if xx in lick_corr_cells[0]]
      # overlap_pre_in_lick = len(overlap_cells)/len(goal_cells)
      # overlap_cells = [xx for xx in lick_corr_cells[0] if xx in goal_cells]
      # overlap_lick_in_pre = len(overlap_cells)/len(lick_corr_cells[0])
      # activty of lick corr cells
      tcs_correct=tcs_correct[[optoep-2,optoep-1]]
      lick_corr_cells_ep1_act = np.nanmean(tcs_correct[:,lick_corr_cells_ep1],axis=2)
      # num lick corr cells
      num_lick_corr=[len(xx)/len(coms_correct[0]) for xx in lick_corr_cells]
      datadct[f'{animal}_{day}']=[rewlocs,rz,lick_tc_cs,lick_corr_cells,lick_corr_cells_ep1_act]
#%%
plt.rc('font', size=20)          # controls default text sizes

df=pd.DataFrame()
# df['rewzone'] = np.concatenate([v[1] for k,v in datadct.items()])
df['epoch']=np.concatenate([['prev', 'opto'] for k,v in datadct.items()])
df['activity_lick_corr_ep1'] = np.concatenate([np.nanmean(v[4],axis=1) for k,v in datadct.items()])
df['av_lick_corr'] = np.concatenate([[np.nanmedian(xx) for xx in v[2]] for k,v in datadct.items()])
df['animals']=np.concatenate([[kk.split('_')[0]]*2 for kk,v in datadct.items()])
df['days']=np.concatenate([[kk.split('_')[1]]*2 for kk,v in datadct.items()]).astype(int)
# df=df[df.num_lick_corr>0]
df = pd.merge(df, conddf, on=['animals', 'days'], how='left')
df['condition']=[xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
# get pre-opto days only for inhib
# df=df[~((df.in_type=='vip') & ((df.optoep>=0) & (df.optoep<2)))]
# df=df[df.epoch>1]

df=df[(df.animals!='e189')&(df.animals!='e190')]
# remove outlier days
df=df[~((df.animals=='z14')&((df.days<33)))]
df=df[~((df.animals=='z16')&((df.days>13)))]
# df=df[~((df.animals=='z17')&((df.days.isin([3,11]))))]
df=df[~((df.animals=='z15')&((df.days.isin([7,12,16]))))]
df=df[~((df.animals=='e217')&((df.days<9)|(df.days.isin([21,29,30,26,29]))))]
df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([47,55,57]))))]
df=df[~((df.animals=='e200')&((df.days.isin([67,68,81]))))]
# df=df[~((df.animals=='e218')&(df.days.isin([41,55])))]
# df=df[df.epoch>1]

df=df.groupby(['animals','days','condition','epoch']).mean(numeric_only=True).reset_index()
df=df.groupby(['animals','condition','epoch']).mean(numeric_only=True).reset_index()
hue_order=['prev','opto']
fig, ax = plt.subplots(figsize=(5,4))
pl = {'opto': 'grey', 'prev': 'k'}
sns.barplot(x='condition',y='av_lick_corr', hue='epoch',data=df,fill=False,errorbar='se',palette=pl,legend=False,hue_order=hue_order)
sns.stripplot(x='condition',y='av_lick_corr', hue='epoch',data=df,dodge=True,palette=pl,s=10,alpha=0.7,hue_order=hue_order)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('')
ax.set_xticklabels(['Control','VIP Inhibition', 'VIP Excitation'],rotation=20)
ax.set_ylabel('Mean lick correlation ($\\rho$)')

conds = df['condition'].unique()
x_locs = dict(zip(sorted(conds), range(len(conds))))  # 'ctrl': 0, 'vip_inh': 1, etc.

# Adjust these for the aesthetics of the comparison lines
line_offset = 0.02
line_height = 0.02
asterisk_y_offset = 0.005

# For saving the max y value (for proper line placement)
y_max = df['av_lick_corr'].max()

for cond in conds:
    subdf = df[df['condition'] == cond]
    pre = subdf[subdf['epoch'] == 'prev']['av_lick_corr'].values
    post = subdf[subdf['epoch'] == 'opto']['av_lick_corr'].values

    if len(pre) != len(post):
        continue  # skip if not matched pairs

    # Compute t-test
    tstat, pval = scipy.stats.ttest_rel(post, pre)  # post = opto, pre = prev
    print(f"Condition: {cond} | t = {tstat:.2f} | p = {pval:.4f} | n = {len(pre)}")

    x = x_locs[cond]
    height = max(np.nanmax(pre), np.nanmax(post)) + line_offset
    h = line_height

    # Draw line
    ax.plot([x - 0.2, x - 0.2, x + 0.2, x + 0.2], [height, height + h, height + h, height],
            lw=1.5, color='k')

    # Add significance asterisks
    if pval < 0.001:
        stars = '***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'

    ax.text(x, height + h + asterisk_y_offset, stars, ha='center', va='bottom', fontsize=16)

    # Optional: draw connecting lines per animal
    animals = subdf['animals'].unique()
    for animal in animals:
        a_pre = subdf[(subdf['animals'] == animal) & (subdf['epoch'] == 'prev')]['av_lick_corr'].values
        a_post = subdf[(subdf['animals'] == animal) & (subdf['epoch'] == 'opto')]['av_lick_corr'].values
        if len(a_pre) > 0 and len(a_post) > 0:
            ax.plot([x - 0.2, x + 0.2], [a_pre[0], a_post[0]], color='gray', alpha=0.5, lw=1.5)
plt.savefig(os.path.join(savedst, f'fig5_mean_lick_correlation.svg'), bbox_inches='tight')

#%%

fig, ax = plt.subplots()
pl = {'opto': 'red', 'prev': 'slategray'}
sns.barplot(x='condition',y='activity_lick_corr_ep1', hue='epoch',data=df,fill=False,errorbar='se',palette=pl)
sns.stripplot(x='condition',y='activity_lick_corr_ep1', hue='epoch',data=df,dodge=True,palette=pl)

conds = df['condition'].unique()
for cond in conds:
   subdf = df[df['condition'] == cond]
   pre = subdf[subdf['epoch'] == 'opto']['activity_lick_corr_ep1']
   post = subdf[subdf['epoch'] == 'prev']['activity_lick_corr_ep1']
   
   tstat, pval = scipy.stats.ttest_rel(pre, post)
   print(f"Condition: {cond} | t = {tstat:.2f} | p = {pval:.4f} | n_pre = {len(pre)}, n_post = {len(post)}")
ax.set_ylabel('$\Delta F/F$ lick correlated cells from ep 1')