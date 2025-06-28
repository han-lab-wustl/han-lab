
"""
zahra
get tuning curves with dark time
reward cell p vs. behavior
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
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle, \
    intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, get_lick_selectivity,smooth_lick_rate
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

#%%
####################################### RUN CODE #######################################
# initialize var
radian_alignment_saved = {} # overwrite
bins = 90
dfs = []
cm_window = 20 # cm
num_iterations=1000
#%%
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (conddf.optoep.values[ii]<2):        
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF','putative_pcs','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
               'stat', 'licks'])
      pcs=np.squeeze(np.sum(pcs,axis=0)>0)
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
      licks=fall['licks'][0]
      time=fall['timedFF'][0]
      fr = 31.25
      if animal=='e190' or animal=='z9':
         fr=fr/2
      if animal=='e145':
         ybinned=ybinned[:-1]
         forwardvel=forwardvel[:-1]
         changeRewLoc=changeRewLoc[:-1]
         trialnum=trialnum[:-1]
         rewards=rewards[:-1]
         licks=licks[:-1]
         time=time[:-1]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      track_length_rad = track_length*(2*np.pi/track_length)
      bin_size=track_length_rad/bins 
      rz = get_rewzones(rewlocs,1/scalingf)       
      # get average success rate
      lasttr = 8
      rates = []; ls_all = []; lr_all = []
      for ep in range(len(eps)-1):
         eprng = range(eps[ep],eps[ep+1])
         success, fail, str_trials, ftr_trials, ttr, \
         total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         rates.append(success/total_trials)
         # lick rate and selecitivty
         mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum[eprng]])
         ls = get_lick_selectivity(ybinned[eprng][mask], trialnum[eprng][mask], licks[eprng][mask], rewlocs[ep], rewsize,
                  fails_only = False)
         lr = smooth_lick_rate(licks[eprng][mask], 1/fr, sigma_sec=0.7)
         ls_all.append(np.nanmean(ls))
         lr_all.append(np.nanmean(lr))
      rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
      # added to get anatomical info
      # takes time
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      # get spatially tuned cells
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      dFF = dFF[:, skew>2] # only keep cells with skew greateer than 2
      Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
      # tc w/ dark time
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,relpos = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt)  
      goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
      # change to relative value 
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      perm_real = list(combinations(range(len(coms_correct)), 2)) 
      rz_perm_real = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm_real]   
      # account for cells that move to the end/front
      # Define a small window around pi (e.g., epsilon)
      epsilon = .7 # 20 cm
      # Find COMs near pi and shift to -pi
      com_loop_w_in_window = []
      for pi,p in enumerate(perm_real):
         for cll in range(coms_rewrel.shape[1]):
               com1_rel = coms_rewrel[p[0],cll]
               com2_rel = coms_rewrel[p[1],cll]
               # print(com1_rel,com2_rel,com_diff)
               if ((abs(com1_rel - np.pi) < epsilon) and 
               (abs(com2_rel + np.pi) < epsilon)):
                     com_loop_w_in_window.append(cll)
      # get abs value instead
      coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
      com_remap = np.array([(coms_rewrel[perm_real[jj][0]]-coms_rewrel[perm_real[jj][1]]) for jj in range(len(perm_real))])        
      com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
      com_goal = [com if len(com) > 0 else [] for com in com_goal]
      # any goal cell
      goal_cells = np.unique(np.concatenate(com_goal)).astype(int)
      dff_goal = np.nanmean(dFF[:,goal_cells],axis=0) 
      other_cells = np.array([cll for cll in np.arange(Fc3.shape[1]) if cll not in goal_cells]).astype(int)           
      dff_other = np.nanmean(dFF[:,other_cells],axis=0)
      df=pd.DataFrame()
      df['meandff']= np.concatenate([dff_goal,dff_other])
      df['type'] = np.concatenate([['rew']*len(dff_goal),['other']*len(dff_other)])
      df['animal']=[animal]*len(df)
      df['day']=[day]*len(df)
      sns.stripplot(x='type',y='meandff',data=df)
      sns.barplot(x='type',y='meandff',data=df)
      dfs.append(df)


####################################### RUN CODE #######################################
#%%
plt.close('all')
bigdf=pd.concat(dfs)

# bigdf=bigdf.groupby(['animal','type']).mean(numeric_only=True).reset_index()
for an in bigdf.animal.unique():
   plt.figure()
   sns.boxplot(x='type', y='meandff',data=bigdf[bigdf.animal==an])
   plt.title(an)