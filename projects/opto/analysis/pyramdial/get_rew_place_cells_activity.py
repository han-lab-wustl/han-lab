"""get place cells between opto and non opto conditions
april 2025
"""

#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from statsmodels.formula.api import ols
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import itertools
from statsmodels.stats.anova import anova_lm  # <-- Correct import
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves, make_tuning_curves_early, intersect_arrays, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_by_trialtype_w_darktime_early
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_place.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\place_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
      datadct = pickle.load(fp)
# initialize var
datadct = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
place_cell_null=[]
other_sp_prop=[]
place_window = 20
num_iterations=1000
bin_size=3 # cm
lasttr=8 # last trials
bins=90

# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   optoep = conddf.optoep.values[ii]
   if ii!=179 and optoep>1:
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

      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      # looser restrictions
      pc_bool = np.sum(pcs,axis=0)>=1        
      if animal=='e200' or animal=='e217' or animal=='z17' or animal=='z14':
         Fc3 = Fc3[:,((skew>2))]
         dFF = dFF[:,((skew>2))]
      else:
         Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greater than 2
         dFF = dFF[:,((skew>2)&pc_bool)]
      if Fc3.shape[1]>0:
         # get abs dist tuning 
         if sum([f'{animal}_{day:03d}' in xx for xx in list(datadct.keys())])>0:
               k = [k for k,xx in datadct.items() if f'{animal}_{day:03d}' in k][0]
               tcs_correct_abs, coms_correct_abs,tcs_fail_abs,coms_fail_abs, tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early,pcs_all=datadct[k]
         else:
               print('#############making tcs#############\n')
               tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
               Fc3,trialnum,rewards,forwardvel,
               rewsize,bin_size) # last 5 trials
               tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early = make_tuning_curves_early(eps,rewlocs,ybinned, Fc3,trialnum,rewards,forwardvel,
               rewsize,bin_size) # last 5 trials

         track_length_dt = 550 # cm estimate based on 99.9% of ypos
         track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
         bins_dt=150 
         bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
         tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
               bins=bins_dt,lasttr=8) 
         tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 


         # get cells that maintain their coms b/wn previous and opto ep
         perm = [(eptest-2, eptest-1)]   
         if perm[0][1]<len(coms_correct_abs): # make sure tested epoch has enough trials
               print(eptest, perm)            
               goal_window = 20*(2*np.pi/track_length) # cm converted to rad
               coms_rewrel = np.array([com-np.pi for com in coms_correct])
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
               com_goal=[xx for xx in com_goal if len(xx)>0]
               if len(com_goal)>0:
                  goal_cells = intersect_arrays(*com_goal)
               else:
                  goal_cells=[]
               # early goal cells
               coms_rewrel = np.array([com-np.pi for com in coms_correct_early])
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
               com_goal=[xx for xx in com_goal if len(xx)>0]
               if len(com_goal)>0:
                  goal_cells_early = intersect_arrays(*com_goal)
               else:
                  goal_cells_early=[]

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
               pcs_early = np.unique(np.concatenate(compc))
               compc=[xx for xx in compc if len(xx)>0]
               if len(compc)>0:
                  pcs_all_early = intersect_arrays(*compc)
                  # exclude goal cells
                  pcs_all_early=[xx for xx in pcs_all if xx not in goal_cells_early]
               else:
                  pcs_all_early=[]      
               # get per comparison
               pcs_p_per_comparison_early = [len(xx)/len(coms_correct_abs_early[0]) for xx in compc]
               pc_p_early=len(pcs_all_early)/len(coms_correct_abs_early[0])
               # get other spatially tuned cells
               other_sp = [xx for xx in np.arange(Fc3.shape[1]) if xx not in pcs_all_early and xx not in pcs_all and xx not in goal_cells_early and xx not in goal_cells]
               other_sp_prop.append(len(other_sp)/len(coms_correct[0]))
               # print props
               print(len(other_sp)/len(coms_correct[0]), pc_p, len(goal_cells)/len(coms_correct[0]))
               epoch_perm.append(perm)
               pc_prop.append([pcs_p_per_comparison,pc_p,pcs_p_per_comparison_early,pc_p_early])
               num_epochs.append(len(coms_correct_abs))
               # get activity
               dff_pc_prev = dFF[eptest-2:eptest-1, pcs_all]
               dff_pc_opto = dFF[eptest-1:eptest, pcs_all]               
               dff_gc_prev = dFF[eptest-2:eptest-1, goal_cells]
               dff_gc_opto = dFF[eptest-1:eptest, goal_cells]
               datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs, dff_pc_prev,dff_pc_opto,dff_gc_prev,dff_gc_opto,tcs_correct[[eptest-2,eptest-1],:],coms_correct[[eptest-2,eptest-1],:],pcs_all,goal_cells]
pdf.close()
# # save pickle of dcts
#%%
# visualize
plt.rc('font', size=20)          # controls default text sizes
from matplotlib.colors import LinearSegmentedColormap

# Example: Define the color points for Parula (simplified for illustration)
# The actual Parula colormap has more precise color definitions.
# You would typically get these from a source that provides Parula's RGB data.
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
[0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
[0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
[0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
[0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
[0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
[0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
[0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
[0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
[0.0589714286, 0.6837571429, 0.7253857143], 
[0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
[0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
[0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
[0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
[0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
[0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
[0.7184095238, 0.7411333333, 0.3904761905], 
[0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
[0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
[0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
[0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
[0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
[0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
[0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
[0.9763, 0.9831, 0.0538]]

parula_cmap = LinearSegmentedColormap.from_list('parula', cm_data)
vmax=4
an = [k.split('_')[0] for jj,(k,v) in enumerate(datadct.items())]
dy = [int(k.split('_')[1]) for jj,(k,v) in enumerate(datadct.items())]
lookup_df = pd.DataFrame({'animals': an, 'days': dy})
# Select matching rows from conddf
matching_conddf = pd.merge(lookup_df, conddf, on=['animals', 'days'], how='inner')

# ctrl
tcs = [v[8] for jj,(k,v) in enumerate(datadct.items()) if 'vip' not in matching_conddf.iloc[jj].in_type and matching_conddf.iloc[jj].optoep>1]
coms = [v[9] for jj,(k,v) in enumerate(datadct.items()) if 'vip' not in matching_conddf.iloc[jj].in_type and matching_conddf.iloc[jj].optoep>1]
pcs = [v[10] for jj,(k,v) in enumerate(datadct.items()) if 'vip' not in matching_conddf.iloc[jj].in_type and matching_conddf.iloc[jj].optoep>1]
gcs = [v[11] for jj,(k,v) in enumerate(datadct.items()) if 'vip' not in matching_conddf.iloc[jj].in_type and matching_conddf.iloc[jj].optoep>1]

fig,axes = plt.subplots(figsize=(20,8),nrows=2,ncols=6,sharex=True)
for ep in range(2):
   place_tc = np.vstack([tc[ep,pcs[jjj]] for jjj,tc in enumerate(tcs) if len(pcs[jjj])>0])
   rew_tc = np.vstack([tc[ep,gcs[jjj]] for jjj,tc in enumerate(tcs) if len(gcs[jjj])>0])
   place_com = np.hstack([com[ep,pcs[jjj]] for jjj,com in enumerate(coms) if len(pcs[jjj])>0])
   rew_com = np.hstack([com[ep,gcs[jjj]] for jjj,com in enumerate(coms) if len(gcs[jjj])>0])
   ax=axes[ep,0]
   ax.imshow(place_tc[np.argsort(place_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(place_tc)-1])
   ax.set_yticklabels([1,len(place_tc)])

   ax=axes[ep,1]
   ax.imshow(rew_tc[np.argsort(rew_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(rew_tc)-1])
   ax.set_yticklabels([1,len(rew_tc)])
axes[0,0].set_title('Control\nPlace')
axes[0,1].set_title('\nReward')
# inhib
tcs = [v[8] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip']
coms = [v[9] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip']
pcs = [v[10] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip']
gcs = [v[11] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip']

# fig,axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey='col')
for ep in range(2):
   place_tc = np.vstack([tc[ep,pcs[jjj]] for jjj,tc in enumerate(tcs) if len(pcs[jjj])>0])
   rew_tc = np.vstack([tc[ep,gcs[jjj]] for jjj,tc in enumerate(tcs) if len(gcs[jjj])>0])
   place_com = np.hstack([com[ep,pcs[jjj]] for jjj,com in enumerate(coms) if len(pcs[jjj])>0])
   rew_com = np.hstack([com[ep,gcs[jjj]] for jjj,com in enumerate(coms) if len(gcs[jjj])>0])
   ax=axes[ep,2]
   ax.imshow(place_tc[np.argsort(place_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(place_tc)-1])
   ax.set_yticklabels([1,len(place_tc)])

   ax=axes[ep,3]
   ax.imshow(rew_tc[np.argsort(rew_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(rew_tc)-1])
   ax.set_yticklabels([1,len(rew_tc)])
axes[0,2].set_title('VIP Inhibition\nPlace')
axes[0,3].set_title('\nReward')


# excite
tcs = [v[8] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip_ex']
coms = [v[9] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip_ex']
pcs = [v[10] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip_ex']
gcs = [v[11] for jj,(k,v) in enumerate(datadct.items()) if matching_conddf.iloc[jj].in_type=='vip_ex']

for ep in range(2):
   place_tc = np.vstack([tc[ep,pcs[jjj]] for jjj,tc in enumerate(tcs) if len(pcs[jjj])>0])
   rew_tc = np.vstack([tc[ep,gcs[jjj]] for jjj,tc in enumerate(tcs) if len(gcs[jjj])>0])
   place_com = np.hstack([com[ep,pcs[jjj]] for jjj,com in enumerate(coms) if len(pcs[jjj])>0])
   rew_com = np.hstack([com[ep,gcs[jjj]] for jjj,com in enumerate(coms) if len(gcs[jjj])>0])
   ax=axes[ep,4]
   ax.imshow(place_tc[np.argsort(place_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(place_tc)-1])
   ax.set_yticklabels([1,len(place_tc)])
   ax=axes[ep,5]
   ax.imshow(rew_tc[np.argsort(rew_com)],aspect='auto',cmap=parula_cmap,vmax=vmax)
   ax.axvline(75,color='w',linestyle='--',linewidth=2)
   ax.set_yticks([0,len(rew_tc)-1])
   ax.set_yticklabels([1,len(rew_tc)])
ax.set_xticks([0,75,149])
ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
axes[0,4].set_title('VIP Excitation\nPlace')
axes[0,5].set_title('\nReward')

im = ax.imshow(rew_tc[np.argsort(rew_com)], aspect='auto', cmap=parula_cmap,vmax=vmax)
# Create a new axes for the colorbar to the right of all plots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.99, 0.5, 0.01, 0.3])  # adjust as needed
# Add colorbar to that axes
fig.colorbar(im, cax=cbar_ax, label='$\Delta F/F$')  # or 'dF/F' etc.
# fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.1, label='$\Delta F/F$')
axes[1,0].set_xlabel('Reward-centric distance ($\Theta$)')
fig.suptitle('All cell tuning curves')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'all_cell_eg.svg'), bbox_inches='tight')

#%%
# quantify 
# # just opto days
s=12
cdf = conddf.copy()
inds = [int(xx[-3:]) for xx in datadct.keys()]
cdf = cdf[(cdf.index.isin(inds))]
df = pd.DataFrame()
df['dff_pc_prev'] = [np.nanquantile(v[4],.75) for k,v in datadct.items()]
df['dff_pc_opto'] = [np.nanquantile(v[5],.75) for k,v in datadct.items()]
df['dff_gc_prev'] = [np.nanquantile(v[6],.75) for k,v in datadct.items()]
df['dff_gc_opto'] = [np.nanquantile(v[7],.75) for k,v in datadct.items()]
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in cdf.in_type.values]
df['animals'] = cdf.animals.values
df['days'] = cdf.days.values
df = pd.merge(df, cdf, on=['animals', 'days'], how='inner')

df=df[(df.animals!='e189')&(df.animals!='e190')]
# remove outlier days
# df=df[~((df.animals=='e201')&((df.days>62)))]
df=df[~((df.animals=='z14')&((df.days<33)))]
# df=df[~((df.animals=='z16')&((df.days>15)))]
df=df[~((df.animals=='z17')&((df.days<15)|(df.days.isin([3,4,5,9,18]))))]
df=df[~((df.animals=='z15')&((df.days<8)|(df.days.isin([15]))))]
df=df[~((df.animals=='e217')&((df.days<2)|(df.days.isin([21,29,30]))))]
# df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([57]))))]
df['delta_dff_pc'] = df['dff_pc_opto'] - df['dff_pc_prev']
df['delta_dff_gc'] = df['dff_gc_opto'] - df['dff_gc_prev']

df=df[df.optoep>1]
# --- Reshape to long format with paired LED off/on values ---
pc_data = df[['animals','days', 'condition', 'dff_pc_prev', 'dff_pc_opto']].copy()
pc_data['cell_type'] = 'pc'
gc_data = df[['animals', 'days','condition', 'dff_gc_prev', 'dff_gc_opto']].copy()
gc_data['cell_type'] = 'gc'

# Rename for melting
pc_data = pc_data.rename(columns={'dff_pc_prev': 'led_off', 'dff_pc_opto': 'led_on'})
gc_data = gc_data.rename(columns={'dff_gc_prev': 'led_off', 'dff_gc_opto': 'led_on'})

# Combine
dff_long = pd.concat([pc_data, gc_data], axis=0)
# Melt to long format for seaborn
dff_melted = pd.melt(
   dff_long,
   id_vars=['animals', 'condition', 'cell_type'],
   value_vars=['led_off', 'led_on'],
   var_name='led_state',
   value_name='dff'
)
# Compute difference: LED on - LED off
dff_long['delta_dff'] = dff_long['led_on'] - dff_long['led_off']
# Keep only needed columns
delta_df = dff_long[['animals', 'days','condition', 'cell_type', 'delta_dff']].copy().reset_index()

# Ensure LED state order
dff_melted['led_state'] = pd.Categorical(dff_melted['led_state'], categories=['led_off', 'led_on'])
dff_melted['cond_cell'] = dff_melted['led_state'].astype(str) + '_' + dff_melted['cell_type']

# Plot
pl=['k','indigo','k','cornflowerblue']
# Extract clean labels
df_long=dff_melted.groupby(['animals','condition','cond_cell']).mean(numeric_only=True).reset_index()
hue=['led_off_pc','led_on_pc','led_off_gc','led_on_gc']
fig,axes=plt.subplots(ncols=2,figsize=(12,5),width_ratios=[2,1])
ax=axes[0]
sns.barplot(data=df_long, x='condition', y='dff', hue='cond_cell', fill=False,legend=False,errorbar='se',hue_order=hue,palette=pl,ax=ax)
sns.stripplot(data=df_long, x='condition', y='dff', hue='cond_cell', s=10,dodge=True, alpha=0.7,hue_order=hue,palette=pl,ax=ax)
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'])
sns.despine()
ax.set_ylabel('Mean ΔF/F')
ax=axes[1]
delta_df=delta_df.groupby(['animals','condition','cell_type']).mean(numeric_only=True).reset_index()
# Barplot with error bars
hue=['pc','gc']
ax = sns.barplot(
    data=delta_df,
    x='condition',
    y='delta_dff',hue_order=hue,
    hue='cell_type',
    palette=['indigo', 'cornflowerblue'],
    errorbar='se',ax=ax,fill=False,legend=False,
)

# Overlay stripplot for individual animals
sns.stripplot(
    data=delta_df,
    x='condition',
    y='delta_dff',
    hue='cell_type',
    dodge=True,
    alpha=0.7,hue_order=hue,
    size=7,
    palette=['indigo', 'cornflowerblue'],
    marker='o',
    linewidth=0.5,
    ax=ax
)
# Draw paired lines between 'pc' and 'gc' for each animal within a condition
for condition in delta_df['condition'].unique():
   subset = delta_df[delta_df['condition'] == condition]
   animals = subset['animals'].unique()
   for animal in animals:
      animal_data = subset[subset['animals'] == animal]
      if len(animal_data) == 2:
         # Match hue_order index to x-axis positions
         x_vals = [-.2,.2] if hue[0] in animal_data['cell_type'].values else [0,.4]
         y_vals = animal_data.sort_values('cell_type', key=lambda x: x.map({hue[0]: 0, hue[1]: 1}))['delta_dff'].values
         # x-offset by condition
         x_offset = list(delta_df['condition'].unique()).index(condition)/2
         x_vals = [x + x_offset * 2 for x in x_vals]  # assuming dodge separates bars by 1
         plt.plot(x_vals, y_vals, color='gray', alpha=0.5, linewidth=1.5, zorder=0)

# Clean up
ax.axhline(0, color='k', linestyle='--')
ax.set_ylabel('Mean ΔF/F (LEDon−LEDoff)')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'],rotation=20)
sns.despine()
plt.legend(title='Cell type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
from scipy.stats import ttest_1samp
from scipy.stats import ttest_1samp

# Store positions and results
signif_results = {}

for condition in delta_df['condition'].unique():
   for cell in ['pc', 'gc']:
      sub = delta_df[(delta_df['condition'] == condition) & (delta_df['cell_type'] == cell)]
      if len(sub) > 1:
         t_stat, p = ttest_1samp(sub['delta_dff'], popmean=0)
         signif_results[(condition, cell)] = p
# Find the maximum bar height per (condition, cell_type)
grouped = delta_df.groupby(['condition', 'cell_type'])['delta_dff'].mean()

# Mapping x-tick positions (dodge logic)
conditions = delta_df['condition'].unique()
cell_types = ['pc', 'gc']
bar_spacing = 0.8  # between bars of different cell types
group_spacing = 2  # between groups (conditions)

for i, condition in enumerate(conditions):
   for j, cell in enumerate(cell_types):
      pval = signif_results.get((condition, cell))
      if pval is None:
         continue
      # Convert to stars
      if pval < 0.001:
         stars = '***'
      elif pval < 0.01:
         stars = '**'
      elif pval < 0.05:
         stars = '*'
      else:
         stars = ''  # Not significant

      # Compute bar x position (match your dodge logic!)
      x = i * group_spacing/2 + j * bar_spacing/3 +.2
      # Get bar height
      y = grouped.get((condition, cell), 0)
      # Annotate
      ax.annotate(
         stars,
         xy=(x, y + 0.01),  # 0.01 offset above bar
         ha='center',
         va='bottom',
         fontsize=16,
         color='black'
      )
      # Annotate
      ax.annotate(
         f'p={np.round(pval,3)}',
         xy=(x, y + 0.01),  # 0.01 offset above bar
         ha='center',
         va='bottom',
         fontsize=16,
         color='black'
      )

plt.savefig(os.path.join(savedst, 'ledoff_v_ledon_rew_place_dff.svg'), bbox_inches='tight')
