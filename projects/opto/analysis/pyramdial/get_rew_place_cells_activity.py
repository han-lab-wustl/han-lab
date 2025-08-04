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
         Fc3 = Fc3[:,((skew>1.5))]
         dFF = dFF[:,((skew>1.5))]
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
               datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs, dff_pc_prev,dff_pc_opto,dff_gc_prev,dff_gc_opto]
pdf.close()
# # save pickle of dcts
#%%

plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
# just opto days
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
df=df[~((df.animals=='e217')&((df.days<2)|(df.days.isin([29]))))]
# df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([57]))))]
df['delta_dff_pc'] = df['dff_pc_opto'] - df['dff_pc_prev']
df['delta_dff_gc'] = df['dff_gc_opto'] - df['dff_gc_prev']
df=df[df.optoep>1]
# --- Reshape to long format with paired LED off/on values ---
pc_data = df[['animals', 'condition', 'dff_pc_prev', 'dff_pc_opto']].copy()
pc_data['cell_type'] = 'pc'
gc_data = df[['animals', 'condition', 'dff_gc_prev', 'dff_gc_opto']].copy()
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

# Ensure LED state order
dff_melted['led_state'] = pd.Categorical(dff_melted['led_state'], categories=['led_off', 'led_on'])
dff_melted['cond_cell'] = dff_melted['led_state'].astype(str) + '_' + dff_melted['cell_type']

# Plot
pl=['k','indigo','k','cornflowerblue']
# Extract clean labels
df_long=dff_melted.groupby(['animals','condition','cond_cell']).mean(numeric_only=True).reset_index()
hue=['led_off_pc','led_on_pc','led_off_gc','led_on_gc']
fig,ax=plt.subplots(figsize=(5,5))

sns.barplot(data=df_long, x='condition', y='dff', hue='cond_cell', fill=False,legend=False,errorbar='se',hue_order=hue,palette=pl,ax=ax)
sns.stripplot(data=df_long, x='condition', y='dff', hue='cond_cell', s=6,dodge=True, alpha=0.7,hue_order=hue,palette=pl,ax=ax)
ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
sns.despine()
plt.grid(False)  # Removes the grid
plt.axhline(0, color='k', linestyle='--')
plt.ylabel('Mean ΔF/F (LEDon-LEDoff)')
plt.savefig(os.path.join(savedst, 'ledoff_v_ledon_rew_place_dff.svg'), bbox_inches='tight')
