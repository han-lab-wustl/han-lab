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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew,intersect_arrays
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves_by_trialtype_w_darktime_early,make_tuning_curves

import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
#%%
ii=47
cm_window=20
day = int(conddf.days.values[ii])
animal = conddf.animals.values[ii]
if animal=='e145': pln=2  
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
eptest=int(eptest)   
lasttr=8 # last trials
bins=90
rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
track_length_rad = track_length*(2*np.pi/track_length)
bin_size=track_length_rad/bins

if True:  
# takes time
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   # for inhibi
   # Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
   # dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]

   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   # if animal!='z14' and animal!='e200' and animal!='e189':                
   Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
   # tc w/ dark time
   print('making tuning curves...\n')
   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt,lasttr=8) 
   tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
   Fc3,trialnum,rewards,forwardvel,
   rewsize,bin_size) # last 5 trials

   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 
   # early tc
   tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)  
   goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
   coms_rewrel = np.array([com-np.pi for com in coms_correct])
   perm = list(combinations(range(len(coms_correct)), 2)) 
   print(perm)
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
   c = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
   goal_cells = intersect_arrays(*com_goal) if len(com_goal) > 0 else []    

#%%
# place v rew v far rew
def moving_average(x, window_size=3):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')
plt.rc('font', size=16) 
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
lbls=['Place', 'Pre-reward',  'Post-reward','Far post-reward','Reward-aligned']
fig,axes=plt.subplots(ncols=5,figsize=(16,2.5),sharey=True)
clls=[2,118,9,50]
for ll,cll in enumerate(clls):
   ax=axes[ll]
   for ep in range(len(tcs_correct_abs)):
      pltt = moving_average(tcs_correct_abs[ep,cll,:])
      ax.plot(pltt,color=colors[ep],linewidth=2)
      ax.axvline(rewlocs[ep]/3,color=colors[ep],linestyle='--')
   if ll==0:
      ax.set_ylabel('$\Delta$ F/F')
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title(lbls[ll])
ax.set_xlabel('Track position (cm)')
ax.set_xticks([0,90])
ax.set_xticklabels([0,270])
ax=axes[4]
for ep in range(len(tcs_correct_abs)):
   pltt = moving_average(tcs_correct[ep,cll,:])
   ax.plot(pltt,color=colors[ep],linewidth=2)
ax.set_title(lbls[4])
ax.axvline(75,color='k',linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('\nReward-relative distance ($\Theta$)')
ax.set_xticks([0,75,150])
ax.set_xticklabels(['$-\pi$',0,'$\pi$'])
plt.savefig(os.path.join(savedst, 'fig1_tc.svg'), bbox_inches='tight')
