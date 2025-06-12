"""
get reward distance cells between opto and non opto conditions
oct 2024
mods in june 2025
control vs. opto epoch only
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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\place_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        datadct = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
datadct_new={}
cm_window = 20

#%%
# iterate through all animals
for ii in range(len(conddf)):
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   # skip e217 day
   if ii!=179:
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
      if sum([f'{animal}_{day:03d}' in xx for xx in list(datadct.keys())])>0:
            k = [k for k,xx in datadct.items() if f'{animal}_{day:03d}' in k][0]
            tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs, pcs=datadct[k]
      # get cells that maintain their coms b/wn previous and opto ep
      perm = (eptest-2, eptest-1)   
      if perm[1]<len(coms_correct_abs): # make sure tested epoch has enough trials
         coms_rewrel = np.array([com-rewlocs[ep] for ep,com in enumerate(coms_correct_abs)])
         com_per_ep = coms_rewrel[perm[0],:]-coms_rewrel[perm[1],:]
         compc = np.where((com_per_ep<cm_window) & (com_per_ep>-cm_window))[0]
         # get cells across OPTO VS. CONTROL EPOCHS
         goal_cells=compc
      else:
         goal_cells=[]
      datadct_new[f'{animal}_{day:03d}_index{ii:03d}'] = [coms_rewrel,goal_cells]

            # early tc
            # tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)        


#%%

# top down approach
# 1) com dist in opto vs. control
# 3) place v. reward
# tcs_correct, coms_correct, tcs_fail, coms_fail,pcs
# 1) get coms correct
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
# df=df[conddf.animals!='e217']
coms_correct = [xx[0] for k,xx in datadct_new.items()]
pcs = [xx[1] for k,xx in datadct_new.items()]
optoep = [xx if xx>1 else 2 for xx in df.optoep.values]
# opto comparison
# tcs_correct = [xx[[optoep[ep]-2,optoep[ep]-1],:] for ep,xx in enumerate(tcs_correct)]
coms_correct_prev = [xx[0] for ep,xx in enumerate(coms_correct)]
coms_correct_opto = [xx[1] for ep,xx in enumerate(coms_correct)]
# tcs_correct_prev = [xx[0] for ep,xx in enumerate(tcs_correct)]
# tcs_correct_opto = [xx[1] for ep,xx in enumerate(tcs_correct)]

vip_in_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]>1))]
vip_in_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]>1))]
vip_in_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]==-1))]
vip_in_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]==-1))]
# excitation
vip_ex_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]>1))]
vip_ex_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]>1))]
vip_ex_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]<1))]
vip_ex_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]<1))]
#control
ctrl_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]>1))]
ctrl_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]>1))]
ctrl_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]<1))]
ctrl_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]<1))]
#%%
plots = [[ctrl_com_prev,vip_in_com_prev,vip_ex_com_prev,ctrl_com_ctrl_prev,vip_in_com_ctrl_prev,vip_ex_com_ctrl_prev],
        [ctrl_com_opto,vip_in_com_opto,vip_ex_com_opto,ctrl_com_ctrl_opto,vip_in_com_ctrl_opto,vip_ex_com_ctrl_opto]]
lbls=['ctrl_ledon','vip_in_ledon','vip_ex_ledon','ctrl_ledoff','vip_in_ledoff','vip_ex_ledoff']
a=0.4
fig,axes=plt.subplots(ncols=3,nrows=2,figsize=(17,10))
axes=axes.flatten()
for pl in range(len(plots[0])):
    ax=axes[pl]
    # Concatenate and subtract pi
    data_prev = np.concatenate(plots[0][pl])
    data_opto = np.concatenate(plots[1][pl])
   #  ax.hist(data_prev,alpha=a,label='prev_ep',density=True)
   #  ax.hist(data_opto,alpha=a,label='opto_ep',density=True)
    # KDE plots
    sns.kdeplot(data_prev, ax=ax, label='prev_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    sns.kdeplot(data_opto, ax=ax, label='opto_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    ax.set_title(lbls[pl])
#     ax.set_xlim([-np.pi/6,np.pi])
    ax.axvline(0, color='gray', linewidth=2,linestyle='--')
ax.legend()
#%%
# 1. Get tuning curves for place cells only
tcs_correct = [xx[0] for k, xx in datadct.items()]  # get full tcs (trials x cells x bins)
optoep = [xx if xx > 1 else 2 for xx in df.optoep.values]
# 2. Extract LEDoff vs. LEDon activity for place cells
tcs_placecells = []
for ep, tc in enumerate(tcs_correct):
    ep_use = optoep[ep]
    if len(tc) > ep_use - 1:
        trial_idx = [ep_use - 2, ep_use - 1]  # previous and opto epochs
    else:
        trial_idx = [ep_use - 3, ep_use - 2]
    pcs_ep = pcs[ep]
    # shape: (2 epochs, n_place_cells, n_bins)
    tc_subset = tc[trial_idx, :, :][:, pcs_ep, :]
    tcs_placecells.append(tc_subset)

# 3. Separate into LEDoff (prev) and LEDon (opto)
tcs_placecells_prev = [xx[0] for xx in tcs_placecells]  # list of (n_cells, n_bins)
tcs_placecells_opto = [xx[1] for xx in tcs_placecells]
# Get mean activity across bins per cell (per animal)
mean_prev = [np.trapz(x, axis=1) for x in tcs_placecells_prev]
mean_opto = [np.trapz(x, axis=1) for x in tcs_placecells_opto]

# Difference LEDon - LEDoff
mean_diff = [o - 0 for o, p in zip(mean_opto, mean_prev)]
def group_vals(vals, cond):
    return [vals[i] for i in range(len(vals)) if cond(i)]

# Define masks
vip_in_mask = (df.in_type == 'vip') & (df.optoep > 1)
vip_ex_mask = (df.in_type == 'vip_ex') & (df.optoep > 1)
ctrl_mask = (~df.in_type.str.contains('vip')) & (df.optoep > 1)
# Extract mean activity diffs (LEDon - LEDoff) per group
vip_in_diffs = group_vals(mean_diff, lambda i: vip_in_mask.values[i])
vip_ex_diffs = group_vals(mean_diff, lambda i: vip_ex_mask.values[i])
ctrl_diffs = group_vals(mean_diff, lambda i: ctrl_mask.values[i])

# Flatten
vip_in_vals = np.concatenate(vip_in_diffs)
vip_ex_vals = np.concatenate(vip_ex_diffs)
ctrl_vals = np.concatenate(ctrl_diffs)
import pandas as pd

# Combine into DataFrame
all_data = pd.DataFrame({
    'change_in_activity': np.concatenate([ctrl_vals, vip_in_vals, vip_ex_vals]),
    'group': ['ctrl'] * len(ctrl_vals) + ['vip_in'] * len(vip_in_vals) + ['vip_ex'] * len(vip_ex_vals)
})

# Plot
plt.figure(figsize=(8,6))
# sns.stripplot(data=all_data, x='group', y='change_in_activity', palette='Set2')
sns.barplot(data=all_data, x='group', y='change_in_activity', palette='Set2')
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('LEDon - LEDoff\n(mean place cell activity)')
plt.title('VIP Modulation of Rew Cell Activity on Correct Trials')
plt.tight_layout()
#%%
# `mean_diff` is a list of arrays (n_place_cells x 1) for each animal
# `df.animals` gives you animal IDs
# Prepare labels for each individual place cell
group_labels = []
animal_labels = []
change_vals = []
days=[]
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
df=df.reset_index()
for i in range(len(mean_diff)):
    if ctrl_mask.values[i]:
        group = 'ctrl'
    elif vip_in_mask.values[i]:
        group = 'vip_in'
    elif vip_ex_mask.values[i]:
        group = 'vip_ex'
    else:
        continue  # skip if not part of any defined group
    animal = df.animals.iloc[i]
    day=df.days.iloc[i]
    n_cells = len(mean_diff[i])
    change_vals.extend(mean_diff[i])
    group_labels.extend([group] * n_cells)
    animal_labels.extend([animal] * n_cells)
    days.extend([day]*n_cells)

# Create DataFrame
all_data = pd.DataFrame({
    'change_in_activity': change_vals,
    'group': group_labels,
    'animal': animal_labels,
    'day': days
})
all_data=all_data[(all_data.animal!='e189')&(all_data.animal!='e190')]
# remove outlier days
all_data=all_data[~((all_data.animal=='z14')&(all_data.day<15))]
all_data=all_data[~((all_data.animal=='z15')&(all_data.day<12))]
all_data=all_data[~((all_data.animal=='e217')&(all_data.day<9))]
all_data=all_data[~((all_data.animal=='e216')&(all_data.day<32))]
all_data=all_data[~((all_data.animal=='e218')&(all_data.day>44))]

# Group by animal for average per-animal effect
per_animal = all_data.groupby(['animal', 'group'])['change_in_activity'].mean().reset_index()
plt.figure(figsize=(3,5))
sns.boxplot(data=per_animal, x='group', y='change_in_activity', palette='Set2')
sns.stripplot(data=per_animal, x='group', y='change_in_activity', color='black', alpha=0.7)
# plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Per-Animal Mean Δ Rew Cell Activity')
plt.title('VIP Modulation of Rew Cell Activity (Averaged per Animal)')
plt.tight_layout()
from scipy.stats import ranksums
import itertools
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit the model
model = ols('change_in_activity ~ group', data=per_animal).fit()
# Run ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
from scipy.stats import ttest_ind

# Grouped values
ctrl_vals = per_animal[per_animal['group'] == 'ctrl']['change_in_activity']
vip_in_vals = per_animal[per_animal['group'] == 'vip_in']['change_in_activity']
vip_ex_vals = per_animal[per_animal['group'] == 'vip_ex']['change_in_activity']

# Run independent t-tests
print("Post hoc Student's t-tests (unpaired):")
t1, p1 = ttest_ind(ctrl_vals, vip_in_vals)
print(f"ctrl vs vip_in: t = {t1:.3f}, p = {p1:.4f}, n = {len(ctrl_vals)} vs {len(vip_in_vals)}")

t2, p2 = ttest_ind(ctrl_vals, vip_ex_vals)
print(f"ctrl vs vip_ex: t = {t2:.3f}, p = {p2:.4f}, n = {len(ctrl_vals)} vs {len(vip_ex_vals)}")
