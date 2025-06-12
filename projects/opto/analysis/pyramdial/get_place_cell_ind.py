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
from projects.pyr_reward.placecell import make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper\vip_r21'
savepth = os.path.join(savedst, 'vip_opto_place.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\place_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        datadct = pickle.load(fp)
# initialize var
# datadct = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
place_cell_null=[]
place_window = 20
num_iterations=1000
bin_size=3 # cm
lasttr=8 # last trials
bins=90

# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if ii!=179:
      if animal=='e145': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
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
      if animal=='e145':
         ybinned=ybinned[:-1]
         forwardvel=forwardvel[:-1]
         changeRewLoc=changeRewLoc[:-1]
         trialnum=trialnum[:-1]
         rewards=rewards[:-1]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      # only test opto vs. ctrl
      eptest = conddf.optoep.values[ii]
      if conddf.optoep.values[ii]<2: 
         eptest = random.randint(2,3)   
         if len(eps)<4: eptest = 2 # if no 3 epochs    

      #   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      #   Fc3 = fall_fc3['Fc3']
      #   dFF = fall_fc3['dFF']
      #   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      #   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      #   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      #   #if pc in all but 1
      #   pc_bool = np.sum(pcs,axis=0)>=len(eps)-2
      #   # looser restrictions
      #   pc_bool = np.sum(pcs,axis=0)>=1
      #   Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
      #   # if no cells pass these crit
      #   if Fc3.shape[1]==0:
      #       Fc3 = fall_fc3['Fc3']
      #       Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      #       # to avoid issues with e217 and z17?
      #       # pc_bool = np.sum(pcs,axis=0)>=1
      #       Fc3 = Fc3[:,((skew>1)&pc_bool)]
      #   if Fc3.shape[1]>0:
            # get abs dist tuning 
            # tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
            #         Fc3,trialnum,rewards,forwardvel,
            #         rewsize,bin_size) # last 5 trials
      if sum([f'{animal}_{day:03d}' in xx for xx in list(datadct.keys())])>0:
            k = [k for k,xx in datadct.items() if f'{animal}_{day:03d}' in k][0]
            tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs=datadct[k]
      # get cells that maintain their coms b/wn previous and opto ep
      perm = [(eptest-2, eptest-1)]   
      if perm[0][1]<len(coms_correct_abs): # make sure tested epoch has enough trials
            com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
            compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
            # get cells across OPTO VS. CONTROL EPOCHS
            pcs = np.unique(np.concatenate(compc))
            pcs_all = pcs
            datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs,pcs]
pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(datadct, fp) 

#%%
# top down approach
# 1) com dist in opto vs. control
# 3) place v. reward
# tcs_correct, coms_correct, tcs_fail, coms_fail,pcs
# 1) get coms correct
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
coms_correct = [xx[1] for k,xx in datadct.items()]
pcs = [xx[4] for k,xx in datadct.items()]
optoep = [xx if xx>1 else 2 for xx in df.optoep.values]
# opto comparison
coms_correct = [xx[[optoep[ep]-2,optoep[ep]-1]][:,pcs[ep]] if len(xx)>optoep[ep]-1 else xx[[optoep[ep]-3,optoep[ep]-2]][:,pcs[ep]] for ep,xx in enumerate(coms_correct)]
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
    data_prev = np.concatenate(plots[0][pl]) - np.pi
    data_opto = np.concatenate(plots[1][pl]) - np.pi
    ax.hist(data_prev,alpha=a,label='prev_ep',density=True)
    ax.hist(data_opto,alpha=a,label='opto_ep',density=True)
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
pcs = [xx[4] for k, xx in datadct.items()]  # place cell indices per animal
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
mean_prev = [np.nanmean(x, axis=1) for x in tcs_placecells_prev]
mean_opto = [np.nanmean(x, axis=1) for x in tcs_placecells_opto]

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
plt.title('VIP Modulation of Place Cell Activity on Correct Trials')
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
all_data=all_data[~((all_data.animal=='z15')&(all_data.day<8))]
all_data=all_data[~((all_data.animal=='e217')&(all_data.day<9))]
all_data=all_data[~((all_data.animal=='e216')&(all_data.day<32))]
all_data=all_data[~((all_data.animal=='e218')&(all_data.day<44))]

# Group by animal for average per-animal effect
per_animal = all_data.groupby(['animal', 'group'])['change_in_activity'].mean().reset_index()
plt.figure(figsize=(3,5))
sns.boxplot(data=per_animal, x='group', y='change_in_activity', palette='Set2')
sns.stripplot(data=per_animal, x='group', y='change_in_activity', color='black', alpha=0.7)
# plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Per-Animal Mean Δ Place Cell Activity')
plt.title('VIP Modulation of Place Cell Activity (Averaged per Animal)')
plt.tight_layout()
from scipy.stats import ranksums
import itertools
import pandas as pd

# Get unique groups
groups = per_animal['group'].unique()
from scipy.stats import kruskal

# Extract values by group
ctrl_vals = per_animal[per_animal['group'] == 'ctrl']['change_in_activity']
vip_in_vals = per_animal[per_animal['group'] == 'vip_in']['change_in_activity']
vip_ex_vals = per_animal[per_animal['group'] == 'vip_ex']['change_in_activity']
# Kruskal-Wallis H-test
stat, pval = kruskal(ctrl_vals, vip_in_vals, vip_ex_vals)
print(f"Kruskal–Wallis test statistic = {stat:.3f}, p = {pval:.4f}")

# Initialize results
pairwise_results = []
# Pairwise rank-sum tests
for g1, g2 in itertools.combinations(groups, 2):
    vals1 = per_animal[per_animal['group'] == g1]['change_in_activity']
    vals2 = per_animal[per_animal['group'] == g2]['change_in_activity']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    pairwise_results.append({
        'Group 1': g1,
        'Group 2': g2,
        'Statistic': stat,
        'p-value': pval
    })
# Create results DataFrame
rank_df = pd.DataFrame(pairwise_results)

# Apply Bonferroni correction
rank_df['p_bonf'] = rank_df['p-value'] * len(rank_df)
rank_df['p_bonf'] = rank_df['p_bonf'].clip(upper=1.0)

print("\nPairwise Rank-Sum Tests (with Bonferroni correction):")
print(rank_df[['Group 1', 'Group 2', 'Statistic', 'p-value', 'p_bonf']])
