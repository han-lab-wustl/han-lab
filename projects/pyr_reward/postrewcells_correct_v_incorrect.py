
#%%
"""
zahra
july 2024
quantify reward-relative cells near reward
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,reward_act_nearrew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'near_rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
num_epochs = [] 
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
lasttr=8 #  last trials
bins=90
saveto = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
# iterate through all animals
dfs = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df=reward_act_nearrew(ii,params_pth,\
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
                total_cells)
        dfs.append(df)
pdf.close()

#%%
plt.rc('font', size=16)          # controls default text sizes
bigdf=pd.concat(dfs)

# average
bigdf=bigdf.groupby(['animal', 'epoch', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()

# plot
s=10
fig,ax = plt.subplots(figsize=(5,5))
sns.stripplot(x='trial_type', y='mean_tc', hue='epoch', data=bigdf,dodge=True,
            s=s,alpha=0.5)
sns.barplot(x='trial_type', y='mean_tc', hue='epoch', data=bigdf, fill=False)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Average $\Delta F/F$ of tuning curve')
ax.set_xlabel('')


# average
bigdf=bigdf.groupby(['animal', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()

fig,ax = plt.subplots(figsize=(2,5))
sns.stripplot(x='trial_type', y='mean_tc', data=bigdf,dodge=True,color='k',
            s=s,alpha=0.7)
sns.barplot(x='trial_type', y='mean_tc', data=bigdf, fill=False,color='k')

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\int$ tuning curve ($\Delta F/F$)')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_tc']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_tc']
t,pval = scipy.stats.ttest_rel(cor,incor)
# statistical annotation       
ii=0.5
y=10
pshift=.01
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Post-reward cells',pad=40)