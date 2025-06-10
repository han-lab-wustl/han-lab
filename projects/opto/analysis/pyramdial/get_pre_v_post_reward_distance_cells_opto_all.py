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
# import warnings
# warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
results_all=[]
radian_alignment = {}
cm_window = 20
# %%

#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = int(conddf.days.values[ii])
    animal = conddf.animals.values[ii]
    # skip e217 day
    if ii!=179:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2  
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        radian_alignment, results_pre, results_post, results_pre_early, results_post_early = get_rew_cells_opto(
            params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
            radian_alignment, cm_window=cm_window)
        results_all.append([results_pre, results_post, results_pre_early, results_post_early])

pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp) 

# %%

#%%
# separate out variables
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
pre_late = [xx[0] for xx in results_all]
post_late = [xx[1] for xx in results_all]
pre_early = [xx[2] for xx in results_all]
post_early = [xx[3] for xx in results_all]

# concat all cell type goal cell prop
all_cells = [pre_late, post_late, pre_early, post_early]
goal_cell_prop = np.concatenate([[xx['goal_cell_prop'] for xx in cll] for cll in all_cells])
realdf= pd.DataFrame()
realdf['goal_cell_prop']=goal_cell_prop
lbl = ['pre_late', 'post_late', 'pre_early', 'post_early']
realdf['cell_type']=np.concatenate([[lbl[kk]]*len(cll) for kk,cll in enumerate(all_cells)])
realdf['animal']=np.concatenate([df.animals]*len(all_cells))
realdf['optoep']=np.concatenate([df.optoep]*len(all_cells))
realdf['opto']=[True if xx>1 else False if xx<1 else np.nan for xx in realdf['optoep']]
realdf['condition']=np.concatenate([df.in_type]*len(all_cells))
realdf['condition']=[xx if 'vip' in xx else 'ctrl' for xx in realdf.condition.values]
realdf['day']=np.concatenate([df.days]*len(all_cells))

# realdf=realdf.drop([2,3,5,7,17,18,24,35,37]) # z14, z15, z17 excluded days
realdf=realdf[realdf['goal_cell_prop']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e200')&(realdf.animal!='e190')]
# remove outlier days
realdf=realdf.drop([715,705,737,526,516,548,416,605])
dfagg = realdf.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
fig,axes=plt.subplots(ncols=4,figsize=(16,5),sharey=True,sharex=True,)
for cl,cll in enumerate(dfagg.cell_type.unique()):
    ax=axes[cl]
    sns.barplot(x='condition',y='goal_cell_prop',hue='opto',data=dfagg[dfagg.cell_type==cll],fill=False,ax=ax)
    ax.set_title(cll)
#%%
# Pivot to get a DataFrame with separate columns for opto==False and opto==True
plt.rc('font', size=20)          # controls default text sizes

pivoted = dfagg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop'
).reset_index()

# Rename the columns for clarity
pivoted.columns.name = None  # remove multiindex name
pivoted = pivoted.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})

# Calculate difference
pivoted['difference'] = pivoted['goal_cell_prop_on'] - pivoted['goal_cell_prop_off']
pivoted['difference'] =pivoted['difference']*100
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
a=0.7;s=12
fig, axes = plt.subplots(ncols=4, figsize=(12,5), sharey=True,sharex=True)
cllty = ['Pre-reward, early', 'Post-reward, early', 'Pre-reward, late', 'Post-reward, late']
cellty = ['pre_early', 'post_early', 'pre_late', 'post_late']
for cl, cll in enumerate(cellty):
    ax = axes[cl]
    sns.barplot(
        x='condition', y='difference', hue='condition',data=pivoted[pivoted['cell_type'] == cll],
        ax=ax, palette=pl, errorbar='se',fill=False,
    )
    sns.stripplot(
        x='condition', y='difference', hue='condition',data=pivoted[pivoted['cell_type'] == cll],
        ax=ax, palette=pl, alpha=a, s=s
    )
    ax.set_title(cllty[cl])
    ax.set_xlabel('')
    ax.set_ylabel('$\\Delta$ % Reward cell \n(LEDon-LEDoff)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
        # --- Stats + annotation ---
    data = pivoted[pivoted['cell_type'] == cll]
    conds = data['condition'].unique()
    pairs = list(combinations(conds, 2))[:2]
    y_max = data['difference'].quantile(.85)
    y_step = 0.5 * abs(y_max)

    for i, (cond1, cond2) in enumerate(pairs):
        vals1 = data[data['condition'] == cond1]['difference']
        vals2 = data[data['condition'] == cond2]['difference']
        stat, pval = scipy.stats.ttest_ind(vals1, vals2)

        # Annotation text
        if pval < 0.001:
            text = '***'
        elif pval < 0.01:
            text = '**'
        elif pval < 0.05:
            text = '*'
        else:
            text = f""

        # Get x-locations
        x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
        y = y_max + y_step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
        ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
        ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(savedst, 'early_v_late_cell_type_reward_cellp_opto.svg'), bbox_inches='tight')
#%%
# Map old cell types to new ones
cell_type_map = {
    'pre_late': 'late',
    'pre_early': 'early',
    'post_late': 'late',
    'post_early': 'early'
}
# Copy and remap
realdf_avg = realdf.copy()
realdf_avg['cell_type'] = realdf_avg['cell_type'].map(cell_type_map)

# Average across pre_early/late and post_early/late per animal/condition/opto
dfagg_avg = realdf_avg.groupby(['animal', 'opto', 'cell_type', 'condition']).median(numeric_only=True).reset_index()
pivoted_avg = dfagg_avg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop'
).reset_index()

pivoted_avg.columns.name = None
pivoted_avg = pivoted_avg.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})
pivoted_avg['difference'] = pivoted_avg['goal_cell_prop_on'] - pivoted_avg['goal_cell_prop_off']
pivoted_avg['difference']=pivoted_avg['difference']*100
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
a = 0.7
s = 12
lbls = ['Early', 'Late']
fig, axes = plt.subplots(ncols=2, figsize=(7.5,6), sharey=True)
for cl, cll in enumerate(pivoted_avg['cell_type'].unique()):
    ax = axes[cl]
    sns.barplot(
        x='condition', y='difference', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, errorbar='se', fill=False
    )
    sns.stripplot(
        x='condition', y='difference', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, alpha=a, s=s
    )
    ax.set_title(lbls[cl])
    ax.set_ylabel('$\\Delta$ % Reward cell\n(LEDon - LEDoff)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
    ax.set_xlabel('')
    # --- Stats + annotation ---
    data = pivoted_avg[pivoted_avg['cell_type'] == cll]

    conds = data['condition'].unique()
    pairs = list(combinations(conds, 2))[:2]
    y_max = data['difference'].quantile(.85)
    y_step = 0.5 * abs(y_max)

    for i, (cond1, cond2) in enumerate(pairs):
        vals1 = data[data['condition'] == cond1]['difference']
        vals2 = data[data['condition'] == cond2]['difference']
        stat, pval = scipy.stats.ttest_ind(vals1, vals2)
        # Annotation text
        if pval < 0.001:
            text = '***'
        elif pval < 0.01:
            text = '**'
        elif pval < 0.05:
            text = '*'
        else:
            text = f""

        # Get x-locations
        x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
        y = y_max + y_step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
        ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
        ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'early_v_late_reward_cellp_opto.svg'), bbox_inches='tight')

# %%

#%%
cll='early'
x1=pivoted_avg.loc[(pivoted_avg['cell_type'] == cll) & (pivoted_avg['condition']=='ctrl'), 'difference'].values
x2=pivoted_avg.loc[(pivoted_avg['cell_type'] == cll) & (pivoted_avg['condition']=='vip_ex'),'difference'].values
_,pval = scipy.stats.ttest_ind(x1,x2)
print(pval)
# %%
