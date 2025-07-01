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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto_dff
import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_dff_rewcells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto_dff.p"
# with open(saveddataset, "rb") as fp: #unpickle
#     radian_alignment_saved = pickle.load(fp)
# # initialize var
radian_alignment_saved = {} # overwrite
results_all=[]
radian_alignment = {}
cm_window = 20

#%%
# iterate through all animals 
for ii in range(len(conddf)):
    day = int(conddf.days.values[ii])
    animal = conddf.animals.values[ii]
    # skip e217 day
    if ii!=202:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2  
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        radian_alignment, results_pre, results_post, results_pre_early, results_post_early = get_rew_cells_opto_dff(
            params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
            radian_alignment, cm_window=cm_window)
        results_all.append([results_pre, results_post, results_pre_early, results_post_early])

pdf.close()
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 

#%%
# top down approach
# 1) com dist in opto vs. control
# 3) place v. reward
# tcs_correct, coms_correct, tcs_fail, coms_fail,
# tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early
# 1) get coms correct
df = conddf.copy()
df = df.drop([202]) # skipped e217 day
# Filter out unwanted
keep = ~((df.animals == 'z14') & (df.days < 15))
keep &= ~((df.animals == 'z15') & (df.days < 8))
keep &= ~((df.animals == 'e217') &((df.days < 9) | (df.days.isin([26,29,30]))))
keep &= ~((df.animals == 'e216') & (df.days < 32))
keep &= ~((df.animals=='e200')&((df.days.isin([67]))))
keep &= ~((df.animals=='e218')&(df.days>44))

df = df[keep].reset_index(drop=True)
mask = keep.values
keys = list(radian_alignment.keys())
radian_alignment_newcoms= {k: radian_alignment[k] for k, m in zip(keys, mask) if m}
coms_correct = [xx[1] for k,xx in radian_alignment_newcoms.items()]
tcs_correct = [xx[0] for k,xx in radian_alignment_newcoms.items()]
optoep = [xx if xx>1 else 2 for xx in df.optoep.values]
# opto comparison
coms_correct = [xx[[optoep[ep]-2,optoep[ep]-1],:] if len(xx)>optoep[ep]-1 else xx[[optoep[ep]-3,optoep[ep]-2],:] for ep,xx in enumerate(coms_correct)]
# tcs_correct = [xx[[optoep[ep]-2,optoep[ep]-1],:] for ep,xx in enumerate(tcs_correct)]
coms_correct_prev = [xx[0] for ep,xx in enumerate(coms_correct)]
coms_correct_opto = [xx[1] for ep,xx in enumerate(coms_correct)]

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
    ax.hist(data_prev,alpha=a,label='prev_ep',bins=100,density=True)
    ax.hist(data_opto,alpha=a,label='opto_ep',bins=100,density=True)
    # KDE plots
    sns.kdeplot(data_prev, ax=ax, label='prev_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    sns.kdeplot(data_opto, ax=ax, label='opto_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    ax.set_title(lbls[pl])
    ax.set_xlim([-np.pi/4,np.pi])
    ax.axvline(0, color='gray', linewidth=2,linestyle='--')
ax.legend()
#%%
# First compute differences for each group
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Combine per-group previous and opto data
labels = ['ctrl', 'vip_in', 'vip_ex']

# Plotting
a = 0.4
fig, axs = plt.subplots(ncols=3, figsize=(17, 10), sharey=True)
axs = axs.flatten()

# Define range for density evaluation
xs = np.linspace(-np.pi, np.pi, 500)

for i, (prev, opto, label) in enumerate(zip(plots[0], plots[1], labels)):
    ax = axs[i]
    prev_vals = np.concatenate(prev) - np.pi
    opto_vals = np.concatenate(opto) - np.pi
    prev_vals = prev_vals[np.isfinite(prev_vals)]
    opto_vals = opto_vals[np.isfinite(opto_vals)]
    # Estimate densities
    prev_kde = scipy.stats.kde.gaussian_kde(prev_vals)
    opto_kde = scipy.stats.kde.gaussian_kde(opto_vals)
    prev_density = prev_kde(xs)
    opto_density = opto_kde(xs)
    diff_density = opto_density - prev_density

    # Plot the density difference
    ax.plot(xs, diff_density, label='opto - prev')
    ax.axhline(0, color='grey', linewidth=2, linestyle='--')
    ax.fill_between(xs, 0, diff_density, color='grey', alpha=0.4)
    ax.set_title(f'{label} (opto - prev)')
    # ax.set_xlim([-np.pi/4, np.pi])

ax.legend()
plt.tight_layout()
plt.show()


#%%
# quantify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Set up
windows = np.arange(0,np.pi,.1)
conds = ['ctrl', 'vip_in', 'vip_ex']
groupings = [
    (ctrl_com_prev, ctrl_com_opto),
    (vip_in_com_prev, vip_in_com_opto),
    (vip_ex_com_prev, vip_ex_com_opto)
]
# Assume df.animals has same order as groupings
animal_ids = {
    'ctrl': df[(~df.in_type.str.contains('vip')) & (df.optoep > 1)].animals.values,
    'vip_in': df[(df.in_type == 'vip') & (df.optoep > 1)].animals.values,
    'vip_ex': df[(df.in_type == 'vip_ex') & (df.optoep > 1)].animals.values,
}

# Collect trial data with animal ID
rows = []
for w in windows:
    for cond, (prev_group, opto_group) in zip(conds, groupings):
        for p, o, animal in zip(prev_group, opto_group, animal_ids[cond]):
            prev = np.array(p) - np.pi
            opto = np.array(o) - np.pi
            frac_prev = np.mean(np.abs(prev) < w)
            frac_opto = np.mean(np.abs(opto) < w)
            diff = frac_opto - frac_prev
            rows.append({'window': w, 'cond': cond, 'diff': diff, 'animal': animal})

df_density = pd.DataFrame(rows)

# Aggregate: per animal average for each window and condition
df_animal_avg = df_density.groupby(['window', 'cond', 'animal'])['diff'].mean().reset_index()

# Seaborn plot
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_animal_avg, x='window', y='diff', hue='cond', marker='o', err_style='bars', errorbar='se')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('COM window width (radians)')
plt.ylabel('Δ density near 0 (LEDon - LEDoff)')
plt.title('Change in reward-centered COM density (per animal avg)')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

# Wilcoxon test per group at window=0.2
for wtest in windows:
    print(f"\nWilcoxon test at ±{wtest} rad (per-animal averages):")
    vals1 = df_animal_avg[(df_animal_avg['window'] == wtest) & (df_animal_avg['cond'] == 'vip_in')]['diff']
    vals2 = df_animal_avg[(df_animal_avg['window'] == wtest) & (df_animal_avg['cond'] == 'ctrl')]['diff']
    stat, pval = scipy.stats.ttest_ind(vals1,vals2)
    print(f"{cond}: p = {pval:.3g}, n = {len(vals1)}")

# %%
# rew cell %
# separate out variables
df = conddf.copy()
df = df.drop([202]) # skipped e217 day
# df=df.iloc[:120]
pre_late = [xx[0] for xx in results_all]
post_late = [xx[1] for xx in results_all]
pre_early = [xx[2] for xx in results_all]
post_early = [xx[3] for xx in results_all]
plt.rc('font', size=20)
# concat all cell type goal cell prop
all_cells = [pre_late, post_late, pre_early, post_early]
goal_cell_prop = np.concatenate([[xx['goal_cell_prop'] for xx in cll] for cll in all_cells])

realdf= pd.DataFrame()
realdf['goal_cell_prop']=goal_cell_prop
lbl = ['pre_late', 'post_late', 'pre_early', 'post_early']
realdf['cell_type']=np.concatenate([[lbl[kk]]*len(cll) for kk,cll in enumerate(all_cells)])
realdf['animal']=np.concatenate([df.animals]*len(all_cells))
realdf['optoep']=np.concatenate([df.optoep]*len(all_cells))
realdf['opto']=[True if xx>1 else False for xx in realdf['optoep']]
realdf['condition']=np.concatenate([df.in_type]*len(all_cells))
realdf['condition']=[xx if 'vip' in xx else 'ctrl' for xx in realdf.condition.values]
realdf['day']=np.concatenate([df.days]*len(all_cells))
# realdf['goal_cell_prop'] = realdf['goal_cell_prop'] - realdf['goal_cell_prop_shuf']
realdf=realdf[realdf['goal_cell_prop']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e190') & (realdf.animal!='e200')]
# remove outlier days
# realdf=realdf[~((realdf.animal=='e201')&((realdf.day>62)))]
realdf=realdf[~((realdf.animal=='z14')&((realdf.day<33)))]
# realdf=realdf[~((realdf.animal=='z16')&((realdf.day>15)))]
realdf=realdf[~((realdf.animal=='z17')&((realdf.day<1)|(realdf.day.isin([3,4,5,9,13,16]))))]

realdf=realdf[~((realdf.animal=='z15')&((realdf.day<8)|(realdf.day.isin([15]))))]
realdf=realdf[~((realdf.animal=='e217')&((realdf.day<9)|(realdf.day.isin([21,29,30,26]))))]
realdf=realdf[~((realdf.animal=='e216')&((realdf.day<32)|(realdf.day.isin([57]))))]
# realdf=realdf[~((realdf.animal=='e200')&((realdf.day.isin([67,68,81]))))]
# realdf=realdf[~((realdf.animal=='e218')&(realdf.day.isin([41,55])))]
# realdf=realdf[~((realdf.animal=='e186')&(realdf.day.isin([34,37,40])))]
# realdf=realdf[(realdf.optoep==0)|(realdf.optoep==1)|(realdf.optoep>1)]
#%%
pl = {False: "slategray", True: 'darkorange'}
dfagg = realdf.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
cllty = ['Pre-reward, early', 'Post-reward, early', 'Pre-reward, late', 'Post-reward, late']
a=0.7;s=12
fig,axes=plt.subplots(ncols=4,figsize=(16,5),sharey=True,sharex=True,)
for cl,cll in enumerate(dfagg.cell_type.unique()):
    ax=axes[cl]
    sns.barplot(x='condition',y='goal_cell_prop',hue='opto',data=dfagg[dfagg.cell_type==cll],fill=False,ax=ax,palette=pl,legend=False)
    sns.stripplot(x='condition',y='goal_cell_prop',hue='opto',data=dfagg[dfagg.cell_type==cll],s=10,alpha=a,ax=ax,palette=pl,legend=False,dodge=True)
    ax.set_title(cllty[cl])
    ax.set_xlabel('')
    ax.set_ylabel('Reward cell %')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
plt.savefig(os.path.join(savedst, 'ledoff_v_ledon_reward_cellp_opto.svg'), bbox_inches='tight')
#%%
# Pivot to get a DataFrame with separate columns for opto==False and opto==True
plt.rc('font', size=20)          # controls default text sizes
pivoted = dfagg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop',
    fill_value=0
).reset_index()
# all sessions
# pivoted = dfagg.reset_index().drop(columns=['index']).pivot(
#     index=['animal', 'cell_type','day', 'condition'],  # keep all rows distinct
#     columns='opto',
#     values='goal_cell_prop'
# ).reset_index()

pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
# Rename the columns for clarity
pivoted.columns.name = None  # remove multiindex name
pivoted = pivoted.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})

# Calculate difference
pivoted['difference'] = pivoted['goal_cell_prop_on']-pivoted['goal_cell_prop_off']
pivoted['difference'] =pivoted['difference']*100
fig, axes = plt.subplots(ncols=4, figsize=(12,5), sharey=True,sharex=True)
cllty = ['Pre-reward, early', 'Post-reward, early', 'Pre-reward, late', 'Post-reward, late']
cellty = ['pre_early', 'post_early', 'pre_late', 'post_late']
data = pivoted[pivoted['cell_type'] == 'post_early']
y_max = data['difference'].quantile(.85)
y_step = 0.4*abs(y_max)

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

    for i, (cond1, cond2) in enumerate(pairs):
        vals1 = data[data['condition'] == cond1]['difference']
        vals2 = data[data['condition'] == cond2]['difference']
        stat, pval = scipy.stats.ranksums(vals1, vals2)

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
dfagg_avg = realdf_avg.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
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
# pivoted_avg=pivoted_avg[pivoted_avg.animal!='e201']
lbls = ['Early', 'Late']
fig, axes = plt.subplots(ncols=2, figsize=(7.5,6), sharey=True)
for cl, cll in enumerate(pivoted_avg['cell_type'].unique()):
    ax = axes[cl]
    sns.barplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, errorbar='se', fill=False
    )
    sns.stripplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, alpha=a, s=s
    )
    ax.set_title(lbls[cl])
    ax.set_ylabel('$\\Delta$ Reward cell %\n(LEDon-LEDoff)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
    ax.set_xlabel('')
    # --- Stats + annotation ---
    data = pivoted_avg[pivoted_avg['cell_type'] == cll]

    conds = data['condition'].unique()
    pairs = list(combinations(conds, 2))[:2]
    if cl==0:
        y_max = data['difference'].quantile(.85)
        y_step = 0.4 * abs(y_max)

    for i, (cond1, cond2) in enumerate(pairs):
        vals1 = data[data['condition'] == cond1]['difference']
        vals2 = data[data['condition'] == cond2]['difference']
        stat, pval = scipy.stats.ranksums(vals1, vals2)
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
#%%
# correlate with behavior

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
# realdf['goal_cell_prop'] = realdf['goal_cell_prop'] - realdf['goal_cell_prop_shuf']
realdf=realdf[realdf['goal_cell_prop']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e190')]
# remove outlier days
realdf=realdf[~((realdf.animal=='z15')&(realdf.day<8))]
realdf=realdf[~((realdf.animal=='e217')&((realdf.day<9)|(realdf.day.isin([21,26]))))]
realdf=realdf[~((realdf.animal=='e216')&((realdf.day<32)|(realdf.day.isin([57]))))]
realdf=realdf[~((realdf.animal=='e200')&((realdf.day.isin([67,68,81]))))]
realdf=realdf[~((realdf.animal=='e218')&(realdf.day==55))]
cell_type_map = {
    'pre_late': 'all',
    'pre_early': 'all',
    'post_late': 'all',
    'post_early': 'all'
}
# Copy and remap
realdf_avg = realdf.copy()
realdf_avg['cell_type'] = realdf_avg['cell_type'].map(cell_type_map)
# Average across pre_early/late and post_early/late per animal/condition/opto
dfagg_avg = realdf_avg.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
pivoted_avg = dfagg_avg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop'
).reset_index()
pivoted_avg.columns.name = None
pivoted_avg = pivoted_avg.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})
pivoted_avg['difference'] = pivoted_avg['goal_cell_prop_on']#-pivoted_avg['goal_cell_prop_off']
pivoted_avg['difference']=pivoted_avg['difference']*100
# pivoted_avg=pivoted_avg[pivoted_avg.cell_type=='early']
beh = pd.read_csv(r'Z:\condition_df\vip_opto_behavior.csv')
beh=beh[(beh.animals.isin(realdf.animal.values))&(beh.days.isin(realdf.day.values))]
beh = beh.groupby(['animals', 'opto']).mean(numeric_only=True).reset_index()
beh=beh[beh.opto==True]
# Perform regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(beh.rates_diff.values, pivoted_avg.difference.values)
print(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")
# Plot scatter plot with regression line
fig,ax=plt.subplots(figsize=(6,5))
sns.scatterplot(x=beh.rates_diff.values, y=pivoted_avg.difference.values,hue=pivoted_avg.condition.values,s=300,alpha=.7,palette=pl,ax=ax)
ax.plot(beh.rates_diff.values, intercept + slope * beh.rates_diff.values, color='steelblue', label='Regression Line',linewidth=3)
ax.legend(['Regression Line', 'Control', 'VIP Inhibition', 'VIP Excitation'], fontsize='small')
ax.set_xlabel("% Correct trials (LEDon-LEDoff)")
ax.set_ylabel("Reward cell %")
ax.set_title(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")
ax.spines[['top', 'right']].set_visible(False)
plt.savefig(os.path.join(savedst, 'rewardcell_v_performance.svg'), bbox_inches='tight')
#%%
# Map old cell types to new ones
cell_type_map = {
    'pre_late': 'pre',
    'pre_early': 'pre',
    'post_late': 'post',
    'post_early': 'post'
}
# Copy and remap
realdf_avg = realdf.copy()
realdf_avg['cell_type'] = realdf_avg['cell_type'].map(cell_type_map)
# Average across pre_early/late and post_early/late per animal/condition/opto
dfagg_avg = realdf_avg.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
pivoted_avg = dfagg_avg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop',
    fill_value=0  # or np.nan_to_num afterwards
).reset_index()

pivoted_avg.columns.name = None
pivoted_avg = pivoted_avg.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})
pivoted_avg['difference'] = pivoted_avg['goal_cell_prop_on'] - pivoted_avg['goal_cell_prop_off']
pivoted_avg['difference']=pivoted_avg['difference']*100
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
a = 0.7
s = 12
lbls = ['Post', 'Pre']
fig, axes = plt.subplots(ncols=2, figsize=(7.5,6), sharey=True)
for cl, cll in enumerate(pivoted_avg['cell_type'].unique()):
    ax = axes[cl]
    sns.barplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, errorbar='se', fill=False
    )
    sns.stripplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, alpha=a, s=s
    )
    ax.set_title(lbls[cl])
    ax.set_ylabel('$\\Delta$ % Reward cell\n(LEDon-LEDoff)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
    ax.set_xlabel('')
    # --- Stats + annotation ---
    data = pivoted_avg[pivoted_avg['cell_type'] == cll]

    conds = data['condition'].unique()
    pairs = list(combinations(conds, 2))[:2]
    if cl==0:
        y_max = data['difference'].quantile(.85)
        y_step = 0.4 * abs(y_max)

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
fig.suptitle('All trials')
plt.savefig(os.path.join(savedst, 'pre_v_post_reward_cellp_opto.svg'), bbox_inches='tight')

# %%
pre = [xx[0]['goal_id'] for xx in results_all]
post = [xx[1]['goal_id'] for xx in results_all]
pre_early= [xx[2]['goal_id'] for xx in results_all]
post_early = [xx[3]['goal_id'] for xx in results_all]

sv = r'Z:\condition_df\goal_cell_id_dff_tc_opto.p'
with open(sv, "wb") as fp:   #Pickling
    pickle.dump(results_all, fp) 
