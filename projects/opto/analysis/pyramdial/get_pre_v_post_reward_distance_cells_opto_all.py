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
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
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
        radian_alignment, results_pre, results_post, results_pre_early, results_post_early = get_rew_cells_opto(
            params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
            radian_alignment, cm_window=cm_window)
        results_all.append([results_pre, results_post, results_pre_early, results_post_early])

pdf.close()
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 

#%%
a=0.5
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
realdf['goal_cell_prop']=realdf['goal_cell_prop']*100
lbl = ['pre_late', 'post_late', 'pre_early', 'post_early']
realdf['cell_type']=np.concatenate([[lbl[kk]]*len(cll) for kk,cll in enumerate(all_cells)])
realdf['epoch_dur'] = ['early' if 'early' in xx else 'late' for xx in realdf.cell_type]
realdf['animal']=np.concatenate([df.animals]*len(all_cells))
realdf['optoep']=np.concatenate([df.optoep]*len(all_cells))
realdf['opto']=[True if xx>1 else False for xx in realdf['optoep']]
realdf['condition']=np.concatenate([df.in_type]*len(all_cells))
realdf['condition']=[xx if 'vip' in xx else 'ctrl' for xx in realdf.condition.values]
realdf['day']=np.concatenate([df.days]*len(all_cells))
# realdf['goal_cell_prop'] = realdf['goal_cell_prop'] - realdf['goal_cell_prop_shuf']
realdf=realdf[realdf['goal_cell_prop']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e190')]
# remove outlier days
# realdf=realdf[~((realdf.animal=='e201')&((realdf.day>62)))]
realdf=realdf[~((realdf.animal=='z14')&((realdf.day<33)))]
# realdf=realdf[~((realdf.animal=='z16')&((realdf.day>13)))]
realdf=realdf[~((realdf.animal=='z15')&((realdf.day<8)|(realdf.day.isin([15]))))]
realdf=realdf[~((realdf.animal=='e217')&((realdf.day<9)|(realdf.day.isin([21,29,30,26]))))]
realdf=realdf[~((realdf.animal=='e216')&((realdf.day<32)|(realdf.day.isin([47,55,57]))))]
# realdf=realdf[~((realdf.animal=='e200')&((realdf.day.isin([67,68,81]))))]
realdf=realdf[~((realdf.animal=='e218')&(realdf.day.isin([41,55])))]
# realdf=realdf[~((realdf.animal=='e186')&(realdf.day.isin([34,37,40])))]
# realdf=realdf[(realdf.optoep==0)|(realdf.optoep==1)|(realdf.optoep>1)]
dfagg=realdf.groupby(['animal', 'cell_type', 'condition','opto']).mean(numeric_only=True).reset_index()
s=12
# Pivot to get a DataFrame with separate columns for opto==False and opto==True
plt.rc('font', size=20)          # controls default text sizes
pivoted = dfagg.pivot_table(
    index=['animal', 'cell_type', 'condition'],
    columns='opto',
    values='goal_cell_prop',
    fill_value=0
).reset_index()
# compare between groups
dfbig=realdf.groupby(['animal', 'epoch_dur','condition','opto']).mean(numeric_only=True).reset_index()
dfbig['animals']=dfbig['animal']
dfagg=dfbig[dfbig.epoch_dur=='early']

# number of epochs vs. reward cell prop    
fig,axes = plt.subplots(ncols=2,figsize=(8,3),sharex=True,sharey=True)
ax=axes[0]
# av across mice
pl =['k','slategray']
df_plt =dfagg
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)
# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    if wide.shape[1]==2:
        t,p = scipy.stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None
# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['goal_cell_prop'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Reward cell %')
ax.set_title('First 8 trials')

ax=axes[1]
### late
dfagg=dfbig[dfbig.epoch_dur=='late']
df_plt = dfagg
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)
# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    if wide.shape[1]==2:
        t,p = scipy.stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None
# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['goal_cell_prop'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_title('Last 8 trials')

plt.savefig(os.path.join(savedst, 'goal_cell_prop_ctrlvopto_early_late.svg'),bbox_inches='tight')
#%%
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
# Rename the columns for clarity
pivoted.columns.name = None  # remove multiindex name
pivoted = pivoted.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})

# Calculate difference
pivoted['difference'] = pivoted['goal_cell_prop_on']-pivoted['goal_cell_prop_off']
pivoted['difference'] =pivoted['difference']*100
fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(7,8), sharey=True,sharex=True)
cllty = ['Pre-reward, early', 'Post-reward, early', 'Pre-reward, late', 'Post-reward, late']
cellty = ['pre_early', 'post_early', 'pre_late', 'post_late']
data = pivoted[pivoted['cell_type'] == 'post_early']
y_max = data['difference'].quantile(.85)
y_step = 0.4*abs(y_max)
axes=axes.flatten()
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
# average all trials

# Map old cell types to new ones
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
pivoted_avg['difference'] = pivoted_avg['goal_cell_prop_on'] - pivoted_avg['goal_cell_prop_off']
pivoted_avg['difference']=pivoted_avg['difference']*100
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
a = 0.7
s = 12
# pivoted_avg=pivoted_avg[pivoted_avg.animal!='e201']
lbls = ['All trials\nReward']
fig, ax = plt.subplots(figsize=(4.5,6), sharey=True)
for cl, cll in enumerate(pivoted_avg['cell_type'].unique()):
    sns.barplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, errorbar='se', fill=False
    )
    sns.stripplot(
        x='condition', y='difference',hue='condition', data=pivoted_avg[pivoted_avg['cell_type'] == cll],
        ax=ax, palette=pl, alpha=a, s=s
    )
    ax.set_title(lbls[cl])
    ax.set_ylabel('$\\Delta$ Reward cell % (LEDon-LEDoff)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
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
plt.savefig(os.path.join(savedst, 'all_reward_cellp_opto.svg'), bbox_inches='tight')
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
realdf=realdf[~((realdf.animal=='z14')&((realdf.day<33)))]
# realdf=realdf[~((realdf.animal=='z16')&((realdf.day>13)))]
realdf=realdf[~((realdf.animal=='z15')&((realdf.day<8)|(realdf.day.isin([15]))))]
realdf=realdf[~((realdf.animal=='e217')&((realdf.day<9)|(realdf.day.isin([21,29,30,26]))))]
# realdf=realdf[~((realdf.animal=='e216')&((realdf.day<32)|(realdf.day.isin([47,55,57]))))]
# realdf=realdf[~((realdf.animal=='e200')&((realdf.day.isin([67,68,81]))))]
# realdf=realdf[~((realdf.animal=='e218')&(realdf.day.isin([41,55])))]
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
pivoted_avg = pivoted_avg.rename(columns={False: 'goaln_cell_prop_off', True: 'goal_cell_prop_on'})
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
fig,ax=plt.subplots(figsize=(4,4))
sns.scatterplot(x=beh.rates_diff.values, y=pivoted_avg.difference.values,hue=pivoted_avg.condition.values,s=300,alpha=.7,palette=pl,ax=ax)
ax.plot(beh.rates_diff.values, intercept + slope * beh.rates_diff.values, color='steelblue', linewidth=3)
ax.legend(['Control', 'VIP Inhibition', 'VIP Excitation'], fontsize='small')
ax.set_xlabel("% Correct trials (LEDon-LEDoff)")
ax.set_ylabel("Reward cell %")
ax.set_title(f"r={r_value:.3g}, p={p_value:.3g}",fontsize=16)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Reward cell vs. performance')
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
# results_pre, results_post, results_pre_early, results_post_early
# get rew cells id to compare to dff
pre = [xx[0]['goal_id'] for xx in results_all]
post = [xx[1]['goal_id'] for xx in results_all]
pre_early= [xx[2]['goal_id'] for xx in results_all]
post_early = [xx[3]['goal_id'] for xx in results_all]

# get dff rew cells and compare
sv = r'Z:\condition_df\goal_cell_id_dff_tc_opto.p'

with open(sv, "rb") as fp: #unpickle
    results_dff = pickle.load(fp)

pre_dff = [xx[0]['goal_id'] for xx in results_dff]
post_dff = [xx[1]['goal_id'] for xx in results_dff]
pre_early_dff= [xx[2]['goal_id'] for xx in results_dff]
post_early_dff = [xx[3]['goal_id'] for xx in results_dff]

# %%

labels = ['pre', 'post', 'pre_early', 'post_early']
reg_sets = [pre, post, pre_early, post_early]
dff_sets = [pre_dff, post_dff, pre_early_dff, post_early_dff]

# Plot Venn diagram per animal
for i in range(len(results_all[:10])):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for j, label in enumerate(labels):
        reg_ids = set(reg_sets[j][i])
        dff_ids = set(dff_sets[j][i])

        venn2([reg_ids, dff_ids], set_labels=('fc3', 'dF/F'), ax=axs[j])
        
        axs[j].set_title(f'{label} - animal {i}')

    plt.tight_layout()
    plt.suptitle(f'Goal Cell ID Overlap - Animal {i}', fontsize=14, y=1.02)
    plt.show()
# %%
