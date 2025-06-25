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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto,get_dff_opto
import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
      radian_alignment_saved = pickle.load(fp)

# tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early
# get all cell activity
dff=[]
for ii in range(len(conddf)):
   if ii!=191:
      if ii%10==0: print(ii)
      dff.append(get_dff_opto(conddf, ii, pc=True))
#%%
s =12 # pointsize
a=.7
# plot relative fc3 transients
plt.rc('font', size=20)          # controls default text sizes
dfc = conddf.copy()[:len(dff)]
df=pd.DataFrame()
df['dff_target'] = np.concatenate([np.nanmean(xx[0],axis=0) for xx in dff])
df['dff_prev'] = np.concatenate([np.nanmean(xx[1],axis=0) for xx in dff])
df['dff_target-prev'] = df['dff_target']-df['dff_prev']
condition = [xx if 'vip' in xx else 'ctrl' for xx in dfc.in_type.values]
df['condition'] = np.concatenate([[cond]*(dff[ii][0].shape[1]) for ii,cond in enumerate(condition)])
df['animals'] = np.concatenate([[cond]*(dff[ii][0].shape[1]) for ii,cond in enumerate(dfc.animals)])
df['days'] = np.concatenate([[cond]*(dff[ii][0].shape[1]) for ii,cond in enumerate(dfc.days)])
df['optoep'] = np.concatenate([[cond]*(dff[ii][0].shape[1]) for ii,cond in enumerate(dfc.optoep)])

# df['opto'] = df.optoep.values>1
df=df[~((df.animals=='z15')&(df.days<8))]
df=df[~((df.animals=='e217')&(df.days<9)&(df.days==26))]
df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([57]))))]
df=df[~((df.animals=='e200')&((df.days.isin([67,68,81]))))]
# df=df[~((df.animals=='e218')&(df.days>44))]
# df=df[df['dff_target-prev']<.4] # xclude outlier fl days
#%%
dforg = df.loc[~((df.animals=='e190')|(df.animals=='e189'))] # exclude noisy days
 =dforg[dforg.optoep.values>1]
df=dforg.groupby(['animals','condition']).median(numeric_only=True)
df=df.reset_index()
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
fig,ax = plt.subplots(figsize=(3,5))
sns.stripplot(x="condition", y="dff_target", hue = 'condition',s=12, data=df,palette=pl, ax=ax,alpha=.7)
sns.barplot(x="condition", y="dff_target", hue = 'condition', data=df,palette=pl, ax=ax,fill=False,errorbar='se')
# sns.violinplot(x="condition", y="dff_target-prev", hue = 'condition', data=df,palette=pl,ax=ax) 

ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Mean $\\Delta F/F$')
# ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xlabel('')
# Prepare groups by condition
groups = [group['dff_target-prev'].values for name, group in df.groupby('condition')]
# Unique conditions
conds = df['condition'].unique()
# Collect data by condition
data_by_group = {cond: df[df['condition'] == cond]['dff_target'].values for cond in conds}
# Perform Kruskal-Wallis H-test
h_stat, p_val = scipy.stats.kruskal(*groups)
import itertools
import statsmodels.stats.multitest as smm
# Generate all pairwise comparisons
comparisons = list(itertools.combinations(conds, 2))[:2]

# Perform Mann-Whitney U tests
pvals = []
results = []
for (g1, g2) in comparisons:
    stat, p = scipy.stats.ranksums(data_by_group[g1], data_by_group[g2], alternative='two-sided')
    pvals.append(p)
    results.append((g1, g2, stat, p))

# Multiple comparisons correction
# rejected, pvals_corrected = smm.multipletests(pvals, method='fdr_bh')[:2]
def get_xtick_positions(ax, conds):
    xticks = []
    for label in conds:
        for i, tick in enumerate(ax.get_xticklabels()):
            if tick.get_text() == label:
                xticks.append(i)
                break
    return xticks

xtick_map = get_xtick_positions(ax, conds)
y_max = df['dff_target-prev'].max()
h=0.03
for idx, ((g1, g2, _, raw_p), corr_p, sig) in enumerate(zip(results, pvals, rejected)):
      x1, x2 = xtick_map[conds.tolist().index(g1)], xtick_map[conds.tolist().index(g2)]
      y = y_max + h + idx * h/3
      ax.plot([x1, x1, x2, x2], [y - 0.003, y, y, y - 0.003], lw=1.5, color='k')
      annotation = f"p = {corr_p:.3f}"
      ax.text((x1 + x2) / 2, y - 0.015, annotation, ha='center', va='bottom', fontsize=14)
ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)

# Print results
print("\nPost hoc Mann-Whitney U tests (Bonferroni-corrected):")
for i, (g1, g2, stat, raw_p) in enumerate(results):
    print(f"{g1} vs. {g2} -> U={stat:.2f}, raw p={raw_p:.4f}, corrected p={pvals_corrected[i]:.4f}, significant={rejected[i]}")
print("Kruskal-Wallis results:")
print(f"H-test statistic = {h_stat:.4f}, p-value = {p_val:.4f}")
plt.savefig(os.path.join(savedst, 'dff.svg'), bbox_inches='tight')
