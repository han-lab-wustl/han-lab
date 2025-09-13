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
from statsmodels.stats.multitest import multipletests

# tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early
# get all cell activity
dff=[]
for ii in range(len(conddf)):
   if ii!=191:
      if ii%10==0: print(ii)
      dff.append(get_dff_opto(conddf, ii, pc=True))
#%%
s =10 # pointsize
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
#%%
dfnew=pd.DataFrame()
dfnew['dff'] = np.concatenate([df.dff_target,df.dff_prev])
dfnew['epoch'] = np.concatenate([['opto']*len(df.dff_target),['prev']*len(df.dff_prev)])
dfnew['condition'] = np.concatenate([df.condition]*2)
dfnew['animals'] = np.concatenate([df.animals]*2)
dfnew['days'] = np.concatenate([df.days]*2)
dfnew['optoep'] = np.concatenate([df.optoep]*2)
dfnew['dff_target-prev'] = np.concatenate([df.dff_target,df.dff_prev])
dfnew = dfnew.loc[~((dfnew.animals=='e190')|(dfnew.animals=='e189'))] # exclude noisy days
dfnew =dfnew[dfnew.optoep.values>1]

#%%
# plot dff
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]

fig, axes = plt.subplots(ncols = 2, figsize=(8,5.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
pl=['k','slategray']
sns.boxplot(x="condition", y="dff", hue='epoch', data=dfnew,hue_order=['prev','opto'],
              fill=False, ax=ax,palette=pl,legend=False,showfliers=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('$\Delta F/F$')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax.set_title('All pyramidal cells')
# ---- add n annotations ----
counts = dfnew.groupby(['condition', 'epoch']).size().reset_index(name='n')

# find positions of boxes
positions = {}
for i, cond in enumerate(dfnew['condition'].unique()):
    for j, epoch in enumerate(['prev', 'opto']):
        positions[(cond, epoch)] = i + j*0.2 - 0.1  # boxplot dodge offset

for (cond, epoch), n in counts.set_index(['condition','epoch'])['n'].items():
    xpos = positions[(cond, epoch)]
    ymax = 0.12
    ax.text(xpos, ymax, f"n={n}", ha='center', va='bottom', fontsize=9)

# ---- aggregate for other panel (as you had) ----
dfagg = dfnew.groupby(['animals','condition']).mean(numeric_only=True).reset_index()


ax=axes[1]
sns.barplot(x="condition", y="dff_target-prev", hue='condition', data=dfagg,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="dff_target-prev", hue='condition', data=dfagg,
              palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('')
ax.set_xlabel('')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, stat,pval, xoffset=0.05,h=.005):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos-h, f'stat={stat:.3g},p={pval:.3g}', ha='center', fontsize=8)
# Pairwise Mann-Whitney U testsn (Wilcoxon rank-sum)
p_vals = []
stats=[]
for c1, c2 in comparisons:
    x1 = dfagg[dfagg['condition'] == c1]['dff_target-prev'].dropna()
    x2 = dfagg[dfagg['condition'] == c2]['dff_target-prev'].dropna()
    stat, p = scipy.stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    stats.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Plot all pairwise comparisons
y_start = dfagg['dff_target-prev'].max()
gap = .002
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, stats[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
ax.set_title('Per mouse')
fig.suptitle('Mean $\Delta F/F$, All pyramidal cells')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'vip_opto_dff.svg'), bbox_inches='tight')
