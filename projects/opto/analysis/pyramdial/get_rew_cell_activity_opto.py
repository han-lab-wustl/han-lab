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
   if ii!=187:
      if ii%10==0: print(ii)
      dff.append(get_dff_opto(conddf, ii))
#%%
s =12 # pointsize
a=.7
# plot relative fc3 transients
plt.rc('font', size=20)          # controls default text sizes
df = conddf.copy()[:len(dff)]
df['dff_target'] = np.array(dff)[:,0]
df['dff_prev'] = np.array(dff)[:,1]
df['dff_target-prev'] = df['dff_target']#-df['dff_prev']
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
# df['opto'] = df.optoep.values>1
df=df[~((df.animals=='z14')&(df.days<15))]
df=df[~((df.animals=='z15')&(df.days<8))]
df=df[~((df.animals=='z17')&(df.days<11))]
df=df[~((df.animals=='e217')&((df.days<9)|(df.days==26)))]
df=df[~((df.animals=='e216')&((df.days<32)))]
df=df[~((df.animals=='e200')&((df.days.isin([67]))))]
df=df[~((df.animals=='e218')&(df.days>44))]

# only initial days as controls
# df['opto'] = [True if xx>1 else False if xx==-1 else np.nan for xx in conddf.optoep.values]
# df = df.loc[~((df.animals=='e217')&(df.days.isin([12,26,29])))] # exclude noisy days
df = df.loc[~((df.animals=='e190')|(df.animals=='e189'))] # exclude noisy days
df=df[df.optoep.values>1]
df=df.groupby(['animals', 'condition']).mean(numeric_only=True)
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
fig,ax = plt.subplots(figsize=(3,5))
sns.barplot(x="condition", y="dff_target-prev", hue = 'condition', data=df,
                palette=pl,
                errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="dff_target-prev", hue = 'condition', data=df,
                palette=pl,
                s=s,ax=ax,alpha=a)

ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Norm. $\\Delta$ F/F (LEDon-LEDoff)')
# ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xlabel('')
# Prepare groups by condition
groups = [group['dff_target-prev'].values for name, group in df.groupby('condition')]
ax.set_xticklabels(['Control', 'VIP\nInhibtion', 'VIP\nExcitation'], rotation=20)
# Perform Kruskal-Wallis H-test
h_stat, p_val = scipy.stats.kruskal(*groups)

print("Kruskal-Wallis results:")
print(f"H-test statistic = {h_stat:.4f}, p-value = {p_val:.4f}")
plt.savefig(os.path.join(savedst, 'dff.svg'), bbox_inches='tight')
