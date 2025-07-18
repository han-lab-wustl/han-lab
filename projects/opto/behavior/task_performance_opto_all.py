"""
TODO: lick rate
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from behavior import get_success_failure_trials,\
get_performance, get_rewzones
from statsmodels.formula.api import ols
import scipy.stats as stats, itertools
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm  # <-- Correct import

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
# corresponding to days analysing

bin_size = 3
# com shift analysis
dcts = []
for dd,day in enumerate(conddf.days.values):
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['VR'])
    VR = fall['VR'][0][0][()]
    eps = np.where(np.hstack(VR['changeRewLoc']>0))[0]
    eps = np.append(eps, len(np.hstack(VR['changeRewLoc'])))
    scalingf = VR['scalingFACTOR'][0][0]
    track_length = 180/scalingf
    nbins = track_length/bin_size
    ybinned = np.hstack(VR['ypos']/scalingf)
    rewlocs = np.ceil(np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf).astype(int)
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
    trialnum = np.hstack(VR['trialNum'])
    rewards = np.hstack(VR['reward'])
    forwardvel = np.hstack(VR['ROE']); time =np.hstack(VR['time'])
    forwardvel=-0.013*forwardvel[1:]/np.diff(time) # make same size
    forwardvel = np.append(forwardvel, np.interp(len(forwardvel)+1, np.arange(len(forwardvel)),forwardvel))
    licks = np.hstack(VR['lick'])
    eptest = conddf.optoep.values[dd]
    # sometimes the epoch tested randomly is not long enough
    # then, change epoch test to the previous epoch
    # should only be the case for nonopto days
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)   
        if len(eps)<4: eptest = 2 # if no 3 epochs    
        while max(trialnum[eps[eptest-1]:eps[eptest]])<4: 
            eptest=eptest-1
    # 15,8 for first and last for opto data
    rates_opto, rates_prev, lick_prob_opto, \
    lick_prob_prev, trials_bwn_success_opto, \
    trials_bwn_success_prev, vel_opto, vel_prev, lick_selectivity_per_trial_opto,\
    lick_selectivity_per_trial_prev, lick_rate_opto, lick_rate_prev, com_opto, com_prev,\
        lick_rate_opto_late, lick_rate_prev_late, lick_selectivity_per_trial_prev_early, lick_selectivity_per_trial_opto_early= get_performance(eptest, eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, time, rewsize,firsttr=8,lasttr=8)
    rewzones = get_rewzones(rewlocs, 1/scalingf)
    # save
    dct['velocity'] = [vel_prev, vel_opto]
    dct['lick_selectivity'] = [lick_selectivity_per_trial_prev, lick_selectivity_per_trial_opto]
    dct['lick_selectivity_early'] = [lick_selectivity_per_trial_prev_early, lick_selectivity_per_trial_opto_early]
    dct['com'] = [com_prev, com_opto]
    dct['rates'] = [rates_prev, rates_opto]    
    dct['lick_p'] = [lick_prob_prev, lick_prob_opto]
    dct['rewlocs'] = [rewlocs[eptest-2], rewlocs[eptest-1]]
    dct['rewzones'] = [rewzones[eptest-2], rewzones[eptest-1]]
    dct['trials_before_success'] = [trials_bwn_success_prev, trials_bwn_success_opto]
    dct['lick_rate'] = [lick_rate_prev, lick_rate_opto]
    dct['lick_rate_late'] = [lick_rate_prev_late, lick_rate_opto_late]
    dcts.append(dct)
#%%
# plot performance 
s = 12 # pontsize
a=0.7
dcts_opto = np.array(dcts)
df = conddf
df['rates_diff'] = [np.diff(dct['rates'])[0] for dct in dcts]
df['rates_diff']=df['rates_diff']*100
# df['rates_diff'] = [dct['rates'][1] for dct in dcts]
df['velocity_diff'] = [np.diff(dct['velocity'])[0] for dct in dcts]
df['velocity'] = [dct['velocity'][0] for dct in dcts]
df['lick_rate'] = [np.nanmean(dct['lick_rate'][1]) for dct in dcts] # opto
df['lick_rate_diff'] = [np.nanmean(dct['lick_rate'][1])-np.nanmean(dct['lick_rate'][0]) for dct in dcts] # opto
df['lick_rate_late'] = [np.nanmean(dct['lick_rate_late'][1]) for dct in dcts] # opto
df['lick_rate_diff_late'] = [np.nanmean(dct['lick_rate_late'][1])-np.nanmean(dct['lick_rate_late'][0]) for dct in dcts] # opto
# com opto
df['com'] = [dct['com'][1] for dct in dcts]
df['lick_selectivity']=[np.nanmean(dct['lick_selectivity'][1])-np.nanmean(dct['lick_selectivity'][0]) for dct in dcts]
df['lick_selectivity_early']=[np.nanmean(dct['lick_selectivity_early'][1])-np.nanmean(dct['lick_selectivity_early'][0]) for dct in dcts]
df['rewzone_transition'] = [str(tuple(dct['rewzones'])) for dct in dcts]
df['opto'] = conddf.optoep.values>1
df['condition'] = ['ctrl' if 'vip' not in xx else xx for xx in conddf.in_type.values]
df = df[(df.animals!='e189') & (df.animals!='e190')]
df=df[(df.optoep.values>1)]
df=df[~((df.animals=='z15')&((df.days<6)|(df.days.isin([15,16]))))]
df=df[~((df.animals=='z14')&(df.days<33))]
df=df[~((df.animals=='z17')&(df.days.isin([3,4,11,12,18,22,23])))]
df=df[~((df.animals=='z9')&(df.days.isin([20,15])))]
# df=df[~((df.animals=='e201')&((df.days.isin([58,61]))))]
# df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([57]))))]
df=df[~((df.animals=='e200')&((df.days.isin([85]))))]
df=df[~((df.animals=='e218')&(df.days==55))]
# plot rates vip vs. ctl led off and on
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
bigdf_plot = df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True).reset_index()

# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['rates_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['rates_diff'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Plot
fig, ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="rates_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="rates_diff", hue='condition', data=bigdf_plot,
              palette=pl, alpha=a, s=s, ax=ax)

ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('% Correct trials (LEDoff-LEDon)')
ax.set_xlabel('')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + 1, y_pos + 1, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos-3, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos-2, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['rates_diff'].max() + 1
gap = 5
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'p_correct_trials_opto_all.svg'),  bbox_inches='tight')
# df.to_csv(r'Z:\condition_df\vip_opto_behavior.csv')
#%%
# Step 1: Calculate the means and standard deviations
group1=x1;group2=x2
mean1 = np.mean(group1)
mean2 = np.mean(group2)
std1 = np.std(group1, ddof=1)
std2 = np.std(group2, ddof=1)
# Step 2: Calculate pooled standard deviation
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
# Step 3: Calculate Cohen's d
cohens_d = (mean1 - mean2) / pooled_std
# Step 4: Perform Power Analysis using the calculated Cohen's d
alpha = 0.05  # Significance level
power = 0.8   # Desired power
from statsmodels.stats import power as smp
analysis = smp.TTestIndPower()
sample_size = analysis.solve_power(effect_size=cohens_d, alpha=alpha, power=power, alternative='two-sided')
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Required sample size per group: {sample_size:.2f}")
#%%
# velocity diff

# plot lick selectivity and lick com
# bigdf_plot = df.groupby(['animals', 'condition', 'opto']).median(numeric_only=True)
a=0.7
fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="velocity",hue='condition', data=bigdf_plot,
    palette=pl,                
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="velocity",hue='condition', data=bigdf_plot,
            palette=pl,alpha=a,                
            s=s,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=30)
ax.set_ylabel(f'Velocity (cm/s; LEDon)')
ax.set_xlabel('')

# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['velocity'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['velocity'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
p_vals_corrected=p_vals
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['velocity'].max()
gap = 2
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i],height=1.5)
    y_start += gap
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'velocity_opto_all.svg'),  bbox_inches='tight')

#%%
# plot lick selectivity and lick com
# bigdf_plot = df.groupby(['animals', 'condition', 'opto']).median(numeric_only=True)
a=0.7
fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="lick_selectivity",hue='condition', data=bigdf_plot,
    palette=pl,                
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="lick_selectivity",hue='condition', data=bigdf_plot,
            palette=pl,alpha=a,                
            s=s,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel(f'Lick Selectivity (LEDon-LEDoff)')
ax.set_xlabel('')

model = ols('lick_selectivity ~ C(condition)', data=bigdf_plot).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_selectivity'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_selectivity'].dropna()
    stat, p = stats.ranksums(x1, x2)
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['lick_selectivity'].max() + .01
gap = .05
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i])
    y_start += gap
plt.tight_layout()
ax.set_title('Last 8 trials')
plt.savefig(os.path.join(savedst, 'lick_selectivity_opto_all.svg'),  bbox_inches='tight')
# plt.savefig(os.path.join(savedst, 'com.svg'),  bbox_inches='tight')
#%%
# only 3 to 1
# bigdf_plot=bigdf_plot[bigdf_plot.animals!='e190']
bigdf_plot = df.groupby(['animals', 'days','condition', 'opto','rewzone_transition']).mean(numeric_only=True)
bigdf_plot = bigdf_plot.reset_index()
df2plt = bigdf_plot[(bigdf_plot.rewzone_transition=='(3.0, 1.0)')|(bigdf_plot.rewzone_transition=='(2.0, 1.0)')|(bigdf_plot.rewzone_transition=='(3.0, 2.0)')]
df2plt = bigdf_plot[(bigdf_plot.rewzone_transition=='(3.0, 1.0)')|(bigdf_plot.rewzone_transition=='(2.0, 1.0)')|(bigdf_plot.rewzone_transition=='(3.0, 2.0)')]

df2plt = df2plt.groupby(['animals', 'days','condition', 'opto']).mean(numeric_only=True).reset_index()
df2plt = df2plt[(df2plt.animals!='e189')&(df2plt.animals!='e190')]

fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="com",hue='condition', data=df2plt, 
            palette=pl,                     
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="com",hue='condition', data=df2plt, palette=pl, alpha=a,
            s=9,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Normalized center-of-mass licks')
# ax.axhline(0, color='k')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_title('Reward zone 3 $\\rightarrow$ 1')
ax.set_xlabel('')
# Get N per condition
n_per_condition = (
    df2plt
    .groupby('condition')[['animals', 'days']]
    .nunique()  # counts unique animals/days per condition
    .reset_index()
)
n_per_condition['n'] = n_per_condition[['animals', 'days']].max(axis=1)

# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:2]
p_vals = []
for c1, c2 in comparisons:
    x1 = df2plt[df2plt['condition'] == c1]['com'].dropna()
    x2 = df2plt[df2plt['condition'] == c2]['com'].dropna()
    stat, p = stats.ranksums(x1, x2)
    p_vals.append(p)
# Correct for multiple comparisons
# reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['lick_selectivity'].max() + 20
gap = 5
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals[i],xoffset=5,height=1)
    y_start += gap
    
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'far2near_com_opto.svg'),  bbox_inches='tight')

#%%
# near to far
# bigdf_plot=bigdf_plot[bigdf_plot.animals!='e190']
bigdf_plot = df.groupby(['animals', 'days','condition', 'opto','rewzone_transition']).mean(numeric_only=True)
bigdf_plot = bigdf_plot.reset_index()
df2plt = bigdf_plot[(bigdf_plot.rewzone_transition=='(1.0, 3.0)')|(bigdf_plot.rewzone_transition=='(1.0, 2.0)')]
df2plt = df2plt.groupby(['animals', 'days','condition', 'opto']).mean(numeric_only=True).reset_index()

fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="com",hue='condition', data=df2plt, 
            palette=pl,                     
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="com",hue='condition', data=df2plt, palette=pl, alpha=a,
            s=10,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'COM Licks-Reward Loc. (cm)')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_title('Near to Far')
ax.set_xlabel('')

model = ols('com ~ C(condition)', data=df2plt).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'near2far_com_opto.svg'),  bbox_inches='tight')
#%%
# lick rate diff
bigdf_plot = df.groupby(['animals','condition', 'opto']).mean(numeric_only=True).reset_index()
bigdf_plot = bigdf_plot[(bigdf_plot.animals!='e190')&(bigdf_plot.animals!='e189')]

fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="lick_rate_diff",hue='condition', data=bigdf_plot, 
            palette=pl,                     
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="lick_rate_diff",hue='condition', data=bigdf_plot,                
              palette=pl, alpha=a,
            s=s,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Pre-reward lick rate (licks/s)')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation =20)
ax.set_xlabel('')
fig.tight_layout()

# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_rate_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_rate_diff'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=46)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['lick_rate_diff'].max()
gap = .2
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i],xoffset=2,height=.1)
    y_start += gap

model = ols('lick_rate_diff ~ C(condition)', data=bigdf_plot).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
plt.savefig(os.path.join(savedst, 'init_lick_rate_opto.svg'),  bbox_inches='tight')
#%%
# early lick selectivity

# plot lick selectivity and lick com
# bigdf_plot = df.groupby(['animals', 'condition', 'opto']).median(numeric_only=True)
a=0.7
fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(x="condition", y="lick_selectivity_early",hue='condition', data=bigdf_plot,
    palette=pl,                
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="lick_selectivity_early",hue='condition', data=bigdf_plot,
            palette=pl,alpha=a,                
            s=s,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=30)
ax.set_ylabel(f'Early lick Selectivity (LEDon-LEDoff)')
ax.set_xlabel('')

model = ols('lick_selectivity_early ~ C(condition)', data=bigdf_plot).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_selectivity_early'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_selectivity_early'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['lick_selectivity_early'].max() + .01
gap = .02
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i],height=0.007)
    y_start += gap
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'lick_selectivity_early_opto_all.svg'),  bbox_inches='tight')
#%%
#%%
