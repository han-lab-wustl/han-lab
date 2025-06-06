"""
quantify lick selectivity
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, pandas as pd, random, scikit_posthocs as sp
from behavior import get_performance, get_success_failure_trials, get_lick_selectivity, lick_selectivity_probes,lick_selectivity_fails
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# plt.rc('font', size=12)          # controls default text sizes
plt.close('all')

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\aha'

dcts = []
lick_selectivity_trial_type = []
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
    ybinned = np.hstack(VR['ypos']/scalingf)
    rewlocs = np.ceil(np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf).astype(int)
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
    trialnum = np.hstack(VR['trialNum'])
    rewards = np.hstack(VR['reward'])
    forwardvel = np.hstack(VR['ROE']); time =np.hstack(VR['time'])
    forwardvel=-0.013*forwardvel[1:]/np.diff(time) # make same size
    forwardvel = np.append(forwardvel, np.interp(len(forwardvel)+1, np.arange(len(forwardvel)),forwardvel))
    licks = np.hstack(VR['lickVoltage'])
    licks = licks<=-0.065 # remake boolean
    eptest = conddf.optoep.values[dd]    
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)   
        if len(eps)<4: eptest = 2 # if no 3 epochs 

    rates_opto, rates_prev, lick_prob_opto, \
    lick_prob_prev, trials_bwn_success_opto, \
    trials_bwn_success_prev, vel_opto, vel_prev, \
    lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev,lick_rate_opto, lick_rate_prev = get_performance(eptest, 
        eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize) 
    lick_selectivity_post_opto_probes, lick_selectivity_post_prev_probes = lick_selectivity_probes(eptest, 
        eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize) 
    lick_selectivity_fails_opto, lick_selectivity_fails_prev = lick_selectivity_fails(eptest, 
        eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize) 
    lick_selectivity_trial_type.append([lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev,
                lick_selectivity_post_opto_probes,lick_selectivity_post_prev_probes,
                lick_selectivity_fails_opto, lick_selectivity_fails_prev,lick_rate_opto, lick_rate_prev]) 
#%%
s=12 # poinstize
plt.rc('font', size=20)          # controls default text sizes
df = conddf[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_last5trials_targetep'] = np.array([np.nanmean(xx[0]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_last5trials_prevep'] = np.array([np.nanmean(xx[1]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_probes_targetep'] = np.array([np.nanmean(xx[2]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_probes_prevep'] = np.array([np.nanmean(xx[3]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_fails_targetep'] = np.array([np.nanmean(xx[4]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_fails_prevep'] = np.array([np.nanmean(xx[5]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_rate_targetep'] = np.array([np.nanmean(xx[6]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_rate_prevep'] = np.array([np.nanmean(xx[7]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]

df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['opto'] = ['LED on' if optoep>1 else 'LED off' for optoep in df.optoep.values]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]
# df = df.drop([12])
df = df[(df.animals!='e189')&(df.animals!='e190')]
df = df[df.optoep>1]
dfagg = df.groupby(['animals', 'opto', 'condition']).mean(numeric_only = True)
# drop 1st row
# performance on opto days
dfagg = dfagg.sort_values('opto')
dfagg['lick_selectivity_last5trials'] = dfagg['lick_selectivity_last5trials_targetep']-dfagg['lick_selectivity_last5trials_prevep']
dfagg['lick_selectivity_probes'] = dfagg['lick_selectivity_probes_targetep']-dfagg['lick_selectivity_probes_prevep']
dfagg['lick_selectivity_fails'] = dfagg['lick_selectivity_fails_targetep']-dfagg['lick_selectivity_fails_prevep']
dfagg['lick_rate'] = dfagg['lick_rate_targetep']-dfagg['lick_rate_prevep']
#%%
# drop 1st row
# performance on opto days
dfagg = dfagg.sort_values('opto')
plt.figure(figsize=(2,5))
ax = sns.barplot(x='condition', y='lick_rate', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='condition', y='lick_rate', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=s)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(f'Lick rate (LEDoff-LEDon)')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_xlabel('')
# lick rate
x1 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_rate'].values
x3 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_rate'].values
t, pval = scipy.stats.ranksums(x1, x3)

# plt.savefig(os.path.join(savedst, 'lick_rate_first_5_trials.svg'), bbox_inches='tight')
#%%
# lick selectivity last 5 trials
plt.figure(figsize=(2,5))
ax = sns.barplot(x='condition', y='lick_selectivity_last5trials', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='condition', y='lick_selectivity_last5trials', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(f'Lick selectivity (LEDoff-LEDon)')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_xlabel('')

# lick s
x1 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_last5trials'].values
x3 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_last5trials'].values
t, pval = scipy.stats.ranksums(x1, x3)
# statistical annotation    
fs=46
ii=0.5; y=.05; pshift=.07
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

from statsmodels.stats.power import TTestPower

# Extract data
vip_led_on = dfagg.loc[(dfagg.index.get_level_values('condition') == 'vip') &
                       (dfagg.index.get_level_values('opto') == 'LED on'), 'lick_selectivity_last5trials'].values
ctrl_led_on = dfagg.loc[(dfagg.index.get_level_values('condition') == 'ctrl') &
                        (dfagg.index.get_level_values('opto') == 'LED on'), 'lick_selectivity_last5trials'].values

# Calculate effect size
mean_diff = np.mean(vip_led_on) - np.mean(ctrl_led_on)
pooled_std = np.sqrt((np.std(vip_led_on, ddof=1) ** 2 + np.std(ctrl_led_on, ddof=1) ** 2) / 2)

# Cohen's d
cohen_d = mean_diff / pooled_std
# Initialize power analysis object
analysis = TTestPower()
# Calculate required sample size for 80% power
sample_size = analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, alternative='two-sided')
print(f"Estimated required sample size per group: {np.ceil(sample_size):.0f}")

# plt.savefig(os.path.join(savedst, 'lick_selectivity_success_trials.svg'), bbox_inches='tight')
#%%
# probes
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='lick_selectivity_probes', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='opto', y='lick_selectivity_probes', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(savedst, 'lick_selectivity_probes.svg'), bbox_inches='tight')
# fails
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='lick_selectivity_fails', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='opto', y='lick_selectivity_fails', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(savedst, 'lick_selectivity_fails.svg'), bbox_inches='tight')

totest = 'lick_selectivity_last5trials'
x1 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_last5trials'].values
x2 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED off')), 'lick_selectivity_last5trials'].values
x3 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_last5trials'].values
x4 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED off')), 'lick_selectivity_last5trials'].values
labels = ['vipledon', 'vipledoff', 'ctrlledon', 'ctrlledoff']
scipy.stats.f_oneway(x1, x2, x3, x4)
p_values= sp.posthoc_ttest([x1,x2,x3,x4])#,p_adjust='holm-sidak')
p_values.columns = labels
p_values.index = labels
print(p_values)

# probes
x1 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_probes'].values
x2 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'vip') & 
            (dfagg.index.get_level_values('opto') == 'LED off')), 'lick_selectivity_probes'].values
x3 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED on')), 'lick_selectivity_probes'].values
x4 = dfagg.loc[((dfagg.index.get_level_values('condition') == 'ctrl') & 
            (dfagg.index.get_level_values('opto') == 'LED off')), 'lick_selectivity_probes'].values
labels = ['vipledon', 'vipledoff', 'ctrlledon', 'ctrlledoff']
scipy.stats.f_oneway(x1, x2, x3[~np.isnan(x3)], x4)
p_values= sp.posthoc_ttest([x1,x2,x3[~np.isnan(x3)],x4])#,p_adjust='holm-sidak')
p_values.columns=labels
print(p_values)

