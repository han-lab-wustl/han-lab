"""
quantify licks and velocity for opto
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, pandas as pd, random
from behavior import get_performance, get_success_failure_trials, get_lick_selectivity
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
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'

bin_size = 3
# com shift analysis
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
    nbins = track_length/bin_size
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
    lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev = get_performance(eptest, 
        eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize)       
    lick_selectivity_trial_type.append([lick_selectivity_per_trial_opto, lick_selectivity_per_trial_prev]) 
#%%
df = conddf[(conddf.optoep!=0) & (conddf.optoep!=1)]
df['lick_selectivity_last5trials_targetep'] = np.array([np.nanmean(xx[0]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['lick_selectivity_last5trials_prevep'] = np.array([np.nanmean(xx[1]) for xx in lick_selectivity_trial_type])[((conddf.optoep!=0) & (conddf.optoep!=1)).values]
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['opto'] = ['LEDon' if optoep>1 else 'LEDoff' for optoep in df.optoep.values]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]
dfagg = df.groupby(['animals', 'opto', 'condition']).mean(numeric_only = True)
# drop 1st row
# performance on opto days
dfagg = dfagg.sort_values('opto')
dfagg['lick_selectivity_last5trials'] = dfagg['lick_selectivity_last5trials_targetep']-dfagg['lick_selectivity_last5trials_prevep']

# drop 1st row
# performance on opto days
dfagg = dfagg.sort_values('opto')

plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='lick_selectivity_last5trials', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='opto', y='lick_selectivity_last5trials', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().set_visible(False)
