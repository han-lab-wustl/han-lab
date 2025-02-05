"""get lick rate in old reward zone 
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# plt.rc('font', size=20)          # controls default text sizes
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import lick_selectivity_current_and_prev_reward, get_rewzones

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior_licks.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'

lick_selectivity = {} # collecting
for dd,day in enumerate(conddf.days.values):
    animal = conddf.animals.values[dd]
    in_type = 'vip' if conddf.in_type.values[dd]=='vip' else 'ctrl'             
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
    opto_ep = eptest
    lick_selectivity_per_trial_opto,
lick_selectivity_per_trial_opto_prevrew = lick_selectivity_current_and_prev_reward(opto_ep, eps, trialnum, rewards, licks, \
    ybinned, rewlocs, forwardvel, rewsize)        
    rewzones = get_rewzones(rewlocs, 1/scalingf)
    rewzone = rewzones[opto_ep-1]
    rewzone_prev = rewzones[opto_ep-2]
    lick_selectivity[f'{animal}_{in_type}_{day:03d}_rz{int(rewzone)}_{int(rewzone_prev)}_{dd:03d}'] = [lick_selectivity_per_trial_opto,lick_selectivity_per_trial_opto_prevrew] 

#%%
# plot
plt.rc('font', size=20)          # controls default text sizes
df = conddf
df['lick_selectivity_last5trials'] = np.array([np.nanmean(v[0]) for k,v in lick_selectivity.items()])
df['lick_selectivity_last5trials_prevrewloc'] = np.array([np.nanmean(v[1]) for k,v in lick_selectivity.items()])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['opto'] = ['LED on' if optoep>1 else 'LED off' for optoep in df.optoep.values]

df = df[(df.animals!='e189')&(df.animals!='e190')]
dfagg = df.groupby(['animals', 'opto', 'condition']).mean(numeric_only = True)
# drop 1st row
dfagg = dfagg.sort_values('opto')

plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='lick_selectivity_last5trials_prevrewloc', hue='condition', data=dfagg, fill=False,
                errorbar='se',
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x='opto', y='lick_selectivity_last5trials_prevrewloc', hue='condition', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(savedst, 'lick_rate_first_5_trials.svg'), bbox_inches='tight')


