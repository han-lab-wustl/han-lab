"""
quantify licks and velocity during consolidation task
"""


import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import consecutive_stretch, get_behavior_tuning_curve, get_success_failure_trials
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')
# save to pdf
condrewloc = pd.read_csv(r"Z:\condition_df\chr2_grab.csv", index_col = None)
src = r"Z:\chr2_grabda\e232"
animal = os.path.basename(src)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"{animal}_opto_peri_analysis.pdf"))
days = [42,43,44,45,46,47,48,49,50]
range_val = 10; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
optodays = [43,44,45,48,49] # memory of opto days rewloc (aka get the next day)

near_reward_per_day = []
for day in days: 
    newrewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
    rewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0]
    # for each plane
    path=list(Path(os.path.join(src, str(day))).rglob('params.mat'))[0]
    params = scipy.io.loadmat(path)
    print(path)
    VR = params['VR'][0]
    # dtype=[('name_date_vr', 'O'), ('ROE', 'O'), ('lickThreshold', 'O'), ('reward', 'O'), 
    # ('time', 'O'), ('lick', 'O'), ('ypos', 'O'), 
    #          ('lickVoltage', 'O'), ('trialNum', 'O'), ('timeROE', 'O'), ('changeRewLoc', 'O'), ('pressedKeys', 'O'), ('world', 'O'), 
    #          ('imageSync', 'O'), ('scalingFACTOR', 'O'), ('wOff', 'O'),
    #          ('catchTrial', 'O'), ('optoTrigger', 'O'), ('settings', 'O')]) 
    velocity = VR[0][1][0]
    lick = VR[0][5][0]
    time = VR[0][4][0]
    velocity=-0.013*velocity[1:]/np.diff(time) # make same size
    velocity = np.append(velocity, np.interp(len(velocity)+1, np.arange(len(velocity)),velocity))
    velocitydf = pd.DataFrame({'velocity': velocity})
    velocity = np.hstack(velocitydf.rolling(10).mean().values)
    rewards = VR[0][3][0]
    ypos = VR[0][6][0]/(2/3)
    trialnum = VR[0][8][0]
    changerewloc = VR[0][10][0]
    rews_centered = np.zeros_like(velocity)
    rews_centered[(ypos >= rewloc-5) & (ypos <= rewloc)]=1
    rews_iind = consecutive_stretch(np.where(rews_centered)[0])
    min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
    rews_centered = np.zeros_like(velocity)
    rews_centered[min_iind]=1
    success, fail, str_trials, ftr_trials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
    # probe trials
    probe = trialnum<str_trials[0] # trials before first successful trial as probes
    com_probe = np.nanmean(ypos[probe][lick.astype(bool)[probe]])-rewloc
    pos_bin, lick_probability_probe = get_behavior_tuning_curve(ypos[probe], lick[probe], bins=270)
    pos_bin, vel_probe = get_behavior_tuning_curve(ypos[probe], velocity[probe], bins=270)
    lick_probability_probe_near_reward = lick_probability_probe.values[int(rewloc)-20:int(rewloc)+20]
    vel_probe_near_reward = vel_probe.values[int(rewloc)-30:int(rewloc)+10]
    near_reward_per_day.append([lick_probability_probe_near_reward,vel_probe_near_reward,com_probe])    
    
#%%
df = pd.DataFrame()
df['days'] = days
df['opto_day_before'] = [xx in optodays for xx in days]
df['lick_prob_near_rewardloc_mean'] = [np.nanmean(xx[0]) for xx in near_reward_per_day]
df['velocity_near_rewardloc_mean'] = [np.nanmean(xx[1]) for xx in near_reward_per_day]
df['com_lick_probe'] = [xx[2] for xx in near_reward_per_day]
# df['lick_prob_near_rewardloc_mean'] = [np.quantile(xx[0], .9) for xx in near_reward_per_day]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]

plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto_day_before', y='velocity_near_rewardloc_mean', hue='opto_day_before', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto_day_before', y='velocity_near_rewardloc_mean', hue='opto_day_before', data=df,
                palette={False: "slategray", True: "mediumturquoise"})
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto_day_before', y='lick_prob_near_rewardloc_mean', hue='opto_day_before', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto_day_before', y='lick_prob_near_rewardloc_mean', hue='opto_day_before', data=df,
                palette={False: "slategray", True: "mediumturquoise"})
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df,
                palette={False: "slategray", True: "mediumturquoise"})

scipy.stats.ranksums(df.loc[df.opto_day_before==True, 'lick_prob_near_rewardloc_mean'].values,
                    df.loc[df.opto_day_before==False, 'lick_prob_near_rewardloc_mean'].values)

scipy.stats.ranksums(df.loc[df.opto_day_before==True, 'com_lick_probe'].values,
                    df.loc[df.opto_day_before==False, 'com_lick_probe'].values)
