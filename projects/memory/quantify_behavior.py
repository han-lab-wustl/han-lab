"""
quantify licks and velocity during consolidation task
"""

import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import consecutive_stretch, get_behavior_tuning_curve, get_success_failure_trials, get_lick_selectivity, \
    get_lick_selectivity_post_reward
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
src = r"Z:\chr2_grabda"
animals = ['e231', 'e232']
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data"
days_all = [[2,3,4,5,6,7,8,9,10,11,12],
        [44,45,46,47,48,49,50,51,54,55]]
range_val = 5; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
# src = r"Z:\chr2_grabda\e231"
# animal = os.path.basename(src)
# days = [2,3,4,5,6,7,8,9,10]
# range_val = 10; binsize=0.2
# planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
# optodays = [3,4,7,8]

near_reward_per_day = []
optodays_before_per_an = []
optodays_per_an = []
performance_opto = []
for ii,animal in enumerate(animals):
    days = days_all[ii]
    optodays_before = []; optodays = []
    for day in days: 
        newrewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
        rewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0]
        optodays_before.append(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'Opto_memory_day'].values[0])
        optodays.append(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'Opto'].values[0])
        # for each plane
        path=list(Path(os.path.join(src, animal, str(day))).rglob('params.mat'))[0]
        params = scipy.io.loadmat(path)
        print(path)
        VR = params['VR'][0][0]
        # dtype=[('name_date_vr', 'O'), ('ROE', 'O'), ('lickThreshold', 'O'), ('reward', 'O'), 
        # ('time', 'O'), ('lick', 'O'), ('ypos', 'O'), 
        #          ('lickVoltage', 'O'), ('trialNum', 'O'), ('timeROE', 'O'), ('changeRewLoc', 'O'), ('pressedKeys', 'O'), ('world', 'O'), 
        #          ('imageSync', 'O'), ('scalingFACTOR', 'O'), ('wOff', 'O'),
        #          ('catchTrial', 'O'), ('optoTrigger', 'O'), ('settings', 'O')]) 
        velocity = VR[1][0]
        lick = VR[5][0]
        time = VR[4][0]
        gainf = VR[14][0][0]
        rewsize = VR[18][0][0][4][0][0]/gainf
        velocity=-0.013*velocity[1:]/np.diff(time) # make same size
        velocity = np.append(velocity, np.interp(len(velocity)+1, np.arange(len(velocity)),velocity))
        velocitydf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(velocitydf.rolling(10).mean().values)
        rewards = VR[3][0]
        ypos = VR[6][0]/gainf
        trialnum = VR[8][0]
        changerewloc = VR[10][0]
        rews_centered = np.zeros_like(velocity)
        rews_centered[(ypos >= rewloc-5) & (ypos <= rewloc)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(velocity)
        rews_centered[min_iind]=1
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum, rewards)        
        catchtrialsnum = trialnum[VR[16][0].astype(bool)]
        
        # probe trials
        probe = trialnum<3
        # probe = trialnum<str_trials[0] # trials before first successful trial as probes
        com_probe = np.nanmean(ypos[probe][lick.astype(bool)[probe]])-rewloc
        pos_bin, vel_probe = get_behavior_tuning_curve(ypos[probe], velocity[probe], bins=270)
        lick_selectivity = get_lick_selectivity(ypos[probe], trialnum[probe], lick[probe], rewloc, rewsize)
        vel_probe_near_reward = vel_probe.interpolate(method='linear').ffill().bfill().values[int(rewloc)-30:int(rewloc)+10]
        
        # failed trials with opto stim
        # opto
        failtr_opto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%2==1) 
                for xx in trialnum])
        newrewloc = int(newrewloc)
        if sum(failtr_opto)>10:
            pos_bin, vel_failed_opto = get_behavior_tuning_curve(ypos[failtr_opto],
                                    velocity[failtr_opto],bins=270)            
            vel_failed_opto = vel_failed_opto.interpolate(method='linear').ffill().bfill().values[int(newrewloc-(.5*rewsize))-1:int(newrewloc+(.5*rewsize))+2]
            lick_selectivity_fail_opto = get_lick_selectivity(ypos[failtr_opto], 
                        trialnum[failtr_opto], lick[failtr_opto], newrewloc, rewsize, fails_only=True)            
            com_opto = np.nanmean(ypos[failtr_opto][lick.astype(bool)[failtr_opto]])-newrewloc
            lick_selectivity_during_stim = get_lick_selectivity_post_reward(ypos[failtr_opto], 
                        trialnum[failtr_opto], lick[failtr_opto], time[failtr_opto], 
                        newrewloc, rewsize)
        else:
            vel_failed_opto = [np.nan];lick_selectivity_fail_opto=[np.nan]
            lick_selectivity_during_stim = [np.nan]
        # even trials
        failtr_nonopto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%2==0) for xx in trialnum])
        if sum(failtr_nonopto)>10:
            pos_bin, vel_failed_nonopto = get_behavior_tuning_curve(ypos[failtr_nonopto], velocity[failtr_nonopto], 
                        bins=270)
            vel_failed_nonopto = vel_failed_nonopto.interpolate(method='linear').ffill().bfill().values[int(newrewloc-(.5*rewsize))-1:int(newrewloc+(.5*rewsize))+2]
            lick_selectivity_fail_nonopto = get_lick_selectivity(ypos[failtr_nonopto], 
                        trialnum[failtr_nonopto], lick[failtr_nonopto], newrewloc, rewsize, fails_only=True)            
            com_nonopto = np.nanmean(ypos[failtr_nonopto][lick.astype(bool)[failtr_nonopto]])-newrewloc
            lick_selectivity_even = get_lick_selectivity_post_reward(ypos[failtr_nonopto], 
                        trialnum[failtr_nonopto], lick[failtr_nonopto], time[failtr_nonopto], 
                        newrewloc, rewsize) # rewloc = old rew zone
        else:
            vel_failed_nonopto = [np.nan];lick_selectivity_fail_nonopto=[np.nan]
            lick_selectivity_even = [np.nan]
            
        near_reward_per_day.append([lick_selectivity,vel_probe_near_reward,com_probe,vel_failed_opto,
                        lick_selectivity_fail_opto,vel_failed_nonopto,lick_selectivity_fail_nonopto,
                        com_opto,com_nonopto,lick_selectivity_during_stim,lick_selectivity_even]) 
        performance_opto.append(success/(total_trials-len(catchtrialsnum)))   
    optodays_per_an.append(optodays)
    optodays_before_per_an.append(optodays_before)
#%%
df = pd.DataFrame()
df['days'] = list(itertools.chain(*days_all))
df['animal'] = list(itertools.chain(*[[xx]*len(days_all[ii]) for ii,xx in enumerate(animals)]))
df['opto_day_before'] = [xx if xx==True else False for xx in list(itertools.chain(*optodays_before_per_an))]
df['opto'] = [xx if xx==True else False for xx in list(itertools.chain(*optodays_per_an))]
df['lick_selectivity_near_rewardloc_mean'] = [np.nanmean(xx[0]) for xx in near_reward_per_day]
df['velocity_near_rewardloc_mean'] = [np.nanmean(xx[1]) for xx in near_reward_per_day]
df['com_lick_probe'] = [xx[2] for xx in near_reward_per_day]
df['com_lick_odd'] = [xx[7] for xx in near_reward_per_day]
df['com_lick_even'] = [xx[8] for xx in near_reward_per_day]
df['vel_failed_odd'] = [np.nanmean(xx[3]) for xx in near_reward_per_day]
df['lick_selectivity_failed_odd'] = [np.nanmean(xx[4]) for xx in near_reward_per_day]
df['vel_failed_even'] = [np.nanmean(xx[5]) for xx in near_reward_per_day]
df['lick_selectivity_failed_even'] = [np.nanmean(xx[6]) for xx in near_reward_per_day]
df['success_rate'] = performance_opto
df['lick_selectivity_during_stim_odd'] = [np.nanmean(xx[9]) for xx in near_reward_per_day]
df['lick_selectivity_during_stim_even'] = [np.nanmean(xx[10]) for xx in near_reward_per_day]
df['licks_during_failed_trials_stim_odd/even'] = df['lick_selectivity_during_stim_odd']/df['lick_selectivity_during_stim_even']
# df['lick_prob_near_rewardloc_mean'] = [np.quantile(xx[0], .9) for xx in near_reward_per_day]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
dfagg = df#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
# drop 1st row
# df = df.iloc[1:]
# # performance on opto days
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='success_rate', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='success_rate', hue='opto', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)

# memory performance the next day
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto_day_before', y='velocity_near_rewardloc_mean', hue='opto_day_before', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto_day_before', y='velocity_near_rewardloc_mean', hue='opto_day_before', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', hue='opto_day_before', data=dfagg, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=8)
plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
dfagg = df#.groupby(['animal', 'opto']).mean(numeric_only = True)
# odd trials (led on vs. off days)
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='vel_failed_odd', hue='opto', data=dfagg, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto', y='vel_failed_odd', hue='opto', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=8)
# too many nans for this
# plt.figure(figsize=(3,6)) 
# ax = sns.barplot(x='opto', y='lick_selectivity_failed_odd', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='lick_selectivity_failed_odd', hue='opto', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='licks_during_failed_trials_stim_odd/even', hue='opto', 
                data=dfagg, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto', y='licks_during_failed_trials_stim_odd/even',
                hue='opto', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=8)
ax.get_legend().set_visible(False)

plt.figure(figsize=(3,6))
ax = sns.barplot(x='opto', y='com_lick_odd', hue='opto', data=dfagg, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto', y='com_lick_odd', hue='opto', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=8)
#%%
# even trials
plt.figure(figsize=(3,6)) 
ax = sns.barplot(x='opto', y='vel_failed_even', hue='opto', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "k"})
ax = sns.stripplot(x='opto', y='vel_failed_odd', hue='opto', data=df,
                palette={False: "slategray", True: "k"},
                s=8)
ax.get_legend().set_visible(False)
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='lick_selectivity_failed_even', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "k"})
# ax = sns.stripplot(x='opto', y='lick_selectivity_failed_odd', hue='opto', data=df,
#                 palette={False: "slategray", True: "k"},
#                 s=8)
# ax.get_legend().set_visible(False)
# #%%
x1 = df.loc[df.opto_day_before==True, 'lick_selectivity_near_rewardloc_mean'].values
x2 = df.loc[df.opto_day_before==False, 'lick_selectivity_near_rewardloc_mean'].values
scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])

x1 = df.loc[df.opto_day_before==True, 'vel_failed_odd'].values
x2 = df.loc[df.opto_day_before==False, 'vel_failed_odd'].values
scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# %%
