"""
quantify licks and velocity during consolidation task
aug 2024
TODO: get first lick during probes
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import consecutive_stretch, get_behavior_tuning_curve, \
get_success_failure_trials, get_lick_selectivity, \
    get_lick_selectivity_post_reward, calculate_lick_rate
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes

plt.close('all')
# save to pdf
condrewloc = pd.read_csv(r"Z:\condition_df\drd.csv", index_col = None)
src = r"Y:\drd"
animals = ['e256', 'e262']

dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# all days to quantify
days_all = [[20,21,22,23],
        [14,15,16,17]]

# days to quantify for stim @ reward memory analysis
# days_all = [[28,29,31,33,34,35,36],
#     [70,71,72,73,74,75,76,77,78]]
# days to quantify for stim @ reward with limited rew eligible
# days_all = [[65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81],
#             [107,108,109,111,112,113,114,115,116,117,118,119,120,121,122,123]]

near_reward_per_day = []
performance_opto = []

for ii,animal in enumerate(animals):
    days = days_all[ii]
    optodays_before = []; optodays = []
    for day in days: 
        newrewloc = float(condrewloc.loc[((condrewloc.Day==day)&\
            (condrewloc.Animal==animal)), 'rewloc'].values[0])
        rewloc = float(condrewloc.loc[((condrewloc.Day==day)&\
            (condrewloc.Animal==animal)), 'prevrewloc'].values[0])
        
                # for each plane
        path=list(Path(os.path.join(src, animal, str(day))).rglob('*roibyclick_F.mat'))[0]
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
        try:
            rewsize = VR[18][0][0][4][0][0]/gainf
        except:
            rewsize = 20
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
        
        # example plot
        # # if before==True:
        #     rewloc = changerewloc[0]
        #     import matplotlib.patches as patches
        #     fig, ax = plt.subplots()
        #     ax.plot(ypos[probe])
        #     ax.scatter(np.where(lick[probe])[0], ypos[np.where(lick[probe])[0]], 
        #     color='k',s=80)
        #     ax.add_patch(
        #     patches.Rectangle(
        #         xy=(0,rewloc-10),  # point of origin.
        #         width=len(ypos[probe]), height=20, linewidth=1, # width is s
        #         color='slategray', alpha=0.3))
        #     ax.set_ylim([0,270])
        #     ax.spines[['top','right']].set_visible(False)
        #     ax.set_title(f'{day}')
        #     plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior_probes.svg'),bbox_inches='tight')

        
        # example plot during learning
        # rew = (rewards==1).astype(int)
        # mask = np.array([True if xx>15 and xx<28 else False for xx in trialnum])
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots()
        # ax.plot(ypos[mask])
        # ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k')
        # ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan')
        # ax.add_patch(
        # patches.Rectangle(
        #     xy=(0,newrewloc-10),  # point of origin.
        #     width=len(ypos[mask]), height=20, linewidth=1, # width is s
        #     color='slategray', alpha=0.3))
        # ax.set_ylim([0,270])
        # ax.spines[['top','right']].set_visible(False)
        # ax.set_title(f'{day}')
        # plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')

        # probe = trialnum<str_trials[0] # trials before first successful trial as probes
        com_probe = np.nanmean(ypos[probe][lick.astype(bool)[probe]])-rewloc
        pos_bin, vel_probe = get_behavior_tuning_curve(ypos[probe], velocity[probe], bins=270)
        lick_selectivity = get_lick_selectivity(ypos[probe], trialnum[probe], lick[probe], rewloc, rewsize,
                        fails_only = True)
        # from vip opto
        window_size = 5
        # also estimate sampling rate
        lick_rate_probes = calculate_lick_rate(lick[probe], 
                    window_size, sampling_rate=31.25*1.5)
        vel_probe_near_reward = vel_probe.interpolate(method='linear').ffill().bfill().values[int(rewloc)-30:int(rewloc+(.5*rewsize))]
        # lick selectivity last 8 trials
        lasttr = 3
        mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum])
        lick_selectivity_success = get_lick_selectivity(ypos[mask], 
                        trialnum[mask], lick[mask], newrewloc, rewsize,
                        fails_only = False)          
            
        near_reward_per_day.append([lick_selectivity,vel_probe_near_reward,com_probe,
                        lick_selectivity_success, lick_rate_probes]) 
        performance_opto.append(success/(total_trials-len(catchtrialsnum)))   
#%%
df = pd.DataFrame()
df['days'] = list(itertools.chain(*days_all))
df['animal'] = list(itertools.chain(*[[xx]*len(days_all[ii]) for ii,xx in enumerate(animals)]))
df['condition'] = ['drd2KO' if xx=='e262' else 'drd2' for xx in df.animal.values]
lds = [[condrewloc.loc[((condrewloc.Day==dy)&(condrewloc.Animal==animals[ii])), 'learning_day'].values[0] for dy in dys] 
    for ii,dys in enumerate(days_all)]
df['learning_day'] = list(itertools.chain(*lds))
df['lick_selectivity_near_rewardloc_mean'] = [np.nanmean(xx[0]) for xx in near_reward_per_day]
df['velocity_near_rewardloc_mean'] = [np.nanmean(xx[1]) for xx in near_reward_per_day]
df['com_lick_probe'] = [xx[2] for xx in near_reward_per_day]
df['success_rate'] = performance_opto
df['licks_selectivity_last8trials'] = [np.nanmean(xx[3]) for xx in near_reward_per_day]
df['lickrate_probes'] = [np.nanmean(xx[4]) for xx in near_reward_per_day]

# df['lick_prob_near_rewardloc_mean'] = [np.quantile(xx[0], .9) for xx in near_reward_per_day]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df = df[df.learning_day==1]#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)

dfagg = df
# drop 1st row
# performance 
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='success_rate', hue='condition', data=df, fill=False,
                errorbar='se')
ax = sns.stripplot(x='condition', y='success_rate', hue='condition', data=df,
            s=12)
ax.spines[['top','right']].set_visible(False)

# ax.get_legend().set_visible(False)

# lick rate
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='lickrate_probes', hue='condition', data=df, fill=False,
                errorbar='se')
ax = sns.stripplot(x='condition', y='lickrate_probes', hue='condition', data=df,
                s=12)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Lick rate (licks/s)')

# lick selectivity on opto days
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='licks_selectivity_last8trials', hue='condition', data=df, fill=False,
                errorbar='se')
ax = sns.stripplot(x='condition', y='licks_selectivity_last8trials', hue='condition', data=df,
                s=12)
ax.spines[['top','right']].set_visible(False)

# memory performance the next day
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='velocity_near_rewardloc_mean', hue='condition', data=df, fill=False,
                errorbar='se')
ax = sns.stripplot(x='condition', y='velocity_near_rewardloc_mean', hue='condition', data=df,
                s=12)
ax.set_ylabel('Velocity near reward zone\nmemory probes')
ax.spines[['top','right']].set_visible(False)


# memory performance the next day
plt.figure(figsize=(3,6))
ax = sns.barplot(x='condition', y='lick_selectivity_near_rewardloc_mean', hue='condition', data=df, fill=False,
                errorbar='se')
ax = sns.stripplot(x='condition', y='lick_selectivity_near_rewardloc_mean', hue='condition', data=df,
                s=12)
ax.set_ylabel('Lick selectivity, memory probes')
ax.spines[['top','right']].set_visible(False)
#%%
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto_day_before', y='com_lick_probe', hue='opto_day_before', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
dfagg = df#.groupby(['animal', 'opto']).mean(numeric_only = True)
# # odd trials (led on vs. off days)
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='vel_failed_odd', hue='opto', data=dfagg, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='vel_failed_odd', hue='opto', data=dfagg,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
# # too many nans for this
# # plt.figure(figsize=(3,6)) 
# ax = sns.barplot(x='opto', y='lick_selectivity_failed_odd', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='lick_selectivity_failed_odd', hue='opto', data=df,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='lick_selectivity_during_stim_odd', hue='opto', 
#                 data=dfagg, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='lick_selectivity_during_stim_odd',
#                 hue='opto', data=dfagg,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
# ax.get_legend().set_visible(False)

# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='com_lick_odd', hue='opto', data=dfagg, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "mediumturquoise"})
# ax = sns.stripplot(x='opto', y='com_lick_odd', hue='opto', data=dfagg,
#                 palette={False: "slategray", True: "mediumturquoise"},
#                 s=8)
#%%
# # even trials
# plt.figure(figsize=(3,6)) 
# ax = sns.barplot(x='opto', y='vel_failed_even', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "k"})
# ax = sns.stripplot(x='opto', y='vel_failed_odd', hue='opto', data=df,
#                 palette={False: "slategray", True: "k"},
#                 s=8)
# ax.get_legend().set_visible(False)
# plt.figure(figsize=(3,6))
# ax = sns.barplot(x='opto', y='lick_selectivity_during_stim_even', hue='opto', data=df, fill=False,
#                 errorbar='se',
#                 palette={False: "slategray", True: "k"})
# ax = sns.stripplot(x='opto', y='lick_selectivity_during_stim_even', hue='opto', data=df,
#                 palette={False: "slategray", True: "k"},
#                 s=8)
# ax.get_legend().set_visible(False)

# #%%
x1 = df.loc[df.opto_day_before==True, 'lick_selectivity_near_rewardloc_mean'].values
x2 = df.loc[df.opto_day_before==False, 'lick_selectivity_near_rewardloc_mean'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Lick selectivity near reward in memory probes\n\
    Per session ranksums p-value: {pvals1:02f}')

x1 = df.loc[(df.opto==1), 'licks_selectivity_last8trials'].values
x2 = df.loc[df.opto==False, 'licks_selectivity_last8trials'].values
t,pval = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Lick selectivity during learning\nPer session t-test p-value: {pval:02f}')

dfagg = df.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'lick_selectivity_near_rewardloc_mean'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'lick_selectivity_near_rewardloc_mean'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Lick selectivity near reward in memory probes\n\
    Paired t-test (n=2) p-value: {pvals2:02f}')

#per session vs. per animal plot
#%%
# lick_selectivity_near_rewardloc_mean
plt.figure(figsize=(2.2,5))
ax = sns.barplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df, fill=False,errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"})
ax = sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df,
                palette={False: "slategray", True: "mediumturquoise"},
                s=12, alpha=0.4)
sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=15,ax=ax)
ax.get_legend().set_visible(False)
ax.set_ylabel('Memory lick selectivity')
ax.spines[['top','right']].set_visible(False)
plt.title(f'persession: {pvals1:.4f}\n paired t-test: {pvals2:.4f}')
plt.savefig(os.path.join(dst, 'memory_lick_selectivity.svg'), bbox_inches='tight')

#%%
# lick selectivity last 8 trials
dfonline = df.groupby(['animal', 'opto']).mean(numeric_only = True)

fig, ax = plt.subplots(figsize=(2.2,5))
sns.barplot(x='opto', y='licks_selectivity_last8trials', hue='opto', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "mediumturquoise"},
                ax=ax)
sns.stripplot(x='opto', y='licks_selectivity_last8trials', hue='opto', data=df,
                palette={False: "slategray", True: "mediumturquoise"},
                s=12,ax=ax,alpha=0.4)
sns.stripplot(x='opto', y='licks_selectivity_last8trials', 
                hue='opto', data=dfonline,
                palette={False: "slategray", True: "mediumturquoise"},
                s=15,ax=ax)
x1 = dfonline.loc[dfonline.index.get_level_values('opto')==True, 'licks_selectivity_last8trials'].values
x2 = dfonline.loc[dfonline.index.get_level_values('opto')==False, 'licks_selectivity_last8trials'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = df.loc[df.opto_day_before==True, 'licks_selectivity_last8trials'].values
x2 = df.loc[df.opto_day_before==False, 'licks_selectivity_last8trials'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}')
ax.set_ylabel('Lick selectivity, last 8 trials')
ax.set_xticklabels(['LED off', 'LED on'])
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
plt.savefig(os.path.join(dst, 'online_performance.svg'), bbox_inches='tight')

#%%
# lick rate
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='opto_day_before', y='lickrate_probes', hue='opto_day_before', data=df,
                palette={False: "slategray", True: "mediumturquoise"},
                s=12,alpha=0.4,ax=ax)
sns.stripplot(x='opto_day_before', y='lickrate_probes', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "mediumturquoise"},
                s=15,ax=ax)
sns.barplot(x='opto_day_before', y='lickrate_probes', 
                hue='opto_day_before', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", 
                True: "mediumturquoise"},
                ax=ax)
ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)
x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'lickrate_probes'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'lickrate_probes'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = df.loc[df.opto_day_before==True, 'lickrate_probes'].values
x2 = df.loc[df.opto_day_before==False, 'lickrate_probes'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}')
ax.set_ylabel('Lick rate, recall probes (licks/s)')
plt.savefig(os.path.join(dst, 'memory_lick_rate.svg'), bbox_inches='tight')
#%%

# velocity
x1 = df.loc[df.opto_day_before==True, 'velocity_near_rewardloc_mean'].values
x2 = df.loc[df.opto_day_before==False, 'velocity_near_rewardloc_mean'].values
t,pval = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Velocity near reward in memory probes\nPer session t-test p-value: {pval:02f}')


x1 = df.loc[df.opto_day_before==True, 'licks_selectivity_last8trials'].values
x2 = df.loc[df.opto_day_before==False, 'licks_selectivity_last8trials'].values
t,pval = scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Lick selectivity last 8 trials\nPer session t-test p-value: {pval:02f}')

# x1 = df.loc[df.opto_day_before==True, 'vel_failed_odd'].values
# x2 = df.loc[df.opto_day_before==False, 'vel_failed_odd'].values
# scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# %%

