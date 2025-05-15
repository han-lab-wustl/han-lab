"""
quantify licks and velocity during consolidation task
may 2025
VIP excitation
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from projects.memory.behavior import consecutive_stretch, get_behavior_tuning_curve, get_success_failure_trials, get_lick_selectivity, \
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
condrewloc = pd.read_csv(r"C:\Users\Han\Downloads\data_organization - pyr_vip_chrimson.csv", index_col = None)
src = r"X:\vipcre"
animals = ['z14']#,'e242','e243']
days_all = [np.arange(45,49)]#,[29,30],[36,37]]
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper"
# all days to quantify for stim @ reward memory analysis
# days to quantify for stim @ reward memory analysis
# days_all = [[28,29,31,33,34,35,36],
#     [70,71,72,73,74,75,76,77,78]]
# days to quantify for stim @ reward with limited rew eligible
# days_all = [[65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81],
#             [107,108,109,111,112,113,114,115,116,117,118,119,120,121,122,123]]
memory_cond = 'Opto_memory'
opto_cond = 'Opto'

near_reward_per_day = []
optodays_before_per_an = []
optodays_per_an = []
performance_opto = []
for ii,animal in enumerate(animals):
    days = days_all[ii]
    optodays_before = []; optodays = []
    for day in days: 
        newrewloc = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'rewloc'].values[0])
        rewloc = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'prevrewloc'].values[0])
        
        before = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), memory_cond].values[0]
        optodays_before.append(before)    
        optodays.append(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), opto_cond].values[0])
        # get vr file
        path=list(Path(os.path.join(src, animal, str(day))).rglob('*time*.mat'))[0]
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
        newrewloc = newrewloc/gainf
        rewloc = rewloc/gainf
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

        
        # # example plot during learning
        # eps = np.where(changerewloc)[0]
        # rew = (rewards==1).astype(int)
        # mask = np.array([True if xx>10 and xx<28 else False for xx in trialnum])
        # mask = np.zeros_like(trialnum).astype(bool)
        # mask[eps[0]+8500:eps[1]+2700]=True
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(figsize=(9,5))
        # ax.plot(ypos[mask],zorder=1)
        # ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k',
        #         zorder=2)
        # ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan',
        #     zorder=2)
        # # ax.add_patch(
        # # patches.Rectangle(
        # #     xy=(0,newrewloc-10),  # point of origin.
        # #     width=len(ypos[mask]), height=20, linewidth=1, # width is s
        # #     color='slategray', alpha=0.3))
        # ax.add_patch(
        # patches.Rectangle(
        #     xy=(0,changerewloc[eps][0]/gainf-10),  # point of origin.
        #     width=len(ypos[mask]), height=20, linewidth=1, # width is s
        #     color='slategray', alpha=0.3))

        # ax.set_ylim([0,270])
        # ax.spines[['top','right']].set_visible(False)
        # plt.savefig(os.path.join(dst, f'hrz_eg_behavior.svg'),bbox_inches='tight')
        # ax.set_title(f'{day}')
        # plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')

        probe = trialnum<str_trials[0] # trials before first successful trial as probes
        com_probe = np.nanmean(ypos[probe][lick.astype(bool)[probe]])-rewloc
        pos_bin, vel_probe = get_behavior_tuning_curve(ypos[probe], velocity[probe], bins=270)
        lick_selectivity = get_lick_selectivity(ypos[probe], trialnum[probe], lick[probe], rewloc, rewsize,
                        fails_only = True)
        # from vip opto
        window_size = 10
        # also estimate sampling rate
        lick_rate_probes = calculate_lick_rate(lick[probe], 
                    window_size, sampling_rate=31.25*1.5)
        
        vel_probe_near_reward = vel_probe.interpolate(method='linear').ffill().bfill().values[int(rewloc)-30:int(rewloc+(.5*rewsize))]
        # lick selectivity last correct 8 trials
        lasttr = 10
        mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum])
        lick_selectivity_success = get_lick_selectivity(ypos[mask], 
                        trialnum[mask], lick[mask], newrewloc, rewsize,
                        fails_only = False)            
        # failed trials with opto stim
        # opto
        failtr_opto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%2==1) 
                for xx in trialnum])
        newrewloc = int(newrewloc)
        if sum(failtr_opto)>10:
            pos_bin, vel_failed_opto = get_behavior_tuning_curve(ypos[failtr_opto],
                                    velocity[failtr_opto],bins=270)            
            vel_failed_opto = vel_failed_opto.interpolate(method='linear').ffill().bfill().values[int(newrewloc-(.5*rewsize))-1:int(newrewloc+(.5*rewsize))+1]/np.nanmean(vel_failed_opto)
            
            lick_selectivity_fail_opto = get_lick_selectivity(ypos[failtr_opto], 
                        trialnum[failtr_opto], lick[failtr_opto], newrewloc, rewsize, fails_only=True)            
            com_opto = np.nanmean(ypos[failtr_opto][lick.astype(bool)[failtr_opto]])-newrewloc
            lick_selectivity_during_stim = get_lick_selectivity_post_reward(ypos[failtr_opto], 
                        trialnum[failtr_opto], lick[failtr_opto], time[failtr_opto], 
                        newrewloc, rewsize)        
        else:
            vel_failed_opto = [np.nan];lick_selectivity_fail_opto=[np.nan]
            lick_selectivity_during_stim = [np.nan]
            com_opto = [np.nan]        # even trials
        failtr_nonopto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and ~(xx%10==0) for xx in trialnum])
        if sum(failtr_nonopto)>10:
            pos_bin, vel_failed_nonopto = get_behavior_tuning_curve(ypos[failtr_nonopto], velocity[failtr_nonopto], 
                        bins=270)
            vel_failed_nonopto = vel_failed_nonopto.interpolate(method='linear').ffill().bfill().values[int(newrewloc-(.5*rewsize))-1:int(newrewloc+(.5*rewsize))+1]/np.nanmean(vel_failed_nonopto)
            lick_selectivity_fail_nonopto = get_lick_selectivity(ypos[failtr_nonopto], 
                        trialnum[failtr_nonopto], lick[failtr_nonopto], newrewloc, rewsize, fails_only=True)            
            com_nonopto = np.nanmean(ypos[failtr_nonopto][lick.astype(bool)[failtr_nonopto]])-newrewloc
            lick_selectivity_even = get_lick_selectivity_post_reward(ypos[failtr_nonopto], 
                        trialnum[failtr_nonopto], lick[failtr_nonopto], time[failtr_nonopto], 
                        newrewloc, rewsize) # rewloc = old rew zone
        else:
            vel_failed_nonopto = [np.nan];lick_selectivity_fail_nonopto=[np.nan]
            lick_selectivity_even = [np.nan];com_nonopto=[np.nan]
            
        near_reward_per_day.append([lick_selectivity,vel_probe_near_reward,com_probe,vel_failed_opto,
                        lick_selectivity_fail_opto,vel_failed_nonopto,lick_selectivity_fail_nonopto,
                        com_opto,com_nonopto,lick_selectivity_during_stim,lick_selectivity_even,
                        lick_selectivity_success, lick_rate_probes]) 
        performance_opto.append(success/(total_trials-len(catchtrialsnum)))   
    optodays_per_an.append(optodays)
    optodays_before_per_an.append(optodays_before)
#%%
df = pd.DataFrame()
df['days'] = list(itertools.chain(*days_all))
df['animal'] = list(itertools.chain(*[[xx]*len(days_all[ii]) for ii,xx in enumerate(animals)]))
lds = [[condrewloc.loc[((condrewloc.Day==dy)&(condrewloc.Animal==animals[ii])), 'learning_day'].values[0] for dy in dys] 
    for ii,dys in enumerate(days_all)]
df['learning_day'] = list(itertools.chain(*lds))
df['opto_day_before'] = [True if xx==True else False for xx in list(itertools.chain(*optodays_before_per_an))]
df['opto'] = [True if xx==True else False for xx in list(itertools.chain(*optodays_per_an))]
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
df['licks_selectivity_last8trials'] = [np.nanmean(xx[11]) for xx in near_reward_per_day]
df['lickrate_probes'] = [np.nanmean(xx[12]) for xx in near_reward_per_day]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df = df[df.learning_day==1]#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
dfagg = df
# drop 1st row

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
# x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'lick_selectivity_near_rewardloc_mean'].values
# x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'lick_selectivity_near_rewardloc_mean'].values
# t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# print(f'Lick selectivity near reward in memory probes\n\
#     Paired t-test (n=2) p-value: {pvals2:02f}')

#per session vs. per animal plot
# lick_selectivity_near_rewardloc_mean
plt.figure(figsize=(2.2,5))
ax = sns.barplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df, fill=False,errorbar='se',
                palette={False: "slategray", True: "crimson"})
ax = sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df,
                palette={False: "slategray", True: "crimson"},
                s=12, alpha=0.4)
sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "crimson"},
                s=15,ax=ax)

ans = dfagg.index.get_level_values('animal').unique().values
for i in range(len(ans)):
    ax = sns.lineplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
    data=dfagg[dfagg.index.get_level_values('animal')==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
    
ax.get_legend().set_visible(False)
ax.set_ylabel('Memory lick selectivity')
ax.spines[['top','right']].set_visible(False)
plt.title(f'persession: {pvals1:.4f}\n paired t-test: {pvals2:.4f}',fontsize=12)
# plt.savefig(os.path.join(dst, 'memory_lick_selectivity.svg'), bbox_inches='tight')

#%%
# lick selectivity last 8 trials
dfonline = df.groupby(['animal', 'opto']).mean(numeric_only = True)

fig, ax = plt.subplots(figsize=(2.2,5))
sns.barplot(x='opto', y='licks_selectivity_last8trials', hue='opto', data=dfonline, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "crimson"},
                ax=ax)
sns.stripplot(x='opto', y='licks_selectivity_last8trials', hue='opto', data=df,
                palette={False: "slategray", True: "crimson"},
                s=12,ax=ax,alpha=0.4)
sns.stripplot(x='opto', y='licks_selectivity_last8trials', 
                hue='opto', data=dfonline,
                palette={False: "slategray", True: "crimson"},
                s=15,ax=ax)

for i in range(len(ans)):
    ax = sns.lineplot(x='opto', y='licks_selectivity_last8trials', 
    data=dfonline[dfonline.index.get_level_values('animal')==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
    
x1 = dfonline.loc[dfonline.index.get_level_values('opto')==True, 'licks_selectivity_last8trials'].values
x2 = dfonline.loc[dfonline.index.get_level_values('opto')==False, 'licks_selectivity_last8trials'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = df.loc[df.opto_day_before==True, 'licks_selectivity_last8trials'].values
x2 = df.loc[df.opto_day_before==False, 'licks_selectivity_last8trials'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)
ax.set_ylabel('Lick selectivity, last 8 trials')
ax.set_xticklabels(['LED off', 'LED on'])
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(dst, 'online_performance.svg'), bbox_inches='tight')

#%%
# lick rate
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='opto_day_before', y='lickrate_probes', hue='opto_day_before', data=df,
                palette={False: "slategray", True: "crimson"},
                s=12,alpha=0.4,ax=ax)
sns.stripplot(x='opto_day_before', y='lickrate_probes', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "crimson"},
                s=15,ax=ax)
sns.barplot(x='opto_day_before', y='lickrate_probes', 
                hue='opto_day_before', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: "crimson"},
                ax=ax)
ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)

for i in range(len(ans)):
    ax = sns.lineplot(x='opto_day_before', y='lickrate_probes', 
    data=dfagg[dfagg.index.get_level_values('animal')==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.get_legend().set_visible(False)

x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'lickrate_probes'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'lickrate_probes'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = df.loc[df.opto_day_before==True, 'lickrate_probes'].values
x2 = df.loc[df.opto_day_before==False, 'lickrate_probes'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)
ax.set_ylabel('Lick rate, recall probes (licks/s)')
# plt.savefig(os.path.join(dst, 'memory_lick_rate.svg'), bbox_inches='tight')
#%%
# success rate
dfld = df#[df.learning_day==1]#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
plt.figure(figsize=(2.2,5))
sns.barplot(x='opto', y='success_rate', hue='opto', data=dfld, 
                fill=False,
                errorbar='se',
                palette={False: "slategray", True: "crimson"})
ax = sns.stripplot(x='opto', y='success_rate', hue='opto', data=dfld,
                palette={False: "slategray", True: "crimson"},
                s=12, alpha=0.4)
dfagg = dfld.groupby(['animal', 'opto']).mean(numeric_only = True)
ax = sns.stripplot(x='opto', y='success_rate', hue='opto', data=dfagg,
                palette={False: "slategray", True: "crimson"},
                s=15)
ans = dfagg.index.get_level_values('animal').unique().values
ax.set_ylabel('Performance')
ax.set_xticklabels(['LED off', 'LED on'])

# for i in range(len(ans)):
#     ax = sns.lineplot(x='opto', y='success_rate', 
#     data=dfagg[dfagg.index.get_level_values('animal')==ans[i]],
#     errorbar=None, color='dimgray', linewidth=2)

x1 = dfagg.loc[dfagg.index.get_level_values('opto')==True, 'success_rate'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto')==False, 'success_rate'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = dfld.loc[dfld.opto_day_before==True, 'success_rate'].values
x2 = dfld.loc[dfld.opto_day_before==False, 'success_rate'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)

ans = dfagg.index.get_level_values('animal').unique().values
for i in range(len(ans)):
    ax = sns.lineplot(x='opto', y='success_rate', 
    data=dfagg[dfagg.index.get_level_values('animal')==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)

ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)
# plt.savefig(os.path.join(dst, 'performance_success_rate.svg'), bbox_inches='tight')
#%%
# velocity
dfld = df#[df.learning_day==1]#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
plt.figure(figsize=(2.2,5))
sns.barplot(x='opto_day_before', y='velocity_near_rewardloc_mean', hue='opto_day_before', data=dfld, 
                fill=False,errorbar='se',
                palette={False: "slategray", True: "crimson"})
ax = sns.stripplot(x='opto_day_before', y='velocity_near_rewardloc_mean', 
                hue='opto_day_before', data=dfld,
                palette={False: "slategray", True: "crimson"},
                s=12, alpha=0.4)
dfagg = dfld.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
ax = sns.stripplot(x='opto_day_before', y='velocity_near_rewardloc_mean', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: "crimson"},
                s=15)

ans = dfagg.index.get_level_values('animal').unique().values
for i in range(len(ans)):
    ax = sns.lineplot(x='opto_day_before', y='velocity_near_rewardloc_mean', 
    data=dfagg[dfagg.index.get_level_values('animal')==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.set_ylabel('Velocity near rew. loc.')
ax.set_xticklabels(['LED off', 'LED on'])
ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)

x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'velocity_near_rewardloc_mean'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'velocity_near_rewardloc_mean'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = dfld.loc[dfld.opto_day_before==True, 'velocity_near_rewardloc_mean'].values
x2 = dfld.loc[dfld.opto_day_before==False, 'velocity_near_rewardloc_mean'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)

# x1 = df.loc[df.opto_day_before==True, 'vel_failed_odd'].values
# x2 = df.loc[df.opto_day_before==False, 'vel_failed_odd'].values
# scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])