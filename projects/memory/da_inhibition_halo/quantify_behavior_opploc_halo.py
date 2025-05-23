"""
quantify licks and velocity during consolidation task
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf, matplotlib, matplotlib as mpl, matplotlib.pyplot as plt
from projects.memory.behavior import consecutive_stretch, get_behavior_tuning_curve, get_success_failure_trials, get_lick_selectivity, \
    get_lick_selectivity_post_reward, calculate_lick_rate
from projects.memory.dopamine import get_rewzones
from projects.DLC_behavior_classification.eye import perireward_binned_activity
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes

plt.close('all')
# save to pdf
condrewloc = pd.read_csv(r"C:\Users\Han\Downloads\data_organization - halo_grab.csv")
src = r"Y:\halo_grabda"
animals = ['e241','e242','e243']#,'e242','e243']
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
days_all = [[85,86,87,88,89,90,91,92],
        [77,78,79,80,81,82,83,84],
        [86,87,88,89,90,91,92,93]]

planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
numtrailstim = 10 # use 1 for every other trial and 10 for 10 trials on / 1 trial off
near_reward_per_day = []
optodays_before_per_an = []
optodays_per_an = []
performance_opto = []
mem_cond = 'Opto_memory_opploc'
opto_cond = 'Opto_opp_loc'
for ii,animal in enumerate(animals):
    days = days_all[ii]
    optodays_before = []; optodays = []
    for day in days: 
        newrewloc = float(condrewloc.loc[((condrewloc.Day.values==day)&(condrewloc.Animal.values==animal)), 'rewloc'].values[0])
        rewloc = float(condrewloc.loc[((condrewloc.Day.values==day)&(condrewloc.Animal.values==animal)), 'prevrewloc'].values[0])

        optodays_before.append(condrewloc.loc[((condrewloc.Day.values==day)&(condrewloc.Animal.values==animal)), mem_cond].values[0])
        optodays.append(condrewloc.loc[((condrewloc.Day.values==day)&(condrewloc.Animal.values==animal)), opto_cond].values[0])
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
        velocity = VR[1][0]; lick = VR[5][0]; time = VR[4][0]; gainf = VR[14][0][0]
        try:
            rewsize = VR[18][0][0][4][0][0]/gainf
        except:
            rewsize = 20
        velocity=-0.013*velocity[1:]/np.diff(time) # make same size
        velocity = np.append(velocity, np.interp(len(velocity)+1, np.arange(len(velocity)),velocity))
        velocitydf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(velocitydf.rolling(10).mean().values)
        rewards = VR[3][0]; ypos = VR[6][0]/gainf; trialnum = VR[8][0]
        changerewloc = VR[10][0]/gainf
        # get rew zone 
        rz = get_rewzones([newrewloc], 1/gainf)
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum, rewards)        
        catchtrialsnum = trialnum[VR[16][0].astype(bool)]
        # probe trials
        probe = trialnum<3
        # example plot
        # if before==True:
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
        # mask[5000:20000]=True
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(figsize=(6,5))
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
        #     xy=(0,changerewloc[eps][0]-10),  # point of origin.
        #     width=len(ypos[mask]), height=20, linewidth=1, # width is s
        #     color='slategray', alpha=0.3))

        # ax.set_ylim([0,270])
        # ax.spines[['top','right']].set_visible(False)
        # # plt.savefig(os.path.join(dst, f'hrz_eg_behavior.svg'),bbox_inches='tight')
        # ax.set_title(f'{day}')
        # plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')

        
        probe = trialnum<str_trials[0] # trials before first successful trial as probes
        com_probe = np.nanmean(ypos[probe][lick.astype(bool)[probe]])-rewloc
        pos_bin, vel_probe = get_behavior_tuning_curve(ypos[probe], velocity[probe], bins=270)
        lick_selectivity = get_lick_selectivity(ypos[probe], trialnum[probe], lick[probe], rewloc, rewsize,
                        fails_only = True)
        vel_probe_near_reward = vel_probe.interpolate(method='linear').ffill().bfill().values[int(rewloc)-30:int(rewloc+(.5*rewsize))]
        # lick selectivity last 5 trials
        lasttr = 8
        mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum])
        lick_selectivity_success = get_lick_selectivity(ypos[mask], 
                        trialnum[mask], lick[mask], newrewloc, rewsize,
                        fails_only = False)            
        # failed trials with opto stim
        # opto
        failtr_opto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%numtrailstim==0) 
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
            lick_selectivity_during_stim = [np.nan];com_opto=[np.nan]
        # even trials
        failtr_nonopto = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%numtrailstim==1) for xx in trialnum])
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
        
        # velocity and lick rate in opposite stim  loc
        stim_opto = np.array([(xx not in catchtrialsnum) and ~(xx%numtrailstim==0) 
                for xx in trialnum])
        stimzone = ((newrewloc*gainf-((rewsize*gainf)/2)+90)%180)/gainf
        # get velocity 2 s after stimzone
        #todo: make modular
        rews_centered = np.zeros_like(ypos[stim_opto])
        rews_centered[(ypos[stim_opto] >= stimzone-3) & (ypos[stim_opto] <= stimzone+3)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(ypos[stim_opto])
        rews_centered[min_iind]=1
        range_val, binsize = 10, 0.2 #s
        _, meanrewvel, __, ___ = perireward_binned_activity(velocity[stim_opto], rews_centered, 
                time[stim_opto], range_val, binsize)
        # ratios
        vel_stim_opto = np.nanmean(meanrewvel[:int(range_val/binsize)])/np.nanmean(meanrewvel[int(range_val/binsize):])
        _, meanrewlick, __, ___ = perireward_binned_activity(lick[stim_opto], rews_centered, 
                time[stim_opto], range_val, binsize)
        lick_stim_opto = np.nanmean(meanrewlick[:int(range_val/binsize)])/np.nanmean(meanrewlick[int(range_val/binsize):])
        
        near_reward_per_day.append([lick_selectivity, vel_probe_near_reward, com_probe, vel_failed_opto,
                        lick_selectivity_fail_opto, vel_failed_nonopto,
                        lick_selectivity_fail_nonopto,
                        com_opto,com_nonopto,lick_selectivity_during_stim,lick_selectivity_even,
                        lick_selectivity_success, vel_stim_opto, 
                        lick_stim_opto, meanrewvel, meanrewlick, rz]) 
        performance_opto.append(success/(total_trials-len(catchtrialsnum)))   
    optodays_per_an.append(optodays)
    optodays_before_per_an.append(optodays_before)
#%%
df = pd.DataFrame()
df['days'] = list(itertools.chain(*days_all))
df['animal'] = list(itertools.chain(*[[xx]*len(days_all[ii]) for ii,xx in enumerate(animals)]))
df['opto_day_before'] = [True if xx==True else False for xx in list(itertools.chain(*optodays_before_per_an))]
df['opto'] = [True if xx==1 else False for xx in list(itertools.chain(*optodays_per_an))]
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
df['licks_selectivity_last5trials'] = [np.nanmean(xx[11]) for xx in near_reward_per_day]
lick_during_stim = np.array([xx[15] for xx in near_reward_per_day])
vel_during_stim = np.array([xx[14] for xx in near_reward_per_day])
df['rewzones'] = np.array([xx[16] for xx in near_reward_per_day])
#%%
# plot peri stim velocity and licks
# split by condition
rztest = 1
ans = df.animal.unique()
fig, axes = plt.subplots(nrows = 2,ncols = len(ans), sharex=True,figsize = (20,8))
for i,an in enumerate(ans):
    lick_peth_ledoff = np.array(lick_during_stim[(df['opto']==False) & (df['animal']==an) & (df.rewzones==rztest)])
    lick_peth_ledon = np.array(lick_during_stim[(df['opto']==True) & (df['animal']==an) & (df.rewzones==rztest)])    
    color='crimson'
    ax=axes[0,i]
    ax.plot(lick_peth_ledon.T, color=color, linestyle='--')
    ax.plot(lick_peth_ledoff.T, color='k', linestyle='--')
    ax.plot(np.nanmean(lick_peth_ledon,axis=0), color=color, 
            linewidth=3)
    ax.plot(np.nanmean(lick_peth_ledoff,axis=0), color='k', linewidth=3)
    ax.set_title(f'{an}\n Licks during stim, \nrewzone {rztest}')
    height=0.25
    ax.add_patch(
            mpl.patches.Rectangle(
                xy=(range_val/binsize,0),  # point of origin.
                width=2/binsize, height=height, linewidth=1, # width is s
                color='sienna', alpha=0.2))

    vel_peth_ledoff = np.array(vel_during_stim[(df['opto']==False) & (df['animal']==an) & (df.rewzones==rztest)])
    # vel_peth_ledoff = (vel_peth_ledoff.T/np.nanmean(vel_peth_ledoff[:, :int(range_val/binsize)], axis=1)).T
    vel_peth_ledon = np.array(vel_during_stim[(df['opto']==True) & (df['animal']==an) & (df.rewzones==rztest)])
    # vel_peth_ledon = (vel_peth_ledon.T/np.nanmean(vel_peth_ledon[:, :int(range_val/binsize)], axis=1)).T
    ax=axes[1,i]
    ax.plot(vel_peth_ledon.T, color=color, linestyle='--')
    ax.plot(vel_peth_ledoff.T, color='k', linestyle='--')
    ax.plot(np.nanmean(vel_peth_ledon,axis=0), color=color, 
        linewidth=3)
    ax.plot(np.nanmean(vel_peth_ledoff,axis=0), color='k', linewidth=3)
    height=np.max(vel_peth_ledon)+5
    ax.add_patch(
            mpl.patches.Rectangle(
                xy=(range_val/binsize,0),  # point of origin.
                width=2/binsize, height=height, linewidth=1, # width is s
                color='sienna', alpha=0.2))
    ax.set_title(f' Velocity during stim')
    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(np.arange(-range_val, range_val+1, 4))
    ax.set_xlabel('Time from stim (s)')

#%%
# average behavior per animal:
dfagg = df.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)

# df['lick_prob_near_rewardloc_mean'] = [np.quantile(xx[0], .9) for xx in near_reward_per_day]
# df['velocity_near_rewardloc_mean'] = [np.quantile(xx[1], .9) for xx in near_reward_per_day]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
dfagg = df#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
# drop 1st row

# #%%
x1 = df.loc[df.opto_day_before==True, 'lick_selectivity_near_rewardloc_mean'].values
x2 = df.loc[df.opto_day_before==False, 'lick_selectivity_near_rewardloc_mean'].values
t,pvals1 = scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Per session t-test p-value: {pvals1:02f}')

dfagg = df.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
x1 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==True, 'lick_selectivity_near_rewardloc_mean'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto_day_before')==False, 'lick_selectivity_near_rewardloc_mean'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Paired t-test (n=2) p-value: {pvals2:02f}')


# velocity
x1 = df.loc[df.opto_day_before==True, 'velocity_near_rewardloc_mean'].values
x2 = df.loc[df.opto_day_before==False, 'velocity_near_rewardloc_mean'].values
t,pval = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
print(f'Velocity near reward in memory probes\nPer session t-test p-value: {pval:02f}')

#per session vs. per animal plot
#%%

# lick_selectivity_near_rewardloc_mean
plt.figure(figsize=(2.2,5))
ax = sns.barplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df, fill=False,errorbar='se',
                palette={False: "slategray", True: color})
ax = sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=df,
                palette={False: "slategray", True:color},
                s=12, alpha=0.4)
sns.stripplot(x='opto_day_before', y='lick_selectivity_near_rewardloc_mean', 
                hue='opto_day_before', data=dfagg,
                palette={False: "slategray", True: color},
                s=15,ax=ax)
ax.get_legend().set_visible(False)
ax.set_ylabel('Memory lick selectivity')
ax.set_xlabel('LED on day before?')
ax.spines[['top','right']].set_visible(False)
plt.title(f'persession: {pvals1:.4f}\n paired t-test: {pvals2:.4f}',fontsize=12)
fig.tight_layout()
# plt.savefig(os.path.join(dst, 'opploc_memory_lick_selectivity.svg'), bbox_inches='tight')

#%%
# lick selectivity last 8 trials
dfonline = df.groupby(['animal', 'opto']).mean(numeric_only = True)

fig, ax = plt.subplots(figsize=(2.2,5))
sns.barplot(x='opto', y='licks_selectivity_last5trials', hue='opto', data=df, fill=False,
                errorbar='se',
                palette={False: "slategray", True: color},
                ax=ax)
sns.stripplot(x='opto', y='licks_selectivity_last5trials', hue='opto', data=df,
                palette={False: "slategray", True: color},
                s=12,ax=ax,alpha=0.4)
sns.stripplot(x='opto', y='licks_selectivity_last5trials', 
                hue='opto', data=dfonline,
                palette={False: "slategray", True:color},
                s=15,ax=ax)
x1 = dfonline.loc[dfonline.index.get_level_values('opto')==True, 'licks_selectivity_last5trials'].values
x2 = dfonline.loc[dfonline.index.get_level_values('opto')==False, 'licks_selectivity_last5trials'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = df.loc[df.opto_day_before==True, 'licks_selectivity_last5trials'].values
x2 = df.loc[df.opto_day_before==False, 'licks_selectivity_last5trials'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)
ax.set_ylabel('Lick selectivity, last 8 trials')
ax.set_xticklabels(['LED off', 'LED on'],rotation=45)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(dst, 'opploc_online_performance.svg'), bbox_inches='tight')

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
# plt.savefig(os.path.join(dst, 'memory_lick_rate.svg'), bbox_inches='tight')
#%%
# success rate
dfld = df#[df.learning_day==1]#.groupby(['animal', 'opto_day_before']).mean(numeric_only = True)
plt.figure(figsize=(2.2,5))
dfagg = dfld.groupby(['animal', 'opto']).mean(numeric_only = True)

sns.barplot(x='opto', y='success_rate', hue='opto', data=dfld, 
                fill=False,
                errorbar='se',
                palette={False: "slategray", True: color})
ax = sns.stripplot(x='opto', y='success_rate', hue='opto', data=dfld,
                palette={False: "slategray", True: color},
                s=12, alpha=0.4)
dfagg = dfld.groupby(['animal', 'opto']).mean(numeric_only = True)
ax = sns.stripplot(x='opto', y='success_rate', hue='opto', data=dfagg,
                palette={False: "slategray", True: color},
                s=15)
ans = dfagg.index.get_level_values('animal').unique().values
ax.set_ylabel('Performance')
ax.set_xticklabels(['LED off', 'LED on'],rotation=45)

x1 = dfagg.loc[dfagg.index.get_level_values('opto')==True, 'success_rate'].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto')==False, 'success_rate'].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
x1 = dfld.loc[dfld.opto_day_before==True, 'success_rate'].values
x2 = dfld.loc[dfld.opto_day_before==False, 'success_rate'].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)

ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)
# plt.savefig(os.path.join(dst, 'opploc_performance_success_rate.svg'), bbox_inches='tight')
#%%
beh_metric = 'velocity_near_rewardloc_mean'
plt.figure(figsize=(2.2,5))
sns.barplot(x='opto', y=beh_metric, hue='opto', data=dfld, 
                fill=False,
                errorbar='se',
                palette={False: "slategray", True: color})
ax = sns.stripplot(x='opto', y=beh_metric, hue='opto', data=dfld,
                palette={False: "slategray", True: color},
                s=12, alpha=0.4)
dfagg = dfld.groupby(['animal', 'opto']).mean(numeric_only = True)
ax = sns.stripplot(x='opto', y=beh_metric, hue='opto', data=dfagg,
                palette={False: "slategray", True: color},
                s=15)
ans = dfagg.index.get_level_values('animal').unique().values
ax.set_ylabel('Velocity near rew. loc. (cm/s)')
ax.set_xticklabels(['LED off', 'LED on'],rotation=45)

x1 = dfagg.loc[dfagg.index.get_level_values('opto')==True, beh_metric].values
x2 = dfagg.loc[dfagg.index.get_level_values('opto')==False, beh_metric].values
t,pvals2 = scipy.stats.ttest_rel(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
dfagg = dfagg.reset_index()
x1 = dfagg.loc[dfagg.opto==True, beh_metric].values
x2 = dfagg.loc[dfagg.opto==False, beh_metric].values
t,pvals1 = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
ax.set_title(f'persession pval = {pvals1:.4f}\n\
    peranimal paired pval = {pvals2:.4f}',fontsize=12)

ax.get_legend().set_visible(False)
ax.spines[['top','right']].set_visible(False)