"""zahra's dopamine hrz analysis
feb 2024
for chr2 experiments
"""

import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches

# plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"chr2_ramp_opto_peri_analysis.pdf"))

condrewloc = pd.read_csv(r"Z:\condition_df\chr2_grabda.csv", index_col = None)
src = r"Z:\chr2_grabda"
animals = ['e232']
# days_all = [[13],
#         [55,56]]

days_all = [np.arange(16,25)]
range_val = 8; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}

# optodays = [18, 19, 22, 23, 24]
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]
    for day in days: 
        newrewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
        rewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0]
        plndff = []
        optoday = (condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'Opto'].values[0])
        optoday = optoday==1
        # for each plane
        for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
            params = scipy.io.loadmat(path)
            VR = params['VR'][0][0]; gainf = VR[14][0][0]             
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 7 - roibasemean3/average
            dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
            fig, ax = plt.subplots()
            ax.plot(dff)
            # # temp remove artifacts
            # mean = np.mean(dff)
            # std = np.std(dff)
            # z_scores = np.abs((dff - mean) / std)
            # if pln==1:
            #     artifact_threshold = np.std(z_scores)*2
            # else:
            #     artifact_threshold = np.std(z_scores)*3        
            # if day in optodays:
            #     mean = np.mean(dff)
            #     std = np.std(dff)
            #     z_scores = np.abs((dff - mean) / std)
            #     artifact_mask = z_scores > artifact_threshold
            #     # Remove artifacts by setting them to NaN
            #     clean_data = dff.copy()
            #     clean_data[artifact_mask] = np.nan
            #     ax.plot(clean_data)
            #     ax.set_ylim(0.9,1.1)
            #     dff = clean_data
            # plt.close(fig)
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)
            rewards = np.hstack(params['solenoid2'])
            trialnum = np.hstack(params['trialnum'])
            ybinned = np.hstack(params['ybinned'])/gainf
            licks = np.hstack(params['licks'])
            timedFF = np.hstack(params['timedFF'])
            # mask out dark time
            dff = dff[ybinned>3]
            rewards = rewards[ybinned>3]
            trialnum = trialnum[ybinned>3]
            licks = licks[ybinned>3]
            timedFF = timedFF[ybinned>3]
            ybinned = ybinned[ybinned>3]
            # plot pre-first reward dop activity    
            firstrew = np.where(rewards==1)[0][0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[min_iind]=1
            #%%
            # plot behavior
            fig, ax = plt.subplots()
            ax.plot(ybinned)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=30)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', 
                marker = '.', s=30)
            ax.axhline(rewloc, color = 'slategray', linestyle = '--')
            ax.axhline(newrewloc, color = 'k', linestyle = '--')
            ax.set_title(f'Animal {animal}, Day {day}, {layer}')
            fig.tight_layout()
            pdf.savefig(fig)

            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, timedFF[:firstrew], range_val, binsize)
            # peri reward initial probes        
            # Find the rows that contain NaNs
            # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
            # Select rows that do not contain any NaNs
            clean_arr = rewdFF.T#[~rows_with_nans]    
            fig, axes = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes[0]
            ax.imshow(clean_arr)
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Probe Trials (Centered by prev. rewloc)')
            fig2, axes2 = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes2[0]
            ax.plot(meanrewdFF)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                    meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)                
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Probe Trials (Centered by prev. rewloc)')
            # # center by old rew zone
            # rews_centered = np.zeros_like(ybinned)
            # rews_centered[(ybinned > rewloc-2) & (ybinned < rewloc+2)]=1
            # rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            # min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            # rews_centered = np.zeros_like(ybinned)
            # rews_centered[min_iind]=1
            
            #TODO: peri reward catch trials
            # failed trials
            trialnumvr = VR[8][0]
            catchtrialsnum = trialnumvr[VR[16][0].astype(bool)]
            
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum, rewards)
            # split into opto vs. non opto
            # opto
            failtr_bool = np.array([(xx in ftr_trials) and 
                    (xx not in catchtrialsnum) and (xx%2==1) for xx in trialnum])
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_opto, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(dff[failtr_bool],
                rews_centered, timedFF[failtr_bool], range_val, binsize)
            
            # nonopto  
            failtr_bool = np.array([(xx in ftr_trials) and 
                    (xx not in catchtrialsnum) and (xx%2==0) for xx in trialnum])        
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
                rewdFF_nonopto = eye.perireward_binned_activity(dff[failtr_bool],
                rews_centered, timedFF[failtr_bool], range_val, binsize)
            # plot
            ax = axes[1]
            ax.imshow(np.concatenate([rewdFF_opto.T, rewdFF_nonopto.T]))
            ax.axhline(rewdFF_opto.T.shape[0], color='yellow')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Failed Trials (Centered by rewloc)')
            ax = axes2[1]
            if optoday:
                colorl = 'mediumturquoise'
            else: colorl = 'slategray'
            
            ax.plot(meanrewdFF_opto, color = colorl)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                    meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color=colorl)
            
            ax.plot(meanrewdFF_nonopto, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color='k')

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Failed Trials (Centered by rewloc)')
            
            # for opto days, plot opto trials only // vs. non opto
            mask = (trialnum%2==1)
            # all subsequent rews
            normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(dff[mask], rewards[mask], timedFF[mask], 
                                        range_val, binsize)
            # Find the rows that contain NaNs
            # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
            # Select rows that do not contain any NaNs
            clean_arr_opto = rewdFF_opto.T#[~rows_with_nans]  normmeanrewdFF, meanrewdFF, normrewdFF, \
            normmeanrewdFF, meanrewdFF_nonopto, normrewdFF, rewdFF_nonopto = eye.perireward_binned_activity(dff[~mask], rewards[~mask], timedFF[~mask], 
                                    range_val, binsize)  
            clean_arr_nonopto = rewdFF_nonopto.T
            ax = axes[2]
            ax.imshow(np.concatenate([clean_arr_opto,clean_arr_nonopto]))
            ax.axhline(clean_arr_opto.shape[0], color='yellow')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Successful Trials (Centered by CS)')
            ax = axes2[2]
            ax.plot(meanrewdFF_nonopto, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), alpha=0.5, color='k')        
            if optoday:
                colorl = 'mediumturquoise'
            else: colorl = 'slategray'
            ax.plot(meanrewdFF_opto, color = colorl)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                    meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color=colorl)        
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            fig.suptitle(f'Peri CS/Rew Loc, Animal {animal}, Day {day}, {layer}')
            fig2.suptitle(f'Mean of Trials, Peri CS/Rew Loc, Animal {animal}, Day {day}, {layer}, Opto = {optoday}')        
            pdf.savefig(fig)
            pdf.savefig(fig2)
            fig.tight_layout()
            fig2.tight_layout()
            #%%
            plt.close('all')
            plndff.append([meanrewdFF_opto, meanrewdFF_nonopto])
        day_date_dff[str(day)] = plndff

#%%
plt.rc('font', size=16)          # controls default text sizes
# average across learning
fig, axes = plt.subplots(4,2,sharex=True, figsize=(20,15))
for k,v in day_date_dff.items():
    day = int(k)
    learning_day = condrewloc.loc[condrewloc.Day==day, 'learning_date'].values[0]-1    
    if learning_day<2:
        optoday = (condrewloc.loc[(condrewloc.Day==day), 'Opto'].values[0])
        optoday = optoday==1
        if optoday:
            colorl = 'mediumturquoise'
            linest = '-'
        else:
            linest = '-'
            colorl='slategray'
        for pln in range(len(v)):
            meandff = v[pln][0]
            ax = axes[pln, int(learning_day)-1]
            ax.plot(meandff, color=colorl, label = learning_day, linestyle = linest)
            # plot even trials
            meandff = v[pln][1]
            ax = axes[pln, int(learning_day)-1]
            ax.plot(meandff, color='k', label = learning_day, linestyle = linest)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(planelut[pln])
            ax.axvline(range_val/binsize, color='slategray', linestyle='--')                    
                
plt.xlabel('Time from CS (s)')
plt.ylabel('dF/F')
fig.tight_layout()
fig.suptitle('e232 and e231, learning day 1 vs. 2')
pdf.savefig(fig)
pdf.close()

#%%
# plot mean and sem of opto days vs. control days
opto_condition = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
            'Opto'].values for ii,animal in enumerate(animals)])
opto_condition = np.array([True if xx==1 else False for xx in opto_condition])
day_date_dff_arr = np.array([v for k,v in day_date_dff.items()])
day_date_dff_arr_opto = day_date_dff_arr[opto_condition]
day_date_dff_arr_nonopto = day_date_dff_arr[~opto_condition]

fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex=True,
                        figsize=(20,15))
for pln in range(4):
    for daytype in range(2):
        if daytype==0: # odd
            trialtype = 0 # odd
            ax = axes[pln, daytype]
            ax.plot(np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0), 
                    color='mediumturquoise', label='LEDon')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                        np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                        np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                        alpha=0.5, color='mediumturquoise')
            trialtype = 1 # even
            ax.plot(np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0), 
                    color='k',label='LEDoff')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                        np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                        np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                        alpha=0.5, color='k')
            ax.legend()
            height = 1.025 # ylim
            ax.add_patch(
            patches.Rectangle(
                xy=(0,0),  # point of origin.
                width=50, height=height, linewidth=1,
                color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Plane {planelut[pln]}, 200 mA')
            ax.set_ylim(.96, height)
            if pln==3: ax.set_xlabel('Time from CS (s)')
        else:
            trialtype = 0 # odd
            ax = axes[pln, daytype]
            ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
                    color='cadetblue', label='odd')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                        np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                        np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                        alpha=0.5, color='cadetblue')
            trialtype = 1 # even
            ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
                    color='k', label='even')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                        np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                        np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                        alpha=0.5, color='k')
            ax.add_patch(
            patches.Rectangle(
                xy=(0,0),  # point of origin.
                width=50, height=height, linewidth=1,
                color='silver', alpha=0.2))

            ax.legend()
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Plane {planelut[pln]}, 0 mA')
            ax.set_ylim(.96, height)
            if pln==3: ax.set_xlabel('Time from CS (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('ChR2 per day per + mouse averages')
#%%
# on same plot for vis
fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex=True,
                        figsize=(7,15))
for pln in range(4):
    trialtype = 0 # odd
    ax = axes[pln]
    ax.plot(np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0), 
            color='mediumturquoise', label='LEDon')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                alpha=0.5, color='mediumturquoise')
    trialtype = 1 # even
    ax.plot(np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0), 
            color='k',label='LEDoff')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                np.nanmean(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                alpha=0.5, color='k')
    ax.legend()
    height = 1.025 # ylim
    ax.add_patch(
    patches.Rectangle(
        xy=(0,0),  # point of origin.
        width=50, height=height, linewidth=1,
        color='mediumspringgreen', alpha=0.2))

    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
    ax.set_xticklabels(range(-range_val, range_val+1, 1))
    ax.set_ylim(.96, height)
    if pln==3: ax.set_xlabel('Time from CS (s)')
    trialtype = 0 # odd    
    ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
            color='cadetblue', label='odd, 0mA')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                alpha=0.5, color='cadetblue')
    trialtype = 1 # even
    ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
            color='peru', label='even, 0mA')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
                np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
                alpha=0.5, color='peru')
    ax.add_patch(
    patches.Rectangle(
        xy=(0,0),  # point of origin.
        width=50, height=height, linewidth=1,
        color='silver', alpha=0.2))

    ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
    ax.set_xticklabels(range(-range_val, range_val+1, 1))
    ax.set_title(f'Plane {planelut[pln]}')
    ax.set_ylim(.96, height)
    if pln==3: ax.set_xlabel('Time from CS (s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('ChR2 per day per + mouse averages')