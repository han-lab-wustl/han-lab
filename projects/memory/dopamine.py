"""functions for dopamine 1 rewloc analysis
july 2024
"""
import os, scipy, numpy as np, pandas as pd, sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.rewardcell import perireward_binned_activity
from projects.opto.behavior.behavior import get_success_failure_trials


def extract_vars(src, animal, day, condrewloc, opto_cond, dst, 
        pdf, rolling_win=3, planes=4,range_val = 5, binsize=0.2):
    # set vars
    print(f'*******Animal: {animal}, Day: {day}*******\n')
    # day=str(day)
    planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
    newrewloc_ = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'rewloc'].values[0])
    numtrialsstim=condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'numtrialsstim'].values[0]
    if ~np.isnan(numtrialsstim): numtrialsstim=int(numtrialsstim)
    rewloc_ = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'prevrewloc'].values[0])
    plndff = []
    optoday = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), opto_cond].values[0]
    # hack
    try:
        optoday=int(optoday)
        optoday = optoday==1
    except Exception as e:
        print(e)
    # for each plane
    for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)        
        if len(stims['stims']>0): stims = np.hstack(stims['stims']) # nan out stims
        else: stims = np.zeros_like(params['forwardvelALL'][0])
        VR = params['VR'][0][0]; gainf = VR[14][0][0]      
        # adjust for gain
        rewloc = rewloc_/gainf; newrewloc = newrewloc_/gainf
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/average
        dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
        # nan out stims
        dff[stims[pln::4].astype(bool)] = np.nan
        
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(rolling_win).mean().values)
        rewards = np.hstack(params['solenoid2'])
        if dff.shape[0]<rewards.shape[0]:
            rewards = np.hstack(params['solenoid2'])[:-1]
            trialnum = np.hstack(params['trialnum'])[:-1]
            ybinned = np.hstack(params['ybinned'])[:-1]/gainf
            licks = np.hstack(params['licks'])[:-1]
            timedFF = np.hstack(params['timedFF'])[:-1]
            forwardvel = np.hstack(params['forwardvel'])[:-1]
        else:
            rewards = np.hstack(params['solenoid2'])
            trialnum = np.hstack(params['trialnum'])
            ybinned = np.hstack(params['ybinned'])/gainf
            licks = np.hstack(params['licks'])
            timedFF = np.hstack(params['timedFF'])
            forwardvel = np.hstack(params['forwardvel'])

        # plot behavior
        if pln==0:
            fig, ax = plt.subplots()
            ax.plot(ybinned)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=30)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', 
                marker = '.', s=30)
            ax.axhline(rewloc, color = 'slategray', linestyle = '--')
            ax.axhline(newrewloc, color = 'k', linestyle = '--')
            ax.set_title(f'Animal {animal}, Day {day}')
            fig.tight_layout()
            pdf.savefig(fig)
            ###################### also get velocity ######################
            # initial probes
            
            firstrew = np.where(rewards==1)[0][0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[min_iind]=1
        
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF = perireward_binned_activity(forwardvel[:firstrew], 
                        rews_centered.astype(bool), timedFF[:firstrew], trialnum[:firstrew],
                        range_val, binsize)

            fig, axes = plt.subplots(nrows=4,ncols=2,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes[0,0]
            ax.imshow(rewdFF.T, cmap="Greys_r")
            ax.axvline(int(range_val/binsize), color='w',linestyle='--')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Velocity, Probe Trials (0=prev. rewloc)')
            # fig2, axes2 = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes[0,1]
            ax.plot(meanrewdFF,color='gray')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                    meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), 
                    alpha=0.5,color='gray')                
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Mean +/- SEM of trials')
            ax.axvline(int(range_val/binsize), color='k',linestyle='--')
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
        
            # fails only  
            failtr_bool = np.array([(xx in ftr_trials) and 
                    (xx not in catchtrialsnum) for xx in trialnum])        
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
                rewdFF_nonopto = perireward_binned_activity(forwardvel[failtr_bool],
                rews_centered.astype(bool), timedFF[failtr_bool], trialnum[failtr_bool],
                range_val, binsize)
            # plot
            ax = axes[1,0]
            ax.imshow(rewdFF_nonopto.T, cmap="Greys_r")
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Failed Trials (0=rewloc)')
            ax = axes[1,1]
            ax.plot(meanrewdFF_nonopto,color='gray')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                    alpha=0.5,color='gray')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.axvline(int(range_val/binsize), color='k',linestyle='--')
            # catch trials only  
            failtr_bool = np.array([(xx in catchtrialsnum) for xx in trialnum])        
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
                rewdFF_nonopto = perireward_binned_activity(forwardvel[failtr_bool],
                rews_centered.astype(bool), timedFF[failtr_bool], trialnum[failtr_bool],
                range_val, binsize)
            # plot
            ax = axes[2,0]
            ax.imshow(rewdFF_nonopto.T, cmap="Greys_r")
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.axvline(int(range_val/binsize), color='w',linestyle='--')
            ax.set_title('Catch Trials (0=rewloc)')
            ax = axes[2,1]
            ax.plot(meanrewdFF_nonopto,color='gray')   
            ax.axvline(int(range_val/binsize), color='k',linestyle='--')
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                    alpha=0.5,color='gray')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            
            # all subsequent rews
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF = perireward_binned_activity(forwardvel, rewards.astype(bool), timedFF, 
                            trialnum, range_val, binsize)
            # Find the rows that contain NaNs
            # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
            # Select rows that do not contain any NaNs
            ax = axes[3,0]
            ax.imshow(rewdFF.T, cmap="Greys_r")
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Successful Trials (0=CS)')                
            ax.axvline(int(range_val/binsize), color='w',linestyle='--')
            ax = axes[3,1]
            ax.plot(meanrewdFF,color='gray')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                    meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), 
                    alpha=0.5, color='gray')        
            fig.suptitle(f'Velocity, Animal {animal}, Day {day}')        
            ax.axvline(int(range_val/binsize),color='gray',linestyle='--')
            fig.tight_layout()
            pdf.savefig(fig)
        ################################### dff ###################################
        # initial probes            
        firstrew = np.where(rewards==1)[0][0]
        rews_centered = np.zeros_like(ybinned[:firstrew])
        rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(ybinned[:firstrew])
        rews_centered[min_iind]=1
        
        # plot pre-first reward dop activity    
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = perireward_binned_activity(dff[:firstrew], rews_centered.astype(bool), timedFF[:firstrew], 
                    trialnum[:firstrew], range_val, binsize)
        # peri reward initial probes        
        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=4,ncols=2,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
        ax = axes[0,0]
        ax.imshow(clean_arr)
        ax.axvline(int(range_val/binsize), color='w',linestyle='--')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('$\Delta$ F/F, Probe Trials (0=prev. rewloc)')
        ax = axes[0,1]
        ax.plot(meanrewdFF)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)                
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
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
                (xx not in catchtrialsnum) and (xx%numtrialsstim==0) for xx in trialnum])
        failed_trialnum = trialnum[failtr_bool]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[min_iind]=1
        normmeanrewdFF_opto, meanrewdFF_opto, normrewdFF, \
            rewdFF_opto = perireward_binned_activity(dff[failtr_bool],
            rews_centered.astype(bool), timedFF[failtr_bool], trialnum[failtr_bool],
            range_val, binsize)
        
        # nonopto  
        failtr_bool = np.array([(xx in ftr_trials) and 
                (xx not in catchtrialsnum) and (xx%numtrialsstim==1) for xx in trialnum])        
        failed_trialnum = trialnum[failtr_bool]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[min_iind]=1
        normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
            rewdFF_nonopto = perireward_binned_activity(dff[failtr_bool],
            rews_centered.astype(bool), timedFF[failtr_bool], trialnum[failtr_bool],
            range_val, binsize)
        # plot
        ax = axes[1,0]
        ax.imshow(np.concatenate([rewdFF_opto.T, rewdFF_nonopto.T]))
        ax.axhline(rewdFF_opto.T.shape[0], color='yellow')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Failed Trials (Centered by rewloc)')
        ax = axes[1,1]
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
        # catch trials only  
        failtr_bool = np.array([(xx in catchtrialsnum) for xx in trialnum])        
        failed_trialnum = trialnum[failtr_bool]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[min_iind]=1
        normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
            rewdFF_nonopto = perireward_binned_activity(dff[failtr_bool],
            rews_centered.astype(bool), timedFF[failtr_bool], trialnum[failtr_bool],
            range_val, binsize)
        # plot
        ax = axes[2,0]
        ax.imshow(rewdFF_nonopto.T)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize), color='w',linestyle='--')
        ax.set_title('Catch Trials (0=rewloc)')
        ax = axes[2,1]
        ax.plot(meanrewdFF_nonopto, color = 'k')   
        ax.axvline(int(range_val/binsize), color='k',linestyle='--')
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                alpha=0.5, color='k')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))

        # for opto days, plot opto trials only // vs. non opto
        mask = ~(trialnum%numtrialsstim==0)
        # all subsequent rews
        normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
            rewdFF_opto = perireward_binned_activity(dff[mask], 
                rewards[mask].astype(bool), timedFF[mask], 
                            trialnum[mask], range_val, binsize)
        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr_opto = rewdFF_opto.T#[~rows_with_nans]  normmeanrewdFF, meanrewdFF, normrewdFF, \
        normmeanrewdFF, meanrewdFF_nonopto, normrewdFF, rewdFF_nonopto = perireward_binned_activity(dff[~mask], rewards[~mask].astype(bool), timedFF[~mask], 
            trialnum[~mask],range_val, binsize)  
        clean_arr_nonopto = rewdFF_nonopto.T
        ax = axes[3,0]
        ax.imshow(np.concatenate([clean_arr_opto,clean_arr_nonopto]))
        ax.axhline(clean_arr_opto.shape[0], color='yellow')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Successful Trials (Centered by CS)')
        ax = axes[3,1]
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
        pdf.savefig(fig)
        fig.tight_layout()
        
        plt.close('all')
        plndff.append([meanrewdFF_opto, meanrewdFF_nonopto, 
                rewdFF_opto, rewdFF_nonopto])
    return plndff

def get_rewzones(rewlocs, gainf):
    # note that gainf is multiplied here
    # gainf = 1/scalingf
    # Initialize the reward zone numbers array with zeros
    rewzonenum = np.zeros(len(rewlocs))
    
    # Iterate over each reward location to determine its reward zone
    for kk, loc in enumerate(rewlocs):
        if loc <= 86 * gainf:
            rewzonenum[kk] = 1  # Reward zone 1
        elif 101 * gainf <= loc <= 120 * gainf:
            rewzonenum[kk] = 2  # Reward zone 2
        elif loc >= 135 * gainf:
            rewzonenum[kk] = 3  # Reward zone 3
            
    return rewzonenum
