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
plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')
# save to pdf
condrewloc = pd.read_csv(r"Z:\condition_df\chr2_grab.csv", index_col = None)
src = r"Z:\chr2_grabda\e232"
animal = os.path.basename(src)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"{animal}_opto_peri_analysis.pdf"))
# days = [4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
days = [40,41,42,43,44,45,46,47,48,49,50]
range_val = 5; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
optodays = [42,43,44,47,48]
day_date_dff = {}
for day in days: 
    newrewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
    rewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0]
    plndff = []
    # for each plane
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        gainf = params['VR']
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/average
        dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
        fig, ax = plt.subplots()
        ax.plot(dff)
        # temp remove artifacts
        mean = np.mean(dff)
        std = np.std(dff)
        z_scores = np.abs((dff - mean) / std)
        if pln==1:
            artifact_threshold = np.std(z_scores)*2
        else:
            artifact_threshold = np.std(z_scores)*3        
        if day in optodays:
            mean = np.mean(dff)
            std = np.std(dff)
            z_scores = np.abs((dff - mean) / std)
            artifact_mask = z_scores > artifact_threshold
            # Remove artifacts by setting them to NaN
            clean_data = dff.copy()
            clean_data[artifact_mask] = np.nan
            ax.plot(clean_data)
            ax.set_ylim(0.9,1.1)
            dff = clean_data
        # plt.close(fig)
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(3).mean().values)
        rewards = np.hstack(params['solenoid2'])
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
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
        
        # plot behavior
        fig, ax = plt.subplots()
        ax.plot(ybinned)
        ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=12)
        ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', marker = '.', s=2)
        ax.axhline(rewloc, color = 'slategray', linestyle = '--')
        ax.axhline(newrewloc, color = 'k', linestyle = '--')
        ax.set_title(f'Behavior, Day {day}, {layer}')
        fig.tight_layout()
        pdf.savefig(fig)

        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, timedFF[:firstrew], range_val, binsize)
        # peri reward initial probes        
        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=3,ncols=1)#,gridspec_kw={'width_ratios':[4,1]})
        ax = axes[0]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Probe Trials (Centered by prev. rewloc)')
        fig2, axes2 = plt.subplots(nrows=3,ncols=1)#,gridspec_kw={'width_ratios':[4,1]})
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
        
        #TODO: peri reward fails 
        #TODO: peri reward catch trials
        # peri reward failed and catch trials
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum, rewards)
        failtr_bool = np.array([any(yy==xx for yy in ftr_trials) for xx in trialnum])
        failed_trialnum = trialnum[failtr_bool]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(failed_trialnum)
        rews_centered[min_iind]=1
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = eye.perireward_binned_activity(dff[failtr_bool],
            rews_centered, timedFF[failtr_bool], range_val, binsize)
        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        ax = axes[1]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Failed / Catch Trials (Centered by rewloc)')
        ax = axes2[1]
        ax.plot(meanrewdFF)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Failed / Catch Trials (Centered by rewloc)')
        
        # for opto days, plot opto trials only // vs. non opto
        mask = trialnum%2==1
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
        ax.plot(meanrewdFF_opto, color = 'mediumturquoise')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), alpha=0.5, color='mediumturquoise')        
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        fig.suptitle(f'Peri CS/Rew Loc, Day {day}, {layer}')
        fig2.suptitle(f'Mean of Trials, Peri CS/Rew Loc, Day {day}, {layer}')        
        pdf.savefig(fig)
        pdf.savefig(fig2)
        fig.tight_layout()
        fig2.tight_layout()
        plt.close('all')
        plndff.append(meanrewdFF_opto)
    day_date_dff[str(day)] = plndff
pdf.close()

#%%

# average across learning
fig, axes = plt.subplots(4,3,sharex=True, figsize=(20,15))
for k,v in day_date_dff.items():
    day = int(k)
    if day>8:
        learning_day = condrewloc.loc[condrewloc.Day==day, 'learning_date'].values[0]-1    
        if learning_day<3:
            if day in optodays: #if learning_day==2:
                linest ='-'
                colorl='mediumturquoise'
            else:
                linest = '-'
                colorl='k'
            for pln in range(len(v)):
                meandff = v[pln]
                ax = axes[pln, int(learning_day)]
                ax.plot(meandff, color=colorl, label = learning_day, linestyle = linest)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
                ax.set_title(planelut[pln])
                ax.axvline(range_val/binsize, color='slategray', linestyle='--')                    
                    
plt.xlabel('Time from CS (s)')
plt.ylabel('dF/F')
fig.tight_layout()
# %%
