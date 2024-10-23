"""zahra's dopamine hrz analysis
feb 2024
for chr2 experiments
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from dopamine import get_rewzones

# plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"drd_peri_analysis.pdf"))

condrewloc = pd.read_csv(r"Z:\condition_df\drd.csv", index_col = None)
# convert rewlcos to float
# Drop rows with non-numeric values
condrewloc = condrewloc[pd.to_numeric(condrewloc['rewloc'], errors='coerce').notnull()]
condrewloc = condrewloc[pd.to_numeric(condrewloc['prevrewloc'], errors='coerce').notnull()]
# Convert to float
condrewloc[['rewloc', 'prevrewloc']] = condrewloc[['rewloc', 'prevrewloc']].astype(float)
condrewloc[['Day']] = condrewloc[['Day']].astype(int)
src = r"Y:\drd"
animals = ['e256']
# controls for gerardo
days_all =[[20]]

range_val = 6; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
#%%
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        newrewloc = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'rewloc'].values[0])
        rewloc = float(condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'prevrewloc'].values[0])
        plndff = []
        # for each plane
        for path in list(Path(os.path.join(src, animal, str(day))).rglob('*roibyclick_F.mat')):
            params = scipy.io.loadmat(path)
            VR = params['VR'][0][0]; gainf = VR[14][0][0]             
            planenum = os.path.basename(os.path.dirname(path))
            pln = int(planenum[-1])
            params_keys = params.keys()
            
            dff = params['dFF']
            dff = np.array([np.hstack(pd.DataFrame(dffcll).rolling(3).mean().values) for dffcll in dff.T]).T
            rewards = np.hstack(params['solenoid2'])
            if dff.shape[0]<rewards.shape[0]:
                rewards = np.hstack(params['solenoid2'])[:-1]
                trialnum = np.hstack(params['trialnum'])[:-1]
                ybinned = np.hstack(params['ybinned'])[:-1]/gainf
                licks = np.hstack(params['licks'])[:-1]
                timedFF = np.hstack(params['timedFF'])[:-1]
            else:
                rewards = np.hstack(params['solenoid2'])
                trialnum = np.hstack(params['trialnum'])
                ybinned = np.hstack(params['ybinned'])/gainf
                licks = np.hstack(params['licks'])
                timedFF = np.hstack(params['timedFF'])
            # # mask out dark time
            # dff = dff[ybinned>3]
            # rewards = rewards[ybinned>3]
            # trialnum = trialnum[ybinned>3]
            # licks = licks[ybinned>3]
            # timedFF = timedFF[ybinned>3]
            # ybinned = ybinned[ybinned>3]

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
            # per cell 
            meancll = []; alltrialscll = []
            for cll, dfcll in enumerate(dff.T):
                # plot pre-first reward dop activity    
                firstrew = np.where(rewards==1)[0][0]
                rews_centered = np.zeros_like(ybinned[:firstrew])
                rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
                rews_iind = consecutive_stretch(np.where(rews_centered)[0])
                min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
                rews_centered = np.zeros_like(ybinned[:firstrew])
                rews_centered[min_iind]=1
                
                normmeanrewdFF, meanrewdFF, normrewdFF, \
                    rewdFF = eye.perireward_binned_activity(dfcll[:firstrew], 
                            rews_centered, timedFF[:firstrew], range_val, binsize)
                meancll.append(meanrewdFF)
                alltrialscll.append(rewdFF)
                # peri reward initial probes        
                # Find the rows that contain NaNs
                # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
                # Select rows that do not contain any NaNs
                fig, axes = plt.subplots(nrows=3,ncols=2,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
                ax = axes[0,0]
                ax.imshow(rewdFF.T)
                ax.axvline(int(range_val/binsize), color='w',linestyle='--')
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
                ax.set_title('Probe Trials (0=prev. rewloc)')
                # fig2, axes2 = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
                ax = axes[0,1]
                ax.plot(meanrewdFF)   
                xmin,xmax = ax.get_xlim()     
                ax.fill_between(range(0,int(range_val/binsize)*2), 
                        meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                        meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)                
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
            
                # nonopto  
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
                    rewdFF_nonopto = eye.perireward_binned_activity(dfcll[failtr_bool],
                    rews_centered, timedFF[failtr_bool], range_val, binsize)
                # plot
                ax = axes[1,0]
                ax.imshow(rewdFF_nonopto.T)
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
                ax.set_title('Failed Trials (0=rewloc)')
                ax = axes[1,1]
                ax.plot(meanrewdFF_nonopto, color = 'k')   
                xmin,xmax = ax.get_xlim()     
                ax.fill_between(range(0,int(range_val/binsize)*2), 
                        meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                        meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                        alpha=0.5, color='k')
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
                
                # all subsequent rews
                normmeanrewdFF, meanrewdFF, normrewdFF, \
                    rewdFF = eye.perireward_binned_activity(dfcll, rewards, timedFF, 
                                            range_val, binsize)
                # Find the rows that contain NaNs
                # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
                # Select rows that do not contain any NaNs
                ax = axes[2,0]
                ax.imshow(rewdFF.T)
                ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
                ax.set_xticklabels(range(-range_val, range_val+1, 1))
                ax.set_title('Successful Trials (0=CS)')                
                ax.axvline(int(range_val/binsize), color='w',linestyle='--')
                ax = axes[2,1]
                ax.plot(meanrewdFF, color = 'k')   
                xmin,xmax = ax.get_xlim()     
                ax.fill_between(range(0,int(range_val/binsize)*2), 
                        meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                        meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5, color='k')        
                fig2.suptitle(f'Mean of Trials, Animal {animal}, Day {day}, Cell # {cll}')        
                ax.axvline(int(range_val/binsize), color='k',linestyle='--')
                pdf.savefig(fig)
                pdf.savefig(fig2)
                fig.tight_layout()
                fig2.tight_layout()
            
            # plt.close('all')
            plndff.append([meanrewdFF])
        day_date_dff[str(day)] = plndff
#%%