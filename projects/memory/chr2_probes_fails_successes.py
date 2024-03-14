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

#%%
plt.close('all')
# save to pdf

src = r"Z:\chr2_grabda\e232"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(os.path.dirname(src),"peri_analysis.pdf"))
days = [11,12,13,14,15,16,17,20]
rewloc = 123*1.5
newrewloc = rewloc
range_val = 10; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
for day in days: 
    if day>8: newrewloc = 65*1.5
    if day>11: rewloc = 65*1.5 # memory rew loc
    if day>14: newrewloc = 91*1.5 
    if day>15: rewloc = 91*1.5 # memory rew loc
    if day>17: newrewloc = 155*1.5 
    if day>18: rewloc = 155*1.5 # memory rew loc
    if day>19: newrewloc = 70*1.5 # memory rew loc
    # for each plane
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        gainf = params['VR']
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/basemean
        dff = np.hstack(params['params'][0][0][6][0][0])/np.hstack(params['params'][0][0][9])
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(2).mean().values)
        rewards = np.hstack(params['solenoid2'])
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        # plot pre-first reward dop activity    
        firstrew = np.where(rewards==1)[0][0]
        rews_centered = np.zeros_like(ybinned[:firstrew])
        rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered = np.zeros_like(ybinned[:firstrew])
        rews_centered[min_iind]=1
        timedFF = np.hstack(params['timedFF'])
        # plot behavior
        fig, ax = plt.subplots()
        ax.plot(ybinned)
        ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=12)
        ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', marker = '.', s=2)
        ax.axhline(rewloc, color = 'slategray', linestyle = '--')
        ax.axhline(newrewloc, color = 'k', linestyle = '--')
        ax.set_title(f'Behavior, Day {day}, {layer}')
        pdf.savefig(fig)

        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, timedFF[:firstrew], range_val, binsize)
        # peri reward initial probes
        fig, axes = plt.subplots(nrows=3,ncols=1)#,gridspec_kw={'width_ratios':[4,1]})
        ax = axes[0]
        ax.imshow(normrewdFF)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Probe Trials (Centered by prev. rewloc)')
        fig2, axes2 = plt.subplots(nrows=3,ncols=1)#,gridspec_kw={'width_ratios':[4,1]})
        ax = axes2[0]
        ax.plot(meanrewdFF)
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
        # peri reward failed + catch trials
        ax = axes[1]
        ax.imshow(normrewdFF)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Failed / Catch Trials (Centered by rewloc)')
        ax = axes2[1]
        ax.plot(meanrewdFF)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Failed / Catch Trials (Centered by rewloc)')
        

        # all subsequent rews
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = eye.perireward_binned_activity(dff, rewards, timedFF, range_val, binsize)
        ax = axes[2]
        ax.imshow(normrewdFF)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Successful Trials (Centered by CS)')
        ax = axes2[2]
        ax.plot(meanrewdFF)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        fig.suptitle(f'Peri CS/Rew Loc, Day {day}, {layer}')
        fig2.suptitle(f'Mean of Trials, Peri CS/Rew Loc, Day {day}, {layer}')

        pdf.savefig(fig)
        pdf.savefig(fig2)
        plt.close('all')
pdf.close()


# %%
