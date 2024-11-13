"""zahra's dopamine hrz analysis
march 2024
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

plt.rcParams["font.family"] = "Arial"

#%%
plt.close('all')
# save to pdf
src = r"Y:\halo_grabda\e243"
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"hrz_{os.path.basename(src)}.pdf"))
days = [18]
range_val = 4 ; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
day_date_dff = {}
for day in days: 
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
        # dff is in row 7 - roibasemean3/basemean
        if old:
            dff = np.hstack(params['params'][0][0][7][0][0])/np.nanmean(np.hstack(params['params'][0][0][7][0][0]))
        else:
            dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))
        
        # plt.close(fig)
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(5).mean().values)
        rewards = np.hstack(params['solenoid2'])
        velocity = np.hstack(params['forwardvel'])
        veldf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(veldf.rolling(5).mean().values)
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        # plot pre-first reward dop activity  
        timedFF = np.hstack(params['timedFF'])
        # plot behavior
        if pln==0:
            fig, ax = plt.subplots()            
            ax.plot(ybinned)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=12)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', marker = '.', s=2)
            ax.set_title(f'Behavior, Day {day}')
            ax.set_ylabel('Position (cm)')
            fig.tight_layout()
            pdf.savefig(fig)

        #TODO: peri reward fails 
        #TODO: peri reward catch trials
        # all subsequent rews   
        # only ep3?
        changeRewLoc = np.hstack(params['changeRewLoc'])     
        scalingf=2/3
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        mask = np.arange(eps[len(eps)-2],eps[len(eps)-1])
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = perireward_binned_activity(dff[mask], rewards[mask], 
                                    timedFF[mask], trialnum[mask],
                                    range_val, binsize)
        _, meanvel, __, \
            vel = perireward_binned_activity(velocity[mask], rewards[mask], 
                                    timedFF[mask], trialnum[mask],
                                    range_val, binsize)

        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(3,5))
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        # ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Correct Trials')
        ax.axvline(int(range_val/binsize),linestyle='--',color='w')
        ax.set_ylabel('Trial #')
        ax = axes[2]
        ax.plot(meanrewdFF)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)        
        ax = axes[3]
        ax.plot(meanvel,color='k')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanvel-scipy.stats.sem(vel,axis=1,nan_policy='omit'),
                meanvel+scipy.stats.sem(vel,axis=1,nan_policy='omit'), alpha=0.3,color='k')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Velocity (cm/s)')
        ax.set_xlabel('Time from CS (s)')
        fig.suptitle(f'Peri CS, Day {day}, {layer}')        
        fig.tight_layout()
        pdf.savefig(fig)        
        plndff.append(meanrewdFF)
    day_date_dff[str(day)] = plndff
pdf.close()

#%%

pln_mean = np.array([[v[i] for i in range(4)] for k,v in day_date_dff.items()])
fig, axes = plt.subplots(nrows = 4, sharex=True,sharey=True,
                        figsize=(2.5,5))
ymin, ymax = .99, 1.01
for pln in range(4):    
    ax = axes[pln]
    ax.plot(np.nanmean(pln_mean[:,pln,:], axis=0), 
            color='slategray')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(pln_mean[:,pln,:],axis=0)-scipy.stats.sem(pln_mean[:,pln,:],
                                axis=0,nan_policy='omit'),
                np.nanmean(pln_mean[:,pln,:],axis=0)+scipy.stats.sem(pln_mean[:,pln,:],
                                axis=0,nan_policy='omit'), 
                alpha=0.5, color='slategray')
    ax.set_ylim(ymin, ymax)
    if pln==3: ax.set_xlabel('Time from CS (s)')
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
    ax.set_xticklabels(range(-range_val, range_val+1, 1))
    ax.set_title(f'Plane {planelut[pln]}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('per day per + mouse averages')
fig.tight_layout()
pdf.savefig(fig)
pdf.close()