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
animal = 'e242'
src = r"Y:\halo_grabda"
src = os.path.join(src,animal)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"hrz_{os.path.basename(src)}.pdf"))
days = [29]

range_val = 8; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
# figs = True # print out per day figs
day_date_dff = {}
for day in days: 
    plndff = []
    # for each plane
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        VR = params['VR'][0][0][()]
        gainf = VR['scalingFACTOR'][0][0]
        try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
            rewsize = 10

        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/basemean
        if old:
            dff = np.hstack(params['params'][0][0][7][0][0])/np.nanmean(np.hstack(params['params'][0][0][7][0][0]))
            # dff = np.hstack(params['params'][0][0][10])/np.nanmean(np.hstack(params['params'][0][0][10]))
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
        changeRewLoc = np.hstack(params['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
        eps = np.append(eps, len(changeRewLoc))

        # plot pre-first reward dop activity  
        timedFF = np.hstack(params['timedFF'])
        # plot behavior
        if pln==0:
            fig, ax = plt.subplots(figsize=(15,6))            
            ax.plot(ybinned,zorder=1)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], 
                color = 'cyan', s=30,zorder=3)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], 
                color = 'k', marker = '.', s=100,zorder=2)
            
            import matplotlib.patches as patches
            for ep in range(len(eps)-1):
                ax.add_patch(
                patches.Rectangle(
                    xy=(eps[ep],rewlocs[ep]-rewsize/2),  # point of origin.
                    width=len(ybinned[eps[ep]:eps[ep+1]]), height=rewsize, linewidth=1, # width is s
                    color='slategray', alpha=0.3))
        
            ax.set_title(f'Behavior, Day {day}')
            ax.set_ylabel('Position (cm)')
            ax.set_xticks(np.arange(0,len(timedFF)+1,1000))
            ax.set_xticklabels(np.round(np.append(timedFF[::1000]/60,timedFF[-1]/60), 1))
            ax.set_xlabel('Time (minutes)')
            fig.tight_layout()
            pdf.savefig(fig)

        #TODO: peri reward fails 
        #TODO: peri reward catch trials
        # all subsequent rews   
        # only ep3?
        changeRewLoc = np.hstack(params['changeRewLoc'])     
        scalingf=2/3
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))        
        mask = np.arange(0,eps[len(eps)-1])
        # mask = np.arange(0,eps[2])
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = perireward_binned_activity(dff[mask], rewards[mask], 
                timedFF[mask], trialnum[mask],
                range_val, binsize)
        _, meanvel, __, vel = perireward_binned_activity(velocity[mask], rewards[mask], 
            timedFF[mask], trialnum[mask],
            range_val, binsize)
        _, meanlick, __, licktr = perireward_binned_activity(licks[mask], rewards[mask], 
                timedFF[mask], trialnum[mask],
                range_val, binsize)

        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
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
        ax = axes[3]
        ax.plot(meanrewdFF)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)        
        ax = axes[2]
        ax2 = ax.twinx()
        meanvel=np.nanmedian(vel,axis=1)
        ax.plot(meanvel,color='k')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanvel-scipy.stats.sem(vel,axis=1,nan_policy='omit'),
            meanvel+scipy.stats.sem(vel,axis=1,nan_policy='omit'), alpha=0.3,color='k')
        # licks
        ax2.plot(meanlick,color='slategray')   
        xmin,xmax = ax.get_xlim()     
        ax2.fill_between(range(0,int(range_val/binsize)*2), 
            meanlick-scipy.stats.sem(licktr,axis=1,nan_policy='omit'),
            meanlick+scipy.stats.sem(licktr,axis=1,nan_policy='omit'), alpha=0.3,
            color='slategray')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Velocity (cm/s)')
        ax2.set_ylabel('Licks')
        ax.set_xlabel('Time from CS (s)')
        fig.suptitle(f'Peri CS, {animal}, Day {day}, {layer}')        
        fig.tight_layout()
        pdf.savefig(fig)        
        plndff.append(clean_arr)
    day_date_dff[str(day)] = plndff
pdf.close()

#%%
# heatmap across days
pln_mean = np.squeeze(np.array([[np.nanmean(v[i],axis=0) for i in range(4)] for k,v in day_date_dff.items()]))
alltr = np.array([np.concatenate([v[i] for k,v in day_date_dff.items()]) for i in range(4)])
# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,2],sharex=True,figsize=(6,3))
    ax=axes[0]
    cax=ax.imshow(alltr[pln,:,:])    
    ax.set_xlabel('Time from CS (s)')
    ax.set_ylabel('Trials (last 4 days)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='w')
    # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
    ax.set_title(f'Plane {planelut[pln]}')
    fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
    ax=axes[1]
    mf = np.nanmean(alltr[pln,:,:],axis=0)
    ax.plot(mf)    
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    mf-scipy.stats.sem(alltr[pln,:,:],axis=0,nan_policy='omit'),
    mf+scipy.stats.sem(alltr[pln,:,:],axis=0,nan_policy='omit'), alpha=0.3)
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    fig.tight_layout()

#%%
# pln_mean = np.squeeze(np.array([[v[i] for i in range(4)] for k,v in day_date_dff.items()]))
fig, axes = plt.subplots(nrows = 4, sharex=True,sharey=True,
                        figsize=(3,6))
ymin, ymax = .98, 1.01
for pln in range(4):    
    ax = axes[pln]
    ax.plot(np.nanmean(pln_mean[pln,:,:], axis=0), 
            color='slategray')
    ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(pln_mean[pln,:,:],axis=0)-scipy.stats.sem(pln_mean[pln,:,:],
                                axis=0,nan_policy='omit'),
                np.nanmean(pln_mean[pln,:,:],axis=0)+scipy.stats.sem(pln_mean[pln,:,:],
                                axis=0,nan_policy='omit'), 
                alpha=0.5, color='slategray')
    ax.set_ylim(ymin, ymax)
    if pln==3: ax.set_xlabel('Time from CS (s)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.set_title(f'Plane {planelut[pln]}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('per day per + mouse averages')
fig.tight_layout()
pdf.savefig(fig)
pdf.close()