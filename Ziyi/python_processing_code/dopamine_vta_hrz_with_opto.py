"""zahra's dopamine hrz analysis
march 2024
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib

from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
from Ziyi.python_processing_code.functions.function_code import find_start_points
from projects.DLC_behavior_classification import eye
plt.rcParams["font.family"] = "Arial"

#%%
plt.close('all')
# save to pdf
animal = ''
src = r"E:\Ziyi\Data\VTA_mice\hrz\E277"
src = os.path.join(src,animal)
dst = r"E:\Ziyi"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"hrz_{os.path.basename(src)}.pdf"))
days = np.arange(20,21)
#days = [16]
range_val=10; binsize=0.2
numtrialsstim=2 # every 10 trials stim w 1 trial off
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
# figs = True # print out per day figs
day_date_dff = {}
for day in days: 
    plndff = []
    # for each plane
    stimspth = list(Path(os.path.join(src, str(day))).rglob('*000*.mat'))[0]
    #stimspth = Path(os.path.join(src, str(day))).rglob('*000*.mat')
    stims = scipy.io.loadmat(stimspth)
    stims = np.hstack(stims['stims']) # nan out stims
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
        velocity = np.hstack(veldf.rolling(10).mean().values)
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        changeRewLoc = np.hstack(params['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
        eps = np.append(eps, len(changeRewLoc))
        
        #eps_opto_mask = np.zeros(len(stims), dtype=bool)


        changeRewLocALL = np.hstack(params['changeRewLocALL'])
        epsALL = np.where(changeRewLocALL>0)[0];rewlocs = changeRewLocALL[epsALL]/gainf
        epsALL = np.append(epsALL, len(changeRewLocALL))

        '''
        # Precompute which epsALL segments have stim
        # Initialize eps_mask
        eps_opto_mask = np.zeros(eps[-1], dtype=bool)

        # Precompute which epsALL segments have stim
        epsALL_has_stim = np.array([
            np.any(stims[epsALL[j]:epsALL[j+1]] == 1)
            for j in range(len(epsALL) - 1)
        ])

        # For each epsALL segment that has stim, mark the first eps segment inside it
        for j in range(len(epsALL) - 1):
            if epsALL_has_stim[j]:
                # Find first eps segment inside epsALL[j]:epsALL[j+1]
                eps_starts_in_segment = np.where(
                    (eps[:-1] >= epsALL[j]) & (eps[1:] <= epsALL[j+1])
                )[0]
                if len(eps_starts_in_segment) > 0:
                    first_i = eps_starts_in_segment[0]
                    eps_opto_mask[eps[first_i]:eps[first_i+1]] = True
        '''
        # Initialize eps_opto_mask
        eps_opto_mask = np.zeros(eps[-1], dtype=bool)

        # Precompute which epsALL segments have stim
        epsALL_has_stim = np.array([
            np.any(stims[epsALL[j]:epsALL[j+1]] == 1)
            for j in range(len(epsALL) - 1)
        ])

        # For each epsALL segment that has stim, mark the corresponding eps segment
        for j in range(len(epsALL_has_stim)):
            if epsALL_has_stim[j] and j < len(eps) - 1:
                eps_opto_mask[eps[j]:eps[j+1]] = True


        eps_nonopto_mask = ~eps_opto_mask

        #for ii in range(len(eps)-1):
        #    epoch_mask = np.arange(eps[ii],eps[ii+1])
        
        newrewloc = rewlocs[epsALL_has_stim]
        
        # split into opto vs. non opto
        # opto
        #trialnumvr = VR[8][0]
        #catchtrialsnum = trialnumvr[VR[16][0].astype(bool)]
        ybinned_opto_eps = ybinned[eps_opto_mask]
        trialnum_opto_eps = trialnum[eps_opto_mask]
        rewards_opto_eps = rewards[eps_opto_mask]
        dff_opto_eps = dff[eps_opto_mask]
        timedFF_opto_eps = timedFF[eps_opto_mask]
        velocity_opto_eps = velocity[eps_opto_mask]
        licks_opto_eps = licks[eps_opto_mask]

        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum_opto_eps, rewards_opto_eps)

        # opto trials in opto epoch
        str_bool_opto = np.array([(xx in str_trials) and 
                 (xx%numtrialsstim==1) for xx in trialnum_opto_eps])
        str_trialnum = trialnum_opto_eps[str_bool_opto]
        rews_centered_opto = np.zeros_like(str_trialnum)
        rews_centered_opto[(ybinned_opto_eps[str_bool_opto] >= newrewloc-5) & (ybinned_opto_eps[str_bool_opto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_opto)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_opto = np.zeros_like(str_trialnum)
        rews_centered_opto[min_iind]=1
        normmeanrewdFF_opto, meanrewdFF_opto, normrewdFF_opto, \
            rewdFF_opto = eye.perireward_binned_activity(dff_opto_eps[str_bool_opto],
            rews_centered_opto, timedFF_opto_eps[str_bool_opto], range_val, binsize)

        _, meanvel_opto, __, vel_opto = perireward_binned_activity(velocity_opto_eps[str_bool_opto],  rewards_opto_eps[str_bool_opto], 
            timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto],
            range_val, binsize)
        _, meanlick_opto, __, licktr_opto = perireward_binned_activity(velocity_opto_eps[str_bool_opto],  rewards_opto_eps[str_bool_opto], 
                timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto],
                range_val, binsize)
           
        
    

        # nonopto trial in the opto epoch
        str_bool_nonopto = np.array([(xx in str_trials) and 
                 (xx%numtrialsstim==0) for xx in trialnum_opto_eps])
        str_trialnum = trialnum_opto_eps[str_bool_nonopto]
        rews_centered_nonopto = np.zeros_like(str_trialnum)
        rews_centered_nonopto[(ybinned_opto_eps[str_bool_nonopto] >= newrewloc-5) & (ybinned_opto_eps[str_bool_nonopto] <= newrewloc+5)]=1
        rews_iind = consecutive_stretch(np.where(rews_centered_nonopto)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        rews_centered_nonopto = np.zeros_like(str_trialnum)
        rews_centered_nonopto[min_iind]=1
        normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF_nonopto, \
            rewdFF_nonopto = eye.perireward_binned_activity(dff_opto_eps[str_bool_nonopto],
            rews_centered_nonopto, timedFF_opto_eps[str_bool_nonopto], range_val, binsize)
        
        _, meanvel_nonopto, __, vel_nonopto = perireward_binned_activity(velocity[str_bool_nonopto], rewards_nonopto_eps[str_bool_nonopto], 
            timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto],
            range_val, binsize)
        _, meanlick_nonopto, __, licktr_nonopto = perireward_binned_activity(licks[str_bool_nonopto], rewards_nonopto_eps[str_bool_nonopto], 
                timedFF_opto_eps[str_bool_nonopto],trialnum_opto_eps[str_bool_nonopto],
                range_val, binsize)
    

        # nonopto trial in the non epoches, can aligned to cs
        ybinned_nonopto_eps = ybinned[eps_nonopto_mask]
        trialnum_nonopto_eps = trialnum[eps_nonopto_mask]
        rewards_nonopto_eps = rewards[eps_nonopto_mask]
        dff_nonopto_eps = dff[eps_nonopto_mask]
        timedFF_nonopto_eps = timedFF[eps_nonopto_mask]

        #changeRewLoc = np.hstack(params['changeRewLoc'])     
        #scalingf=2/3
        #eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))        
        #mask = np.arange(0,eps[len(eps)-1])
        # mask = np.arange(0,eps[2])
        normmeanrewdFF_nonopto_eps, meanrewdFF_nonopto_eps, normrewdFF_nonopto_eps, \
            rewdFF_nonopto_eps = perireward_binned_activity(dff_nonopto_eps, rewards_nonopto_eps, 
                timedFF_nonopto_eps, trialnum_nonopto_eps,
                range_val, binsize)
        _, meanvel_nonopto_eps, __, vel_nonopto_eps = perireward_binned_activity(velocity[eps_nonopto_mask], rewards[eps_nonopto_mask], 
            timedFF[eps_nonopto_mask], trialnum[eps_nonopto_mask],
            range_val, binsize)
        _, meanlick_nonopto_eps, __, licktr_nonopto_eps = perireward_binned_activity(licks[eps_nonopto_mask], rewards[eps_nonopto_mask], 
                timedFF[eps_nonopto_mask], trialnum[eps_nonopto_mask],
                range_val, binsize)











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
            ax.set_xticks(np.arange(0,len(timedFF)+1000,1000))
            ax.set_xticklabels(np.round(np.append(timedFF[::1000]/60,timedFF[-1]/60), 1))
            ax.set_xlabel('Time (minutes)')
            fig.tight_layout()
            pdf.savefig(fig)

        #TODO: peri reward fails 
        #TODO: peri reward catch trials
        # all subsequent rews                      
        # only ep3?
        # aligned to CS

            #offpln=pln+1 if pln<3 else pln-1
            #min_iind = find_start_points(stims[offpln::4])
            #startofstims = np.zeros_like(dff)
            #startofstims[min_iind]=1

            #rewards = startofstims
        '''
            starts = [0] if stims[offpln::4][0] == 1 else []
            changes = np.where(np.diff(stims[offpln::4]) == 1)[0]
            starts.extend(changes + 1)
            min_iind = starts
            startofstims = np.zeros_like(dff)
            startofstims[min_iind]=1

            #rewards = startofstims
        '''


        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
# plot the dff of opto trials in opto epoch
        clean_arr = rewdFF_opto.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Opto Trials in no Opto epoches')
        ax.axvline(int(range_val/binsize),linestyle='--',color='w')
        ax.set_ylabel('Trial #')
        ax = axes[3]
        ax.plot(meanrewdFF_opto)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)        
        ax = axes[2]
        ax2 = ax.twinx()
        meanvel_opto=np.nanmedian(vel_opto,axis=1)
        ax.plot(meanvel_opto,color='k')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanvel_opto-scipy.stats.sem(vel_opto,axis=1,nan_policy='omit'),
            meanvel_opto+scipy.stats.sem(vel_opto,axis=1,nan_policy='omit'), alpha=0.3,color='k')
        # licks
        ax2.plot(meanlick_opto,color='slategray')   
        xmin,xmax = ax.get_xlim()     
        ax2.fill_between(range(0,int(range_val/binsize)*2), 
            meanlick_opto-scipy.stats.sem(licktr_opto,axis=1,nan_policy='omit'),
            meanlick_opto+scipy.stats.sem(licktr_opto,axis=1,nan_policy='omit'), alpha=0.3,
            color='slategray')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Velocity (cm/s)')
        ax2.set_ylabel('Licks')
        ax.set_xlabel('Time from CS (s)')
        fig.suptitle(f'Peri CS {animal}, Day {day}, {layer}')        
        fig.tight_layout()
        pdf.savefig(fig)        
        plndff.append(clean_arr)






# plot the dff of nonopto trials in opto epoch
        clean_arr = rewdFF_nonopto.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Trials in no Opto epoches')
        ax.axvline(int(range_val/binsize),linestyle='--',color='w')
        ax.set_ylabel('Trial #')
        ax = axes[3]
        ax.plot(meanrewdFF_nonopto)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)        
        ax = axes[2]
        ax2 = ax.twinx()
        meanvel_nonopto=np.nanmedian(vel_nonopto,axis=1)
        ax.plot(meanvel_nonopto,color='k')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanvel_nonopto-scipy.stats.sem(vel_nonopto,axis=1,nan_policy='omit'),
            meanvel_nonopto+scipy.stats.sem(vel_nonopto,axis=1,nan_policy='omit'), alpha=0.3,color='k')
        # licks
        ax2.plot(meanlick_nonopto,color='slategray')   
        xmin,xmax = ax.get_xlim()     
        ax2.fill_between(range(0,int(range_val/binsize)*2), 
            meanlick_nonopto-scipy.stats.sem(licktr_nonopto,axis=1,nan_policy='omit'),
            meanlick_nonopto+scipy.stats.sem(licktr_nonopto,axis=1,nan_policy='omit'), alpha=0.3,
            color='slategray')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Velocity (cm/s)')
        ax2.set_ylabel('Licks')
        ax.set_xlabel('Time from CS (s)')
        fig.suptitle(f'Peri CS {animal}, Day {day}, {layer}')        
        fig.tight_layout()
        pdf.savefig(fig)        
        plndff.append(clean_arr)


        # plot the dff of None Opto epoches
        clean_arr = rewdFF_nonopto_eps.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Trials in no Opto epoches')
        ax.axvline(int(range_val/binsize),linestyle='--',color='w')
        ax.set_ylabel('Trial #')
        ax = axes[3]
        ax.plot(meanrewdFF_nonopto_eps)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF_nonopto_eps-scipy.stats.sem(rewdFF_nonopto_eps,axis=1,nan_policy='omit'),
                meanrewdFF_nonopto_eps+scipy.stats.sem(rewdFF_nonopto_eps,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)        
        ax = axes[2]
        ax2 = ax.twinx()
        meanvel_nonopto_eps=np.nanmedian(vel_nonopto_eps,axis=1)
        ax.plot(meanvel_nonopto_eps,color='k')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanvel_nonopto_eps-scipy.stats.sem(vel_nonopto_eps,axis=1,nan_policy='omit'),
            meanvel_nonopto_eps+scipy.stats.sem(vel_nonopto_eps,axis=1,nan_policy='omit'), alpha=0.3,color='k')
        # licks
        ax2.plot(meanlick_nonopto_eps,color='slategray')   
        xmin,xmax = ax.get_xlim()     
        ax2.fill_between(range(0,int(range_val/binsize)*2), 
            meanlick_nonopto_eps-scipy.stats.sem(licktr_nonopto_eps,axis=1,nan_policy='omit'),
            meanlick_nonopto_eps+scipy.stats.sem(licktr_nonopto_eps,axis=1,nan_policy='omit'), alpha=0.3,
            color='slategray')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.axvline(int(range_val/binsize),linestyle='--',color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Velocity (cm/s)')
        ax2.set_ylabel('Licks')
        ax.set_xlabel('Time from CS (s)')
        fig.suptitle(f'Peri CS {animal}, Day {day}, {layer}')        
        fig.tight_layout()
        pdf.savefig(fig)        
        plndff.append(clean_arr)
    day_date_dff[str(day)] = plndff
pdf.close()

#%%
# heatmap across days
# alltr = np.array([np.concatenate([v[i] for k,v in day_date_dff.items()]) for i in range(4)])
normalized_dff = {}

for k, v in day_date_dff.items():
    # v is a list of 4 arrays: v[0], v[1], v[2], v[3]
    lengths = [arr.shape[0] for arr in v]
    min_len = min(lengths)

    # Trim all 4 planes to the shortest one
    trimmed = [arr[:min_len] for arr in v]

    normalized_dff[k] = trimmed

alltr = np.array([
    np.concatenate([v[i] for v in normalized_dff.values()], axis=0)
    for i in range(4)
])


# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,1.5],sharex=True,figsize=(6,3))
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
'''
# find linear fit for each plane
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

slopes = []

for pln in range(4):
    mean_vec = np.mean(alltr, axis=1)[pln]  # shape: (100,)
    
    half_len = mean_vec.shape[0] // 2  # Integer division
    y_vals = mean_vec[:half_len]       # First half
    x_vals = np.arange(half_len)  

    # Filter out NaNs
    mask = ~np.isnan(y_vals)
    x_clean = x_vals[mask]
    y_clean = y_vals[mask]

    if len(x_clean) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        slopes.append(slope)

        # Plot
        plt.plot(x_clean, y_clean, label='Data')
        plt.plot(x_clean, slope * x_clean + intercept, '--', label='Linear Fit')
        plt.xlabel('Index')
        plt.ylabel('Mean Value')
        plt.title(f'Linear Regression on First Half - PLN {pln}')
        plt.legend()
        plt.ylim(0.99, 1.01)  # Set y-axis limits
        plt.grid(True)
        plt.show()
    else:
        slopes.append(np.nan)
        print(f"PLN {pln}: Not enough valid data")

print("Slopes for each PLN:")
for pln, slope in enumerate(slopes):
    print(f"PLN {pln}: {slope}")
# Result: slopes = [slope0, slope1, slope2, slope3]
'''
# %%
def find_start_points(data):
    # Check if the first element is 1
    starts = [0] if data[0] == 1 else []
    
    # Find the indices where data changes from 0 to 1
    changes = np.where(np.diff(data) == 1)[0]
    
    # Since np.diff reduces the length by 1, add 1 to each index to get the actual start points
    starts.extend(changes + 1)
    
    return starts
