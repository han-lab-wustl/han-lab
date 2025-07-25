""" 
Ziyi's dopamine hrz with opto analysis
For each day, only one epoch has opto, in this one epoch, every other trial has opto at the oppsite 
location of the reward zone
Adapted from zahra's dopamine hrz analysis
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib

from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
from projects.opto.behavior.behavior import smooth_lick_rate, get_lick_selectivity
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
#days = np.arange(19,23)
# opto days [19,20,22,23]
# control days [21,24]
days = [28]
range_val=10; binsize=0.2
numtrialsstim=2 # every 10 trials stim w 1 trial off
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
# figs = True # print out per day figs

day_date_dff_opto ={}
day_date_dff_nonopto ={}
day_date_dff_nonopto_eps = {}
day_date_dff_nonopto_after_opto = {}

# opto centered lickrate in opto epoch
day_lickrate_opto_centered_opto = {}
day_lickrate_opto_centered_nonopto = {}
#day_lickerate_opto_centered_




for day in days: 
    plndff_opto = []
    plndff_nonopto = []
    plndff_nonopto_eps = []
    lick_rate_opto_centered_opto = []
    lick_rate_opto_centered_nonopto = []
    stimspth = list(Path(os.path.join(src, str(day))).rglob('*000*.mat'))[0]
    #stimspth = Path(os.path.join(src, str(day))).rglob('*000*.mat')
    stims = scipy.io.loadmat(stimspth)
    stims = np.hstack(stims['stims']) # nan out stims
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        VR = params['VR'][0][0][()]
        gainf = VR['scalingFACTOR'][0][0]
        try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/gainf        
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
        framerate = 7.8
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(5).mean().values)
        rewards = np.hstack(params['solenoid2'])
        velocity = np.hstack(params['forwardvel'])
        veldf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(veldf.rolling(10).mean().values)
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        timedFF = np.hstack(params['timedFF'])
        changeRewLoc = np.hstack(params['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
        eps = np.append(eps, len(changeRewLoc))
        lick_rate = smooth_lick_rate(licks, 1/framerate, sigma_sec=0.7)
        
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
        
        #epsALL_has_stim = [False, True, False, False]
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
        lick_rate_opto_eps = lick_rate[eps_opto_mask]
       
 
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum_opto_eps, rewards_opto_eps)

        all_tr = sorted(str_trials + ftr_trials)
        str_bool_opto_all = np.array([(xx in all_tr) and 
                 (xx%numtrialsstim==1) for xx in trialnum_opto_eps])

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
        # This is aligned to the cs rewards_opto_eps
        normmeanrewdFF_opto, meanrewdFF_opto, normrewdFF_opto, \
            rewdFF_opto = eye.perireward_binned_activity(dff_opto_eps[str_bool_opto],
            rewards_opto_eps[str_bool_opto], timedFF_opto_eps[str_bool_opto], range_val, binsize)

        _, meanvel_opto, __, vel_opto = perireward_binned_activity(velocity_opto_eps[str_bool_opto],  rewards_opto_eps[str_bool_opto], 
            timedFF_opto_eps[str_bool_opto], trialnum_opto_eps[str_bool_opto],
            range_val, binsize)
        _, meanlick_opto, __, licktr_opto = perireward_binned_activity(licks_opto_eps[str_bool_opto],  rewards_opto_eps[str_bool_opto], 
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
            rewards_opto_eps[str_bool_nonopto], timedFF_opto_eps[str_bool_nonopto], range_val, binsize)
        
        _, meanvel_nonopto, __, vel_nonopto = perireward_binned_activity(velocity_opto_eps[str_bool_nonopto], rewards_opto_eps[str_bool_nonopto], 
            timedFF_opto_eps[str_bool_nonopto], trialnum_opto_eps[str_bool_nonopto],
            range_val, binsize)
        _, meanlick_nonopto, __, licktr_nonopto = perireward_binned_activity(licks_opto_eps[str_bool_nonopto], rewards_opto_eps[str_bool_nonopto], 
                timedFF_opto_eps[str_bool_nonopto],trialnum_opto_eps[str_bool_nonopto],
                range_val, binsize)
    
        # nonopto trial in the non epoches, can aligned to cs
        ybinned_nonopto_eps = ybinned[eps_nonopto_mask]
        trialnum_nonopto_eps = trialnum[eps_nonopto_mask]
        rewards_nonopto_eps = rewards[eps_nonopto_mask]
        dff_nonopto_eps = dff[eps_nonopto_mask]
        timedFF_nonopto_eps = timedFF[eps_nonopto_mask]
        lick_rate_nonopto_eps = lick_rate[eps_nonopto_mask]

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


        # Opto trials in Opto epoch!! This is aligned to when the opto happens 
        #stimzone = ((newrewloc*gainf-((rewsize*gainf)/2)+90)%180)/gainf
        stimzone = 20/gainf
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[(ybinned_opto_eps >= stimzone-5) & (ybinned_opto_eps <= stimzone+5)]=1
        rews_iind = consecutive_stretch(np.where(opto_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[min_iind]=1
        #opto_centered = opto_centered*str_bool_opto
        normmeanOptodFF, meanOptodFF, normOptodFF, \
        optodFF = eye.perireward_binned_activity(dff_opto_eps[str_bool_opto],
        opto_centered[str_bool_opto], timedFF_opto_eps[str_bool_opto], range_val, binsize)
        _, meanlick_rate_opto_centered_opto, __, lick_rate_opto_centered_opto = eye.perireward_binned_activity(lick_rate_opto_eps[str_bool_opto],  opto_centered[str_bool_opto], 
                timedFF_opto_eps[str_bool_opto],range_val, binsize)        
    
        # Non Opto trials in Opto epoch!! This is aligned to when the opto happens 
        '''
        stimzone = ((newrewloc*gainf-((rewsize*gainf)/2)+90)%180)/gainf
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[(ybinned_opto_eps >= stimzone-5) & (ybinned_opto_eps <= stimzone+5)]=1
        rews_iind = consecutive_stretch(np.where(opto_centered)[0])
        min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
        opto_centered = np.zeros_like(ybinned_opto_eps)
        opto_centered[min_iind]=1
        #opto_centered = opto_centered*str_bool_opto
        '''
        normmeannonOptodFF, meannonOptodFF, normnonOptodFF, \
        NonoptodFF = eye.perireward_binned_activity(dff_opto_eps[ str_bool_nonopto],
        opto_centered[str_bool_nonopto], timedFF_opto_eps[ str_bool_nonopto], range_val, binsize)
        _, meanlick_rate_opto_centered_nonopto, __, lick_rate_opto_centered_nonopto = eye.perireward_binned_activity(lick_rate_opto_eps[str_bool_nonopto],  opto_centered[str_bool_nonopto], 
        timedFF_opto_eps[str_bool_nonopto],range_val, binsize)    









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

            plt.figure(figsize=(6, 3))
            
            # X-axis values
            x_vals = range(0, int(range_val / binsize) * 2)

            # Opto
            sem_opto = scipy.stats.sem(lick_rate_opto_centered_opto, axis=1, nan_policy='omit')
            plt.plot(meanlick_rate_opto_centered_opto, label='Opto')
            plt.fill_between(x_vals,
                            meanlick_rate_opto_centered_opto - sem_opto,
                            meanlick_rate_opto_centered_opto + sem_opto,
                            alpha=0.3)

            # Non-opto
            sem_nonopto = scipy.stats.sem(lick_rate_opto_centered_nonopto, axis=1, nan_policy='omit')
            plt.plot(meanlick_rate_opto_centered_nonopto, label='Non-opto')
            plt.fill_between(x_vals,
                            meanlick_rate_opto_centered_nonopto - sem_nonopto,
                            meanlick_rate_opto_centered_nonopto + sem_nonopto,
                            alpha=0.3)

            # X-axis ticks and labels
            plt.xticks(ticks=range(0, (int(range_val / binsize) * 2) + 1, 10),
                    labels=range(-range_val, range_val + 1, 2))
            plt.xlabel("Time (s)")
            plt.ylabel("Lick Rate (Hz)")
            plt.title(f'Day {day}')

            # Vertical line at time 0
            plt.axvline(int(range_val / binsize), linestyle='--', color='k')
            

            day_lickrate_opto_centered_opto[str(day)] = lick_rate_opto_centered_opto.T
            day_lickrate_opto_centered_nonopto[str(day)] = lick_rate_opto_centered_nonopto.T

            plt.legend()
            plt.tight_layout()
            plt.show()


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
        clean_arr_opto= rewdFF_opto.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr_opto)
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
        plndff_opto.append(clean_arr_opto)






# plot the dff of nonopto trials in opto epoch
        clean_arr_nonopto = rewdFF_nonopto.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr_nonopto)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('No opto Trials in Opto epoches')
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
        plndff_nonopto.append(clean_arr_nonopto)


        # plot the dff of None Opto epoches
        clean_arr_nonopto_eps = rewdFF_nonopto_eps.T#[~rows_with_nans]    
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
        axes = axes.flatten()  # Flatten the axes array for easier plotting
        ax=axes[0]
        ax.imshow(params['params'][0][0][0],cmap="Greys_r")
        ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
        ax.axis('off')
        ax = axes[1]
        ax.imshow(clean_arr_nonopto_eps)
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
        plndff_nonopto_eps.append(clean_arr_nonopto_eps)
    
    day_date_dff_opto[str(day)] = plndff_opto
    day_date_dff_nonopto[str(day)] = plndff_nonopto
    day_date_dff_nonopto_eps[str(day)] = plndff_nonopto_eps
pdf.close()

#%%
# heatmap across days
# alltr = np.array([np.concatenate([v[i] for k,v in day_date_dff.items()]) for i in range(4)])
normalized_dff_opto = {}

for k, v in day_date_dff_opto.items():
    # v is a list of 4 arrays: v[0], v[1], v[2], v[3]
    lengths = [arr.shape[0] for arr in v]
    min_len = min(lengths)

    # Trim all 4 planes to the shortest one
    trimmed = [arr[:min_len] for arr in v]

    normalized_dff_opto[k] = trimmed

alltr_opto = np.array([
    np.concatenate([v[i] for v in normalized_dff_opto.values()], axis=0)
    for i in range(4)
])


# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,1.5],sharex=True,figsize=(6,3))
    ax=axes[0]
    cax=ax.imshow(alltr_opto[pln,:,:])    
    ax.set_xlabel('Time from CS (s)')
    ax.set_ylabel('Trials (last 4 days)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='w')
    # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
    ax.set_title(f'Opto trials in opto epoch Plane {planelut[pln]}')
    fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
    ax=axes[1]
    mf = np.nanmean(alltr_opto[pln,:,:],axis=0)
    ax.plot(mf)    
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    mf-scipy.stats.sem(alltr_opto[pln,:,:],axis=0,nan_policy='omit'),
    mf+scipy.stats.sem(alltr_opto[pln,:,:],axis=0,nan_policy='omit'), alpha=0.3)
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    fig.tight_layout()


normalized_dff_nonopto = {}

for k, v in day_date_dff_nonopto.items():
    # v is a list of 4 arrays: v[0], v[1], v[2], v[3]
    lengths = [arr.shape[0] for arr in v]
    min_len = min(lengths)

    # Trim all 4 planes to the shortest one
    trimmed = [arr[:min_len] for arr in v]

    normalized_dff_nonopto[k] = trimmed

alltr_nonopto = np.array([
    np.concatenate([v[i] for v in normalized_dff_nonopto.values()], axis=0)
    for i in range(4)
])


# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,1.5],sharex=True,figsize=(6,3))
    ax=axes[0]
    cax=ax.imshow(alltr_nonopto[pln,:,:])    
    ax.set_xlabel('Time from CS (s)')
    ax.set_ylabel('Trials (last 4 days)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='w')
    # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
    ax.set_title(f'No opto trials in opto epoch Plane {planelut[pln]}')
    fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
    ax=axes[1]
    mf = np.nanmean(alltr_nonopto[pln,:,:],axis=0)
    ax.plot(mf)    
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    mf-scipy.stats.sem(alltr_nonopto[pln,:,:],axis=0,nan_policy='omit'),
    mf+scipy.stats.sem(alltr_nonopto[pln,:,:],axis=0,nan_policy='omit'), alpha=0.3)
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    fig.tight_layout()


normalized_dff_nonopto_eps = {}

for k, v in day_date_dff_nonopto_eps.items():
    # v is a list of 4 arrays: v[0], v[1], v[2], v[3]
    lengths = [arr.shape[0] for arr in v]
    min_len = min(lengths)

    # Trim all 4 planes to the shortest one
    trimmed = [arr[:min_len] for arr in v]

    normalized_dff_nonopto_eps[k] = trimmed

alltr_nonopto_eps = np.array([
    np.concatenate([v[i] for v in normalized_dff_nonopto_eps.values()], axis=0)
    for i in range(4)
])


# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,1.5],sharex=True,figsize=(6,3))
    ax=axes[0]
    cax=ax.imshow(alltr_nonopto_eps[pln,:,:])    
    ax.set_xlabel('Time from CS (s)')
    ax.set_ylabel('Trials (last 4 days)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='w')
    # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
    ax.set_title(f'No opto epoches Plane {planelut[pln]}')
    fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
    ax=axes[1]
    mf = np.nanmean(alltr_nonopto_eps[pln,:,:],axis=0)
    ax.plot(mf)    
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    mf-scipy.stats.sem(alltr_nonopto_eps[pln,:,:],axis=0,nan_policy='omit'),
    mf+scipy.stats.sem(alltr_nonopto_eps[pln,:,:],axis=0,nan_policy='omit'), alpha=0.3)
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    fig.tight_layout()


# Get the lick rate for ALL the opto and nonopto trials in the opto epoches
alltr_lickrate_opto_centered_opto = np.concatenate(list(day_lickrate_opto_centered_opto.values()), axis=0)
alltr_lickrate_opto_centered_nonopto = np.concatenate(list(day_lickrate_opto_centered_nonopto.values()), axis=0)
#%%

'''
# It's the per day average for the lic_rate, not all trials

alltr_lickrate_opto_centered_opto = np.array([
    np.nanmean(arr, axis=0) for arr in day_lickrate_opto_centered_opto.values()
])

alltr_lickrate_opto_centered_nonopto = np.array([
    np.nanmean(arr, axis=0) for arr in day_lickrate_opto_centered_nonopto.values()
])

# Means (ignoring NaNs)
mean_opto = np.nanmean(alltr_lickrate_opto_centered_opto, axis=0)
mean_nonopto = np.nanmean(alltr_lickrate_opto_centered_nonopto, axis=0)

# SEMs (ignoring NaNs)
sem_opto = scipy.stats.sem(alltr_lickrate_opto_centered_opto, axis=0, nan_policy='omit')
sem_nonopto = scipy.stats.sem(alltr_lickrate_opto_centered_nonopto, axis=0, nan_policy='omit')

# Time axis
n_timepoints = alltr_lickrate_opto_centered_opto.shape[1]
time = np.linspace(-range_val, range_val, n_timepoints)

# Absolute difference
abs_diff = np.abs(mean_opto - mean_nonopto)
area_between_curves = np.nansum(abs_diff) * binsize
mean_abs_diff = np.nanmean(abs_diff)

# Cohen's d vectorized (with NaN-safe filtering)
def cohens_d_vec(a, b):
    d_vals = []
    for i in range(a.shape[1]):
        x, y = a[:, i], b[:, i]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        pooled_sd = np.sqrt(((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)/2))
        d = (np.mean(x) - np.mean(y)) / pooled_sd if pooled_sd > 0 else np.nan
        d_vals.append(d)
    return np.array(d_vals)

d_curve = cohens_d_vec(alltr_lickrate_opto_centered_opto, alltr_lickrate_opto_centered_nonopto)
mean_abs_d = np.nanmean(np.abs(d_curve))

# Correlation (nan-safe)
valid_mask = ~np.isnan(mean_opto) & ~np.isnan(mean_nonopto)
if np.sum(valid_mask) > 2:
    r, p_corr = pearsonr(mean_opto[valid_mask], mean_nonopto[valid_mask])
else:
    r, p_corr = np.nan, np.nan

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Lick rate curves
axs[0].plot(time, mean_opto, label='Opto', color='#1f77b4')
axs[0].fill_between(time, mean_opto - sem_opto, mean_opto + sem_opto, color='#1f77b4', alpha=0.3)
axs[0].plot(time, mean_nonopto, label='Non-opto', color='#ff7f0e')
axs[0].fill_between(time, mean_nonopto - sem_nonopto, mean_nonopto + sem_nonopto, color='#ff7f0e', alpha=0.3)
axs[0].axvline(0, linestyle='--', color='black')
axs[0].set_title("Mean Lick Rate: Opto vs Non-Opto")
axs[0].set_ylabel("Lick Rate (Hz)")
axs[0].legend()

# Abs diff
axs[1].plot(time, abs_diff, color='black')
axs[1].set_title(f"Absolute Difference | Area = {area_between_curves:.2f} Hz·s, Mean = {mean_abs_diff:.2f} Hz")
axs[1].set_ylabel("Abs Diff (Hz)")

# Cohen's d
axs[2].plot(time, d_curve, color='purple')
axs[2].axhline(0, linestyle='--', color='gray')
axs[2].set_title(f"Cohen's d | Mean |d| = {mean_abs_d:.2f} | Corr r = {r:.2f}, p = {p_corr:.3f}")
axs[2].set_ylabel("Cohen's d")
axs[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

#%%

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

def find_start_points(data):
    # Check if the first element is 1
    starts = [0] if data[0] == 1 else []
    
    # Find the indices where data changes from 0 to 1
    changes = np.where(np.diff(data) == 1)[0]
    
    # Since np.diff reduces the length by 1, add 1 to each index to get the actual start points
    starts.extend(changes + 1)
    
    return starts

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, sem

alltr_lick_binary_opto = np.concatenate(list(day_lickrate_opto_centered_opto.values()), axis=0)
alltr_lick_binary_opto = np.concatenate(list(day_lickrate_opto_centered_nonopto.values()), axis=0)


# ---- Parameters ----
threshold = 0.1   # Hz threshold to detect licks
max_ili = 0.2     # max interval (s) between licks for burst
jitter = 0.02     # for scatter plot

# ---- First Lick Latency ----
def compute_first_lick_latency(lickrate_array, binsize, range_val, threshold=0.1):
    latencies = []
    zero_bin = int(range_val / binsize)
    for trial in lickrate_array:
        post_zero = trial[zero_bin:]
        lick_bins = np.where(post_zero > threshold)[0]
        latency = lick_bins[0] * binsize if len(lick_bins) > 0 else np.nan
        latencies.append(latency)
    return np.array(latencies)

lat_opto = compute_first_lick_latency(alltr_lickrate_opto_centered_opto, binsize, range_val, threshold)
lat_nonopto = compute_first_lick_latency(alltr_lickrate_opto_centered_nonopto, binsize, range_val, threshold)

# ---- Lick Burst Count from Lick Rate ----
def detect_burst_counts(lickrate_array, binsize, threshold=0.1, max_ili=0.2):
    burst_counts = []
    for trial in lickrate_array:
        lick_bins = np.where(trial > threshold)[0]
        if len(lick_bins) < 2:
            burst_counts.append(0)
            continue
        ilis = np.diff(lick_bins) * binsize
        burst_count = 0
        in_burst = False
        for ili in ilis:
            if ili <= max_ili:
                if not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                in_burst = False
        burst_counts.append(burst_count)
    return np.array(burst_counts)

burst_opto = detect_burst_counts(alltr_lickrate_opto_centered_opto, binsize, threshold, max_ili)
burst_nonopto = detect_burst_counts(alltr_lickrate_opto_centered_nonopto, binsize, threshold, max_ili)

# ---- Plot: Latency ----
x = [0.4, 0.6]
labels = ['Opto', 'Non-opto']
means = [np.nanmean(lat_opto), np.nanmean(lat_nonopto)]
sems = [sem(lat_opto, nan_policy='omit'), sem(lat_nonopto, nan_policy='omit')]

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(x, means, yerr=sems, width=0.15, color=['#1f77b4', '#ff7f0e'], alpha=0.6, capsize=4, edgecolor='black')
ax.scatter(np.full_like(lat_opto, x[0]) + np.random.uniform(-jitter, jitter, len(lat_opto)), lat_opto,
           alpha=0.6, color='black', s=10)
ax.scatter(np.full_like(lat_nonopto, x[1]) + np.random.uniform(-jitter, jitter, len(lat_nonopto)), lat_nonopto,
           alpha=0.6, color='black', s=10)

# Significance
stat, p_val = mannwhitneyu(lat_opto, lat_nonopto, alternative='two-sided')
sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
y_max = max(means[0] + sems[0], means[1] + sems[1])
ax.plot([x[0], x[0], x[1], x[1]], [y_max+0.05, y_max+0.07, y_max+0.07, y_max+0.05], color='black')
ax.text(0.5, y_max + 0.1, sig, ha='center', va='bottom', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("First Lick Latency (s)")
ax.set_title("Latency to First Lick", pad=12)
plt.tight_layout()
plt.show()

# ---- Plot: Burst Count ----
means = [np.nanmean(burst_opto), np.nanmean(burst_nonopto)]
sems = [sem(burst_opto, nan_policy='omit'), sem(burst_nonopto, nan_policy='omit')]

fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(x, means, yerr=sems, width=0.15, color=['#1f77b4', '#ff7f0e'], alpha=0.6, capsize=4, edgecolor='black')
ax.scatter(np.full_like(burst_opto, x[0]) + np.random.uniform(-jitter, jitter, len(burst_opto)), burst_opto,
           alpha=0.6, color='black', s=10)
ax.scatter(np.full_like(burst_nonopto, x[1]) + np.random.uniform(-jitter, jitter, len(burst_nonopto)), burst_nonopto,
           alpha=0.6, color='black', s=10)

# Significance
stat, p_val = mannwhitneyu(burst_opto, burst_nonopto, alternative='two-sided')
sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
y_max = max(means[0] + sems[0], means[1] + sems[1])
ax.plot([x[0], x[0], x[1], x[1]], [y_max+0.5, y_max+0.7, y_max+0.7, y_max+0.5], color='black')
ax.text(0.5, y_max + 0.9, sig, ha='center', va='bottom', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Lick Bursts per Trial")
ax.set_title("Lick Burst Count", pad=12)
plt.tight_layout()
plt.show()
'''
