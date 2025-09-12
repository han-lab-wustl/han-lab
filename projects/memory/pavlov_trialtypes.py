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
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
from projects.DLC_behavior_classification.eye import perireward_binned_activity
plt.rcParams["font.family"] = "Arial"

plt.close('all')
# save to pdf
animal = 'E169params'
src = r"\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\analysisdrive"
# src=r'Y:\halo_grabda'
src = os.path.join(src,animal)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"pavlov_cs_{os.path.basename(src)}.pdf"))
days = ['D10_RRauto1']#np.arange(11,27)
range_val=2; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = True
# figs = True # print out per day figs
day_date_dff = {}
for day in days: 
   plndff = []
   # for each plane
   for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
      params = scipy.io.loadmat(path)
      try:
         VR = params['VR'][0][0][()]
         gainf = VR['scalingFACTOR'][0][0]
         rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
      except:
         rewsize = 10
         gainf=1/1.5

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
      dff = np.hstack(dffdf.rolling(2).mean().values)
      cs = np.hstack(params['solenoid2'])
      us = np.hstack(params['rewards'])
      velocity = np.hstack(params['forwardvel'])
      veldf = pd.DataFrame({'velocity': velocity})
      velocity = np.hstack(veldf.rolling(10).mean().values)
      trialnum = np.hstack(params['trialnum'])
      ybinned = np.hstack(params['ybinned'])/(2/3)
      licks = np.hstack(params['licks'])
      changeRewLoc = np.hstack(params['changeRewLoc'])
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/gainf
      eps = np.append(eps, len(changeRewLoc))
      doubles=us>1
      # plot pre-first reward dop activity  
      timedFF = np.hstack(params['timedFF'])
      # aligned to CS
      changeRewLoc = np.hstack(params['changeRewLoc'])     
      scalingf=2/3
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))        
      mask = np.arange(0,eps[len(eps)-1])
      # rewards[:1000]=0
      # rewards[-1000:]=0
      # mask = np.arange(0,eps[2])
      # does not take trial input
      normmeanrewdFF, meanrewdFF, normrewdFF, \
         rewdFF = perireward_binned_activity(dff[mask], cs[mask], 
               timedFF[mask], 
               range_val, binsize)
      _, meanvel, __, vel = perireward_binned_activity(velocity[mask], cs[mask], 
         timedFF[mask], 
         range_val, binsize)
      _, meanlick, __, licktr = perireward_binned_activity(licks[mask], cs[mask], 
               timedFF[mask], 
               range_val, binsize)

      # Find the rows that contain NaNs
      # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
      # Select rows that do not contain any NaNs
      clean_arr = rewdFF.T#[~rows_with_nans]    
      fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(6,4))
      axes = axes.flatten()  # Flatten the axes array for easier plotting
      ax=axes[0]
      ax.imshow(params['params'][0][0][0],cmap="Greys_r")
      ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
      ax.axis('off')
      ax = axes[1]
      ax.imshow(clean_arr,aspect='auto')
      ax.set_xticks([0,int(len(clean_arr.T)/2),len(clean_arr.T)])
      ax.set_xticklabels([-range_val, 0,range_val])
      ax.set_title('Correct Trials')
      ax.axvline(int(range_val/binsize),linestyle='--',color='w')
      ax.set_ylabel('Trial #')
      ax = axes[3]
      ax.plot(meanrewdFF)   
      xmin,xmax = ax.get_xlim()     
      ax.fill_between(range(0,int(range_val/binsize)*2), 
               meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
               meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
      ax.set_xticks([0,int(len(rewdFF)/2),len(rewdFF)])
      ax.set_xticklabels([-range_val, 0,range_val])
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
      ax.set_xticks([0,int(len(meanlick)/2),len(meanlick)])
      ax.set_xticklabels([-range_val, 0,range_val])
      ax.axvline(int(range_val/binsize),linestyle='--',color='k')
      ax.spines[['top','right']].set_visible(False)
      ax.set_ylabel('Velocity (cm/s)')
      ax2.set_ylabel('Licks')
      ax.set_xlabel('Time from CS (s)')
      fig.suptitle(f'Peri CS, {animal}, Day {day}, {layer}')        
      fig.tight_layout()
      pdf.savefig(fig)  
      # --- classify trial types ---
      cs_idx = np.where(cs > 0)[0]
      us_idx = np.where(us > 0)[0]

      # 1) CS followed by US==2
      cs_followed_us2_idx = []
      for i in cs_idx:
         next_us = us_idx[us_idx > i]
         if len(next_us) > 0 and us[next_us[0]] == 2:
            cs_followed_us2_idx.append(i)
      cs_followed_us2_idx = np.array(cs_followed_us2_idx)

      # 2) US without CS
      #secs check
      sec= 5
      us_wo_cs_idx = []
      for j in us_idx:
         # check if no CS within a short window before US
         if not np.any((cs_idx < j+int(sec/np.nanmedian(np.diff(timedFF)))) & (cs_idx > j-int(sec/np.nanmedian(np.diff(timedFF))))):
            us_wo_cs_idx.append(j)
      us_wo_cs_idx = np.array(us_wo_cs_idx)

      # 3) CS without US
      cs_wo_us_idx = []
      for i in cs_idx:
         if not np.any((us_idx < i+int(sec/np.nanmedian(np.diff(timedFF)))) & (us_idx > i-int(sec/np.nanmedian(np.diff(timedFF))))):
            cs_wo_us_idx.append(i)
      cs_wo_us_idx = np.array(cs_wo_us_idx)

      # --- helper function using your perireward_binned_activity ---
      def peri_raster(signal, align_events, timedFF, range_val, binsize):
         if len(align_events) == 0:
            return np.empty((0, int((2*range_val)/binsize)))  # empty
         _, _, _, peri_mat = perireward_binned_activity(
            signal, np.isin(np.arange(len(signal)), align_events).astype(int),
            timedFF, range_val, binsize
         )
         return peri_mat.T  # trials x time
      # other normal cs-us
      cs_idx=[xx for xx in cs_idx if xx not in cs_followed_us2_idx and xx not in cs_wo_us_idx]
      # --- collect rasters ---
      raster_cs_us2 = peri_raster(dff, cs_followed_us2_idx, timedFF, range_val, binsize)
      raster_us_wo_cs = peri_raster(dff, us_wo_cs_idx, timedFF, range_val, binsize)
      raster_cs_wo_us = peri_raster(dff, cs_wo_us_idx, timedFF, range_val, binsize)
      # all normal cs-us
      raster_cs_us = peri_raster(dff, cs_idx, timedFF, range_val, binsize)
      # licks
      licks_cs_us2 = peri_raster(licks, cs_followed_us2_idx, timedFF, range_val, binsize)
      licks_us_wo_cs = peri_raster(licks, us_wo_cs_idx, timedFF, range_val, binsize)
      licks_cs_wo_us = peri_raster(licks, cs_wo_us_idx, timedFF, range_val, binsize)
      licks_cs_us = peri_raster(licks, cs_idx, timedFF, range_val, binsize)

      #vel
      vel_cs_us2 = peri_raster(velocity, cs_followed_us2_idx, timedFF, range_val, binsize)
      vel_us_wo_cs = peri_raster(velocity, us_wo_cs_idx, timedFF, range_val, binsize)
      vel_cs_wo_us = peri_raster(velocity, cs_wo_us_idx, timedFF, range_val, binsize)
      vel_cs_us = peri_raster(velocity, cs_idx, timedFF, range_val, binsize)

      # --- plot rasters ---
      fig, axes = plt.subplots(nrows=3,ncols=4, figsize=(8,5),sharex=True,sharey='row')
      lick_rasters=[licks_cs_us,licks_cs_us2,licks_us_wo_cs,licks_cs_wo_us]
      vel_rasters=[vel_cs_us,vel_cs_us2,vel_us_wo_cs,vel_cs_wo_us]
      for ii, (raster, title) in enumerate(zip(
         [raster_cs_us, raster_cs_us2, raster_us_wo_cs, raster_cs_wo_us],
         ['Single, CS-US',"Doubles, CS→US=2", "Surprise, US w/o CS", "Omission, CS w/o US"]
      )):
         if raster.size > 0:
            ax=axes[0,ii]
            ax.plot(np.nanmean(raster,axis=0))
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(raster,axis=0)-scipy.stats.sem(raster,axis=0,nan_policy='omit'),
            np.nanmean(raster,axis=0)+scipy.stats.sem(raster,axis=0,nan_policy='omit'), alpha=0.3)
            ax.axvline(raster.shape[1]//2, color='k', linestyle='--')
            if ii==0:ax.set_ylabel('dFF')
            ax.spines[['top','right']].set_visible(False)

            ax=axes[1,ii]
            raster=lick_rasters[ii]
            ax.plot(np.nanmean(raster,axis=0),color='slategray')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(raster,axis=0)-scipy.stats.sem(raster,axis=0,nan_policy='omit'),
            np.nanmean(raster,axis=0)+scipy.stats.sem(raster,axis=0,nan_policy='omit'), alpha=0.3,color='slategray')
            ax.axvline(raster.shape[1]//2, color='k', linestyle='--')
            if ii==0:ax.set_ylabel('Licks')
            ax.spines[['top','right']].set_visible(False)

            ax=axes[2,ii]
            raster=vel_rasters[ii]
            ax.plot(np.nanmean(raster,axis=0),color='k')
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(raster,axis=0)-scipy.stats.sem(raster,axis=0,nan_policy='omit'),
            np.nanmean(raster,axis=0)+scipy.stats.sem(raster,axis=0,nan_policy='omit'), alpha=0.3,color='k')
            ax.axvline(raster.shape[1]//2, color='k', linestyle='--')
            if ii==0:ax.set_ylabel('Velocity')
            ax.spines[['top','right']].set_visible(False)
            axes[0,ii].set_title(title)
            ax.set_xticks([0,int(len(raster.T)/2),len(raster.T)])
            ax.set_xticklabels([-range_val, 0,range_val])
      ax.set_xlabel("Time")
      fig.suptitle(f'Trial types, {animal}, Day {day}, {layer}')
      plt.tight_layout()
      pdf.savefig(fig)

      # plt.close('all')      
      plndff.append(clean_arr)
   day_date_dff[str(day)] = [plndff,vel,licktr]
pdf.close()

#%%
trialnum = 100
# heatmap across days
alltr = [np.vstack([v[0][i] for k,v in day_date_dff.items() if len(v[0])==4]) for i in range(4)]
velalltr = np.hstack([v[1] for k,v in day_date_dff.items()]).T
lickalltr = np.hstack([v[2] for k,v in day_date_dff.items()]).T

# all trials
for pln in range(4): 
   fig, axes = plt.subplots(ncols=2,nrows=6,sharex=True,figsize=(5,8),sharey='row')
   ax=axes[0,0]
   arr=alltr[pln][:trialnum]
   cax=ax.imshow(arr,aspect='auto',cmap='jet')    
   ax.set_ylabel('Trials (early trials)')
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   ax.set_title(f'Plane {planelut[pln]}')
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[1,0]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
   ax.set_xticklabels(range(-range_val, range_val+1, 2))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   fig.tight_layout()

   ax=axes[0,1]
   arr=alltr[pln][-trialnum:]
   cax=ax.imshow(arr,aspect='auto',cmap='jet')    
   ax.set_xlabel('Time from CS (s)')
   ax.set_ylabel('Trials (late trials)')
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   ax.set_title(f'Plane {planelut[pln]}')
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[1,1]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
   ax.set_xticklabels(range(-range_val, range_val+1, 2))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   
   # licks
   ax=axes[2,0]
   arr=lickalltr[:trialnum]
   cax=ax.imshow(arr,aspect='auto',cmap='Blues')    
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[3,0]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.set_ylabel('Licks')
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
   ax.set_xticklabels(range(-range_val, range_val+1, 2))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   ax=axes[2,1]
   arr=lickalltr[-trialnum:]
   cax=ax.imshow(arr,aspect='auto',cmap='Blues')    
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[3,1]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.set_ylabel('Licks')
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   
   # vel
   # licks
   ax=axes[4,0]
   arr=velalltr[:trialnum]
   cax=ax.imshow(arr,aspect='auto',cmap='Greys')    
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[5,0]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.set_ylabel('Velocity')
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   ax=axes[4,1]
   arr=velalltr[-trialnum:]
   cax=ax.imshow(arr,aspect='auto',cmap='Greys')    
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
   fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
   ax=axes[5,1]
   mf = np.nanmean(arr,axis=0)
   ax.plot(mf)    
   ax.set_ylabel('Velocity')
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(arr,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(arr,axis=0,nan_policy='omit'), alpha=0.3)
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
   ax.set_xticklabels(range(-range_val, range_val+1, 4))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   ax.set_xlabel('Time from CS (s)')

   fig.tight_layout()

# %%
