"""zahra's pavlov 4s analysis
sept 2025
e221 - 15-26 redo
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
animal = 'e222'
src = r"Z:\pavlov_extinction\cs_4s_us"
# src=r'Y:\halo_grabda'
src = os.path.join(src,animal)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,f"pavlov4s_{animal}.pdf"))
#e222
# days = [ 1, 2,3,4,  5,  6,  7,  8,  9,10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 26, 27, 28,29, 30, 31]#np.arange(11,27)
#e221
days = [31]
# days=[1,2,4,6,7,9,10,12,13,14,16,17,18] #e220
range_val=15; binsize=0.2
close=True
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
# False = True # print out per day figs
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
      # old
      dff = np.hstack(params['params'][0][0][7][0][0])/np.nanmean(np.hstack(params['params'][0][0][7][0][0]))
      if np.nanmax(dff)>1.1:          
         dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))
      
      # plt.close(fig)
      dffdf = pd.DataFrame({'dff': dff})
      dff = np.hstack(dffdf.rolling(10).mean().values)
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
         rewdFF = perireward_binned_activity(dff[mask], rewards[mask], 
               timedFF[mask], 
               range_val, binsize)
      _, meanvel, __, vel = perireward_binned_activity(velocity[mask], rewards[mask], 
         timedFF[mask], 
         range_val, binsize)
      _, meanlick, __, licktr = perireward_binned_activity(licks[mask], rewards[mask], 
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
      ax.imshow(clean_arr)
      tick=20
      ticklbl=4
      ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,tick))
      ax.set_xticklabels(range(-range_val, range_val+1, ticklbl))
      ax.set_title('Correct Trials')
      ax.axvline(int(range_val/binsize),linestyle='--',color='w')
      ax.set_ylabel('Trial #')
      ax = axes[3]
      ax.plot(meanrewdFF)   
      xmin,xmax = ax.get_xlim()     
      ax.fill_between(range(0,int(range_val/binsize)*2), 
               meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
               meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
      ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,tick))
      ax.set_xticklabels(range(-range_val, range_val+1, ticklbl))
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
      ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,tick))
      ax.set_xticklabels(range(-range_val, range_val+1, ticklbl))
      ax.axvline(int(range_val/binsize),linestyle='--',color='k')
      ax.spines[['top','right']].set_visible(False)
      ax.set_ylabel('Velocity (cm/s)')
      ax2.set_ylabel('Licks')
      ax.set_xlabel('Time from CS (s)')
      fig.suptitle(f'Peri CS, {animal}, Day {day}, {layer}')        
      fig.tight_layout()
      
      if close:
         pdf.savefig(fig)  
         plt.close('all') 
      else: 
         plt.show()    
      plndff.append(clean_arr)
   day_date_dff[str(day)] = [plndff,vel,licktr]
if close:
   pdf.close()
#%%
# across days heatmap
alltr = [[np.nanmean(v[0][i],axis=0) for k,v in day_date_dff.items() if len(v[0])==4] for i in range(4)]
velalltr = [np.nanmean(v[1],axis=1) for k,v in day_date_dff.items()]
lickalltr = [np.nanmean(v[2],axis=1)  for k,v in day_date_dff.items()]

from mpl_toolkits.axes_grid1 import make_axes_locatable

time_start, time_end = -5, 15.0   # desired display window (s)
vline1, vline2 = 0.0, 4.0           # vertical lines to draw
vmax=1.007
fig, axes = plt.subplots(nrows=4, ncols=4,sharex=True, figsize=(10,8), sharey='row')
for pln in range(4):


   # ---- helper to crop columns to [time_start, time_end] ----
   def crop_to_time_window(arr, time_start, time_end, range_val):
      """
      arr: (n_rows, n_bins)
      range_val: original peri window half-width used when binning (so arr maps to [-range_val, +range_val])
      returns: cropped_arr, start_idx, end_idx
      """
      n_bins = arr.shape[1]
      orig_time = np.linspace(-range_val, range_val, n_bins)   # original bin centers
      start_idx = np.searchsorted(orig_time, time_start, side='left')
      end_idx = np.searchsorted(orig_time, time_end, side='right') - 1
      start_idx = max(0, start_idx)
      end_idx = min(n_bins-1, end_idx)
      if end_idx <= start_idx:
         raise ValueError(f"Bad crop: start {start_idx}, end {end_idx}, n_bins {n_bins}, orig_time[0]={orig_time[0]:.2f}, orig_time[-1]={orig_time[-1]:.2f}")
      cropped = arr[:, start_idx:(end_idx+1)]
      return cropped, start_idx, end_idx

   # --- ΔF/F raster (top) ---
   ax = axes[0,pln]
   arr = np.array(alltr[pln])         # shape: (n_days / trials, n_bins)
   cropped, sidx, eidx = crop_to_time_window(arr, time_start, time_end, range_val)
   # show with extent mapping columns -> time_start..time_end
   im = ax.imshow(cropped, aspect='auto', cmap='viridis',
                  extent=[time_start, time_end, 0, cropped.shape[0]])
   if pln==0: ax.set_ylabel('Days')
   ax.set_title(f'Plane {planelut[pln]}')
   ax.axvline(vline1, linestyle='--', color='k')
   ax.axvline(vline2, color='k')
   div = make_axes_locatable(ax)
   cax = div.append_axes("right", size="5%", pad=0.05)
   fig.colorbar(im, cax=cax)
   # optional: print crop info
   # print(f"Plane {pln}: ΔF/F cropped cols {sidx}:{eidx+1} -> {cropped.shape[1]} bins")

   # --- licks raster (middle 1) ---
   ax = axes[1,pln]
   arr = np.array(lickalltr)
   cropped, sidx, eidx = crop_to_time_window(arr, time_start, time_end, range_val)
   im = ax.imshow(cropped, aspect='auto', cmap='Blues',
                  extent=[time_start, time_end, 0, cropped.shape[0]])
   ax.axvline(vline1, linestyle='--', color='k')
   ax.axvline(vline2, color='k')
   div = make_axes_locatable(ax)
   cax = div.append_axes("right", size="5%", pad=0.05)
   fig.colorbar(im, cax=cax,label='Licks')

   # --- velocity raster (middle 2) ---
   ax = axes[2,pln]
   arr = np.array(velalltr)   # or whichever vel array you use
   cropped, sidx, eidx = crop_to_time_window(arr, time_start, time_end, range_val)
   im = ax.imshow(cropped, aspect='auto', cmap='Greys',
                  extent=[time_start, time_end, 0, cropped.shape[0]])
   ax.axvline(vline1, linestyle='--', color='k')
   ax.axvline(vline2, color='k')
   div = make_axes_locatable(ax)
   cax = div.append_axes("right", size="5%", pad=0.05)
   fig.colorbar(im, cax=cax,label='Velocity')

   # --- mean trace (bottom) ---
   ax = axes[3,pln]
   # compute mean from the SAME columns used for the ΔF/F raster above
   # (use alltr[pln] and sidx/eidx from that crop)
   mean_trace = np.nanmean(np.array(alltr[pln])[:, sidx:(eidx+1)][-5:], axis=0)  # last 10 days as before
   time_axis = np.linspace(time_start, time_end, mean_trace.size)
   ax.plot(time_axis, mean_trace)
   ax.axvline(vline1, linestyle='--', color='k')
   ax.axvline(vline2, color='k')
   ax.set_xlim(time_start, time_end)
   div = make_axes_locatable(ax)
   cax = div.append_axes("right", size="5%", pad=0.05)
   fig.colorbar(im, cax=cax)
   ax.set_title('Last 5 day average')
   ax.set_ylim([.995,vmax])
fig.suptitle(f'{animal}')
fig.tight_layout()

