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
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
plt.rcParams["font.family"] = "Arial"

#%%
plt.close('all')
# save to pdf
pth = r"X:\da_hrz.csv"
csv = pd.read_csv(pth,index_col=None)
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
range_val=6; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
animals = np.unique(csv.animal.values)
# figs = True # print out per day figs
old=False
ctrl_an = ['e170', 'e171', 'e179',
        't10', 't11']
day_date_dff = {}
for animal in animals: 
   # last 4 days
   if animal in ctrl_an:
      dys = csv[csv.animal==animal]
   else: 
      dys = csv[csv.animal==animal][-4:]
   if animal=='t10' or animal=='t11': old=True
   else: old=False
   for dy in range(len(dys)):
      df = dys.iloc[dy]
      fl = str(df.file)
      # for each plane
      plndff=[];plnvel=[]
      for path in Path(fl).rglob('params.mat'):
         params = scipy.io.loadmat(path)
         try:
            VR = params['VR'][0][0][()]
            gainf = VR['scalingFACTOR'][0][0]
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/gainf
         except:
            gainf=1
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
         if np.sum(rewards)==0:
            # get us
            rewards = np.hstack(params['rewards'])
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
      
            ax.set_title(f'Behavior, {animal}, {os.path.basename(fl)}')
            ax.set_ylabel('Position (cm)')
            ax.set_xticks(np.arange(0,len(timedFF)+1000,1000))
            ax.set_xticklabels(np.round(np.append(timedFF[::1000]/60,timedFF[-1]/60), 1))
            ax.set_xlabel('Time (minutes)')
            fig.tight_layout()
         # plot raw traces:
         
         # plt.figure(); plt.plot(dff)

         #TODO: peri reward fails 
         #TODO: peri reward catch trials
         # all subsequent rews   
         # only ep3?
         # aligned to CS
         changeRewLoc = np.hstack(params['changeRewLoc'])     
         scalingf=2/3
         eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))        
         mask = np.arange(0,eps[len(eps)-1])
         rewards[:1000]=0
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
         ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)
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
         fig.suptitle(f'Peri CS, {animal}, {os.path.basename(fl)}, {layer}')        
         fig.tight_layout()
         plt.close('all')
         # pdf.savefig(fig)        
         plndff.append(clean_arr)
         plnvel.append(vel)
      day_date_dff[f'{animal}_{df.type}_{dy}'] = [plndff,plnvel]
# pdf.close()

#%%
ans = csv.animal.unique()
ans = ['e156', 'e157', 'e231', 'e232', 'e243','e171',
      'e170', 'e179', 't10', 't11']
# heatmap across days
alltr = [np.concatenate([v[0][i] for k,v in day_date_dff.items() if len(v[0])==4 and 'ctrl' in k and k.split('_')[0] in ans]) for i in range(4)]
# all trials
for pln in range(4): 
    fig, axes = plt.subplots(ncols=2,width_ratios=[1,1.5],sharex=True,figsize=(6,3))
    ax=axes[0]
    cax=ax.imshow(alltr[pln][:,:],aspect='auto')    
    ax.set_xlabel('Time from CS (s)')
    ax.set_ylabel('Trials (last 4 days)')
    ax.axvline(int(range_val/binsize),linestyle='--',color='w')
    # ax.set_yticks(range(0,pln_mean[:,pln,:].shape[0],2))
    ax.set_title(f'Plane {planelut[pln]}')
    fig.colorbar(cax,ax=ax,fraction=0.01, pad=0.04)
    ax=axes[1]
    mf = np.nanmean(alltr[pln][:,:],axis=0)
    ax.plot(mf)    
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    mf-scipy.stats.sem(alltr[pln][:,:],axis=0,nan_policy='omit'),
    mf+scipy.stats.sem(alltr[pln][:,:],axis=0,nan_policy='omit'), alpha=0.3)
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize),linestyle='--',color='k')
    fig.tight_layout()
    
    
#%%
def rgba_hex_to_rgb(rgba_hex):
  """Converts an 8-digit RGBA hex string to an RGB tuple."""
  rgba_hex = rgba_hex.lstrip('#')  # Remove '#' if present
  r_hex = rgba_hex[0:2]
  g_hex = rgba_hex[2:4]
  b_hex = rgba_hex[4:6]
  
  r = int(r_hex, 16)
  g = int(g_hex, 16)
  b = int(b_hex, 16)
  
  return np.array([r, g, b])/255

color_code = 'e65480ff'
rgb_color = rgba_hex_to_rgb(color_code)
print(rgb_color)
#%%
# for poster
plt.rc('font', size=12)
alltr = [np.concatenate([v[0][i] for k,v in day_date_dff.items() if len(v[0])==4 and 'ctrl' not in k and k.split('_')[0] in ans]) for i in range(4)]
# all trials
fig, axes = plt.subplots(ncols=2,nrows=5,sharex=True,sharey='row',figsize=(5,7),height_ratios=[2,2,2,2,1])
normsec = 3
colors=['e65480ff','cba33dff','008000b3','0000ffff']
colors = [rgba_hex_to_rgb(xx) for xx in colors]
colors=np.array(colors)[::-1]
vmin=-.5
vmax=1

for nm,pln in enumerate([3,2,1,0]):     
   ax=axes[nm,0]
   trace=alltr[pln][:,:]
   if pln==3:
      trace=np.array([xx-np.nanmean(xx[int(normsec/binsize):int(range_val/binsize)]) for xx in trace])
   else:
      trace=np.array([xx-np.nanmean(xx[:int(normsec/binsize)]) for xx in trace])
   trace=trace*100
   mf = np.nanmean(trace,axis=0)
   ax.plot(mf,color=colors[pln])    
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(trace,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(trace,axis=0,nan_policy='omit'), alpha=0.5,color=colors[pln])
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
   ax.set_xticklabels(range(-range_val, range_val+1, 2))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   if nm==0:
      ax.set_title(f'GRABDA\n{planelut[pln]}\nn=5 mice')
      ax.set_ylabel('% $\Delta F/F$')
      ax.set_ylim([vmin,vmax])
   else: 
      ax.set_title(f'{planelut[pln]}')
      ax.set_ylim([-.2,1])
   ax.spines[['top','right']].set_visible(False)   

alltr = [np.concatenate([v[0][i] for k,v in day_date_dff.items() if len(v[0])==4 and 'ctrl' in k and k.split('_')[0] in ans]) for i in range(4)]
for nm,pln in enumerate([3,2,1,0]):     
   ax=axes[nm,1]
   trace=alltr[pln][:,:]
   if pln==3:
      trace=np.array([xx-np.nanmean(xx[int(normsec/binsize):int(range_val/binsize)]) for xx in trace])
   else:
      trace=np.array([xx-np.nanmean(xx[:int(normsec/binsize)]) for xx in trace if np.nanmax(xx)<1.017]) # HACK!!!
   trace=trace*100
   mf = np.nanmean(trace,axis=0)
   ax.plot(mf,color=colors[pln])    
   ax.fill_between(range(0,int(range_val/binsize)*2), 
   mf-scipy.stats.sem(trace,axis=0,nan_policy='omit'),
   mf+scipy.stats.sem(trace,axis=0,nan_policy='omit'), alpha=0.5,color=colors[pln])
   ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
   ax.set_xticklabels(range(-range_val, range_val+1, 2))
   ax.axvline(int(range_val/binsize),linestyle='--',color='k')
   if nm==0:
      ax.set_title(f'GRABDA-mut\n{planelut[pln]}\nn=4 mice')
   else: 
      ax.set_title(f'{planelut[pln]}')
   ax.spines[['top','right']].set_visible(False)
   if nm==3:
      ax.set_xlabel('Time from CS (s)')
# vel
ctrlvel = np.hstack([v[1][0] for k,v in day_date_dff.items() if len(v[0])==4 and 'ctrl' in k and k.split('_')[0] in ans]).T
davel = np.hstack([v[1][0] for k,v in day_date_dff.items() if len(v[0])==4 and 'ctrl' not in k and k.split('_')[0] in ans]).T

ax=axes[4,0]
trace=davel
mf = np.nanmean(trace,axis=0)
ax.plot(mf,color='k')    
ax.fill_between(range(0,int(range_val/binsize)*2), 
mf-scipy.stats.sem(trace,axis=0,nan_policy='omit'),
mf+scipy.stats.sem(trace,axis=0,nan_policy='omit'), alpha=0.5,color='k')
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
ax.axvline(int(range_val/binsize),linestyle='--',color='k')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Velocity (cm/s)')

ax=axes[4,1]
trace=ctrlvel
mf = np.nanmean(trace,axis=0)
ax.plot(mf,color='k')    
ax.fill_between(range(0,int(range_val/binsize)*2), 
mf-scipy.stats.sem(trace,axis=0,nan_policy='omit'),
mf+scipy.stats.sem(trace,axis=0,nan_policy='omit'), alpha=0.5,color='k')
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
ax.axvline(int(range_val/binsize),linestyle='--',color='k')
ax.spines[['top','right']].set_visible(False)

fig.tight_layout()
plt.savefig(os.path.join(dst, 'traces_grc.svg'), bbox_inches='tight')


# %%
