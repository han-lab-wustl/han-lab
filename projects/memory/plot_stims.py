"""zahra
sept 2024
opn3/halo power tests
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from behavior import consecutive_stretch
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
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

# src = r"Y:\opto_control_grabda_2m"
src = r'Y:\halo_grabda'
animals = ['e243']
days_all = [[5]]
range_val = 5; binsize=0.2 #s
dur=1.6# s stim duration
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}

day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        plndff = []
        fig,axes=plt.subplots(nrows=2, ncols=4, figsize=(12,5))
        for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
            params = scipy.io.loadmat(path)
            VR = params['VR'][0][0]; gainf = VR[14][0][0]             
            timedFF = np.hstack(params['timedFF'])
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 6 - roibasemean3/average
            # raw in row 7
            row = 6
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])            
            # nan out stims
            # dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')
            # plt.legend()
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)
            # get off plane stim
            # offpln=pln+1 if pln<3 else pln-1
            # startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
            # min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            # # remove rewarded stims
            # cs=params['solenoid2'][0]
            # # cs within 50 frames of start of stim - remove
            # framelim=20
            # unrewstimidx = [idx for idx in min_iind if sum(cs[idx-framelim:idx+framelim])==0]            
            # startofstims = np.zeros_like(dff)
            # startofstims[unrewstimidx]=1
            # # get on plane stim for red laser
            # offpln=pln
            # ss = consecutive_stretch(np.where(stims[offpln::4])[0])
            # min_iind = [min(xx) for xx in ss if len(xx)>0]
            # # remove rewarded stims
            # cs=params['solenoid2'][0]
            # # cs within 50 frames of start of stim - remove
            # framelim=20
            # unrewstimidx = [idx for idx in min_iind if sum(cs[idx-framelim:idx+framelim])==0]            
            # startofstims[unrewstimidx]=1
            startofstims=params['optoEvent'][0]
            ax=axes[0,pln]
            ax.plot(dff-1,label=f'plane: {pln}')
            ax.plot(startofstims-1)
            ax.set_ylim([-.1,.1])
            ax.set_title(f'Stim events, {animal}, day {day}, {planelut[pln]}')
            # peri stim binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            prewin = 4
            binss = np.ceil(prewin/binsize).astype(int)
            bound = int(range_val/binsize)
            #normalize
            meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[(bound-binss):bound])
            rewdFF = np.array([rewdFF[:,tr]-np.nanmean(meanrewdFF[(bound-binss):bound]) \
                for tr in range(rewdFF.shape[1])]).T

            ax=axes[1,pln]
            ax.plot(meanrewdFF, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),            
            color='k',alpha=0.4)
            ax.set_ylim([-0.03,0.03])
            ax.axhline(0, color='k', linestyle='--')
            ax.axhline(-.01, color='k', linestyle='--')
            ymin=min(meanrewdFF)-.02
            ymax=max(meanrewdFF)+.02-ymin
            ax.add_patch(
                patches.Rectangle(
            xy=(range_val/binsize,ymin),  # point of origin.
            width=dur/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim')
            plndff.append(rewdFF)
            fig.tight_layout()
            # plt.show()            
    
        day_date_dff[str(day)] = plndff

#%%

# power tests
condition = [200,80,280]
condition_org = [80,200,280,280,280]
condition_org = [80,200,280,280]
condition_col = {280:'k', 200:'slategray',80:'darkcyan'}
stimsec = 5 # stim duration (s)
ymin=-0.04
ymax=0.03-(ymin)
planes=4
# assumes 4 planes
fig, axes = plt.subplots(nrows=4, figsize=(4,7), sharex=True)
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []; condition = condition_org.copy() # custom condition
    for dy,v in day_date_dff.items():
        rewdFF = day_date_dff[dy][pln] # so only
        if rewdFF.shape[1]>0:            
            meanrewdFF = np.nanmean(rewdFF,axis=1)
            meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[20:25]) #pre-window
            rewdFF_prewin = np.array([xx-np.nanmean(xx[20:25]) for xx in rewdFF.T]).T
            condition_dff.append([meanrewdFF, rewdFF_prewin])
        else: idx_to_catch.append(int(dy))
    # remove 0 trial days from condition vector
    if len(idx_to_catch)>0: [condition.pop(np.where(np.array(days)==idx)[0][0]) for idx in idx_to_catch]
    ax = axes[pln]
    meanrewdFF = np.vstack([x[0] for x in condition_dff])
    rewdFF = [x[1] for x in condition_dff]
    # plot per condition
    for cond in np.unique(condition):
        meancond = np.nanmean(meanrewdFF[condition==cond],axis=0)
        ax.plot(meancond, label=cond, color=condition_col[cond])   
        xmin,xmax = ax.get_xlim() 
        trialcond = np.concatenate([[condition[ii]]*xx.shape[1] for ii,xx in enumerate(rewdFF)])
        rewcond = np.hstack(rewdFF).T[trialcond==cond].T
        ax.fill_between(range(0,int(range_val/binsize)*2), 
        meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.4,color=condition_col[cond])        
    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=ymax, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    ii+=1
    ax.set_title(f'\nPlane {planelut[pln]}')
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
ax.legend(bbox_to_anchor=(1.1, 1.05))
fig.tight_layout()
fig.suptitle(f'{animal}, Per day plots')