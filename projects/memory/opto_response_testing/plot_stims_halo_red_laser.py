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
from projects.memory.behavior import consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from projects.memory.dopamine import get_rewzones

# plt.rc('font', size=12)          # controls default text sizes

plt.close('all')
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'Y:\halo_grabda'
animals = ['e243']
days_all = [[8,9,10,13]]

range_val = 8; binsize=0.2 #s
dur=3# s stim duration
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
prewin = 3 # for which to normalize
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        # stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        # stims = scipy.io.loadmat(stimspth)
        # stims = np.hstack(stims['stims']) # nan out stims
        plndff = []
        fig,axes=plt.subplots(nrows=3, ncols=4, figsize=(12,6))

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
            row =  6
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])            
            # plot mean img
            ax=axes[0,pln]
            ax.imshow(params['params'][0][0][0],cmap='Greys_r')
            ax.axis('off')
            ax.set_title(f'{animal}, day {day}, {planelut[pln]}')
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
            ax=axes[1,pln]
            ax.plot(dff-1,label=f'plane: {pln}')
            ax.plot(startofstims-1)
            ax.set_ylim([-.1,.1])
            ax.set_title(f'Stim events')
            # peri stim binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            
            binss = np.ceil(prewin/binsize).astype(int)
            bound = int(range_val/binsize)
            #normalize
            meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[(bound-binss):bound])
            rewdFF = np.array([rewdFF[:,tr]-np.nanmean(meanrewdFF[(bound-binss):bound]) \
                for tr in range(rewdFF.shape[1])]).T

            ax=axes[2,pln]
            ax.plot(meanrewdFF, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),            
            color='k',alpha=0.4)
            ax.set_ylim([-0.02,0.02])
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

# plot all trials
slm = np.hstack([v[0] for k,v in day_date_dff.items()][:-1])
sr = np.hstack([v[1] for k,v in day_date_dff.items()])
sp = np.hstack([v[2] for k,v in day_date_dff.items()])
so = np.hstack([v[3] for k,v in day_date_dff.items()])

fig, axes = plt.subplots(nrows=4,figsize=(3,7),sharex=True,sharey=True)
allplns = [so,sp,sr,slm]
lbls=['SO', 'SP', 'SR', 'SLM']
for nm,pl in enumerate(allplns):
    ax=axes[nm]
    m=np.nanmean(pl,axis=1)
    # pre window sub
    m=m-np.nanmean(m[:15])
    sem=scipy.stats.sem(pl,axis=1,nan_policy='omit')
    ax.plot(m)
    ax.fill_between(np.arange(len(m)),m-sem,m+sem,alpha=0.2)
    ax.set_title(lbls[nm])
    ymin=-.01
    ymax=.02
    ax.add_patch(
    patches.Rectangle(
            xy=(range_val/binsize,ymin),  # point of origin.
            width=dur/binsize, height=ymax, linewidth=1, # width is s
            color='lightcoral', alpha=0.2))
    ax.axhline(0, color='grey',linestyle='--')
fig.suptitle('SNc halo, ~40mA, n=1')