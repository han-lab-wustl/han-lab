"""zahra
sept 2024
halo power tests
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from behavior import get_success_failure_trials, consecutive_stretch
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

src = r"Z:\halo_grabda"
animals = ['e241']
days_all = [[2]]

range_val = 5; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}

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
            row = 7
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
            offpln=pln+1 if pln<3 else pln-1
            startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
            min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            startofstims = np.zeros_like(dff)
            startofstims[min_iind]=1
            fig,ax=plt.subplots()
            ax.plot(dff,label=f'plane: {pln}')
            ax.plot(startofstims)
            ax.legend()

            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            fig, ax = plt.subplots()
            ax.plot(meanrewdFF, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            color='k',alpha=0.4)
            ymin=min(meanrewdFF)-.02
            ymax=max(meanrewdFF)+.02-ymin
            ax.add_patch(
                patches.Rectangle(
            xy=(range_val/binsize,ymin),  # point of origin.
            width=2/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim, {animal}, 280mA, plane {pln}')
            plndff.append(rewdFF)
    
        day_date_dff[str(day)] = plndff

#%%
# subtract stims from drug condition
condition = ['no drug', 'drug']

ii=0; pln=0
fig, ax = plt.subplots()
for dy,v in day_date_dff.items():
    rewdFF = day_date_dff[dy][pln] # so only
    meanrewdFF = np.nanmean(rewdFF,axis=1)
    meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[15:25]) #pre-window
    rewdFF_prewin = np.array([xx-np.nanmean(xx[15:25]) for xx in rewdFF.T]).T
    ax.plot(meanrewdFF, label=condition[ii])   
    xmin,xmax = ax.get_xlim()     
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meanrewdFF-scipy.stats.sem(rewdFF_prewin,axis=1,nan_policy='omit'),
    meanrewdFF+scipy.stats.sem(rewdFF_prewin,axis=1,nan_policy='omit'),
    alpha=0.4)
    ymin=min(meanrewdFF)-.02
    ymax=max(meanrewdFF)+.02-ymin
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=2/binsize, height=ymax, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    ii+=1
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(range(-range_val, range_val+1, 1))
ax.set_title(f'Peri-stim, {animal}, 280mA, plane {pln}')
ax.legend()