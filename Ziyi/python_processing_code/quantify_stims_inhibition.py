"""zahra
sept 2024
halo power tests
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\HanLab\Documents\GitHub\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches

# plt.rc('font', size=12)          # controls default text sizes

plt.close('all')
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

# src = r"Z:\halo_grabda"
srcs = [r"\\storage1.ris.wustl.edu\ebhan\Active\Ziyi\Shared_Data\VTA_mice\E277\250103_ZH_000_000"]
# animals = ['e241']
# days_all = [[1]]

range_val = 8; binsize=0.2 #s
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
#%%
day_date_dff = {}
for src in srcs:
    print(src)
    # for each plane
    stimspth = list(Path(src).rglob('*000*.mat'))[0]
    stims = scipy.io.loadmat(stimspth)
    stims = np.hstack(stims['stims']) # nan out stims
    for path in Path(src).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        VR = params['VR'][0][0]; gainf = VR[14][0][0]             
        #planenum = os.path.basename(os.path.dirname(path))
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 6 - roibasemean3/average
        # raw in row 7
        row = 6
        dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])


        timedFF = np.hstack(params['timedFF'])
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
            rewdFF= eye.perireward_binned_activity(dff, startofstims, timedFF, 
                                    range_val, binsize)
        fig, ax = plt.subplots()
        ax.plot(meanrewdFF, color = 'slategray')   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
        meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
        meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
        color='slategray',alpha=0.4)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))

        ax.set_title(f'Peri-stim, 280mA, plane {pln} \n {src}')

        # day_date_dff[str(day)] = plndff
