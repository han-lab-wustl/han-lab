"""zahra's dopamine hrz analysis
march 2024
"""
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

#%%
plt.close('all')
# save to pdf
src = r"Z:\chr2_grabda\e232"
# src = r"\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\HRZ\E168HRZparams"
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,"hrz.pdf"))
days = []
# days = ['Day_1','Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6',
#         'Day_7', 'Day_8']
days = [30,31,32,33,34,35,36,37,38,39]
range_val = 5; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
old = False
day_date_dff = {}
for day in days: 
    plndff = []
    # for each plane
    for path in Path(os.path.join(src, str(day))).rglob('params.mat'):
        params = scipy.io.loadmat(path)
        gainf = params['VR']
        planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pln = int(planenum[-1])
        layer = planelut[pln]
        params_keys = params.keys()
        keys = params['params'].dtype
        # dff is in row 7 - roibasemean3/basemean
        if old:
            dff = np.hstack(params['params'][0][0][7][0][0])/np.hstack(params['params'][0][0][11])
        else:
            dff = np.hstack(params['params'][0][0][6][0][0])/np.hstack(params['params'][0][0][9])
        # plt.close(fig)
        dffdf = pd.DataFrame({'dff': dff})
        dff = np.hstack(dffdf.rolling(3).mean().values)
        rewards = np.hstack(params['solenoid2'])
        trialnum = np.hstack(params['trialnum'])
        ybinned = np.hstack(params['ybinned'])/(2/3)
        licks = np.hstack(params['licks'])
        # plot pre-first reward dop activity  
        timedFF = np.hstack(params['timedFF'])
        # mask out dark time
        dff = dff[ybinned>3]
        rewards = rewards[ybinned>3]
        trialnum = trialnum[ybinned>3]
        licks = licks[ybinned>3]
        timedFF = timedFF[ybinned>3]
        ybinned = ybinned[ybinned>3]
        # plot behavior
        if pln==0:
            fig, ax = plt.subplots()
            ax.plot(ybinned)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=12)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', marker = '.', s=2)
            ax.set_title(f'Behavior, Day {day}')
            ax.set_ylabel('Position (cm)')
            fig.tight_layout()
            pdf.savefig(fig)

        #TODO: peri reward fails 
        #TODO: peri reward catch trials
        
        mask = (np.ones_like(trialnum)*True).astype(bool)
        # all subsequent rews
        normmeanrewdFF, meanrewdFF, normrewdFF, \
            rewdFF = eye.perireward_binned_activity(dff[mask], rewards[mask], timedFF[mask], 
                                    range_val, binsize)
        # Find the rows that contain NaNs
        # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
        # Select rows that do not contain any NaNs
        clean_arr = rewdFF.T#[~rows_with_nans]    
        fig, axes = plt.subplots(2,1)
        ax = axes[0]
        ax.imshow(clean_arr)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.set_title('Successful Trials (Centered by CS)')
        ax = axes[1]
        ax.plot(meanrewdFF)   
        xmin,xmax = ax.get_xlim()     
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        fig.suptitle(f'Peri CS/Rew Loc, Day {day}, {layer}')
        
        pdf.savefig(fig)

        fig.tight_layout()

        plndff.append(meanrewdFF)
    day_date_dff[str(day)] = plndff
pdf.close()
