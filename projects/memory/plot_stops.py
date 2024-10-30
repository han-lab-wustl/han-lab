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
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops

# plt.rc('font', size=12)          # controls default text sizes
#%%

plt.close('all')
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'Z:\chr2_grabda'
animals = ['e232']
days_all = [[42]]
range_val = 8; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
prewin = 2 # for which to normalize
planes=4
frames_stopped = 31 # number of frames (in the full frame rate) when animal is stopped
velocity_thres = 5 # cm/s, velocity below which considered stop
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        plndff = []
        fig,axes=plt.subplots(nrows=4, ncols=4, figsize=(12,6))

        for path in list(Path(os.path.join(src, animal, str(day))).rglob('params.mat')):
            print(path)
            params = scipy.io.loadmat(path)

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
            dff = np.hstack(dffdf.rolling(5).mean().values)
            # find stops
            velocity = params['forwardvelALL'][0]
            veldf = pd.DataFrame({'velocity': velocity})
            velocity = np.hstack(veldf.rolling(5).mean().values)
            moving_middle,stop = get_moving_time_v3(velocity,velocity_thres,
                                    frames_stopped,10)
            pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
            nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
                    post_win_framesALL,velocity, params['rewardsALL'][0])
            nonrew_per_plane = np.zeros_like(params['changeRewLoc'][0])
            nonrew = np.floor(nonrew/planes)
            nonrew_per_plane[nonrew.astype(int)] = 1

            ax=axes[1,pln]
            ax.plot(dff-1,label=f'plane: {pln}')
            ax.plot(nonrew_per_plane-1)
            ax.set_ylim([-.1,.1])
            ax.set_title(f'Stop events')
            # peri stop binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, nonrew_per_plane, 
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
            # ax.set_ylim([-0.02,0.02])
            ax.axhline(0, color='k', linestyle='--')
            ax.axhline(-.01, color='k', linestyle='--')
            ax.axvline(int(range_val/binsize), color='k', linestyle='--')
            ymin=min(meanrewdFF)-.02
            ymax=max(meanrewdFF)+.02-ymin

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stop (DA)')
            ax.set_ylabel('$\Delta$ F/F')
            
            # peri stop velocity
            normmeanrewdFF, meanv, normrewdFF, \
                rewv = eye.perireward_binned_activity(params['forwardvel'][0], nonrew_per_plane, 
                    timedFF, range_val, binsize)
                
            ax=axes[3,pln]
            ax.plot(meanv, color = 'slategray')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanv-scipy.stats.sem(rewv,axis=1,nan_policy='omit'),
            meanv+scipy.stats.sem(rewv,axis=1,nan_policy='omit'),            
            color='slategray',alpha=0.4)
                        
            ax.axvline(int(range_val/binsize), color='k', linestyle='--')
            ymin=min(meanv)-.02
            ymax=max(meanv)+.02-ymin

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stop')
            ax.set_ylabel('velocity, cm/s')

            plndff.append(rewdFF)
            fig.tight_layout()
            # plt.show()            
    
        day_date_dff[str(day)] = plndff
#%%