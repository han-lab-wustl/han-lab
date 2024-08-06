#%%
from scipy.io import loadmat
import os, scipy, sys
import glob, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye

# Definitions and setups
mice = ["e216"]#, "e217", "e218"]
dys_s = [[37,38,39,40,41]]#, [14, 26, 27], [35, 38, 41, 44, 47, 50]]
dys_s = [[43,44,45,48]]
opto_ep_s = [[2,2,2,2]]#, [2, 3, 3], [3, 2, 3, 2, 3, 2]]
cells_to_plot_s = [[466,159,423,200,299]]#, [16,6,9], 
cells_to_plot_s = [[2352,1804,751,2231]]#, [16,6,9], 
        # [[453,63,26,38], [301, 17, 13, 320], [17, 23, 36, 10], 
        # [6, 114, 11, 24], [49, 47, 6, 37], [434,19,77,5]]]
src = "X:/vipcre"
dffs_cp_dys = []
mind = 0
# Define the number of bins and the size of each bin
nbins = 90
bin_size = 3                
binsize = 0.1 # s
range_val = 8
# Processing loop
for m, mouse_name in enumerate(mice):
        days = dys_s[m]
        cells_to_plot = cells_to_plot_s[m]
        opto_ep = opto_ep_s[m]
        dyind = 0
        for dy in days:
                daypath = glob.glob(os.path.join(src, mouse_name, str(dy), '**/*Fall.mat'), recursive=True)[0]
                data = loadmat(daypath, variable_names=['dFF', 'forwardvel', 'ybinned', 'VR',
                'timedFF', 'changeRewLoc', 'rewards', 'licks', 'trialnum'])
                dFF = data['dFF']
                changeRewLoc = data['changeRewLoc'].flatten()
                VR = data['VR']
                ybinned = data['ybinned'].flatten()
                forwardvel = data['forwardvel'].flatten()
                timedFF = data['timedFF'].flatten()
                rewards = np.hstack(data['rewards'])==1
                licks = data['licks']
                trialnum = data['trialnum'].flatten()
                print(daypath)

                # Additional processing
                eps = np.where(changeRewLoc > 0)[0]
                eps = np.append(eps, len(changeRewLoc))
                gainf = 1 / VR['scalingFACTOR'].item()
                rewloc = np.hstack(changeRewLoc[changeRewLoc > 0] * gainf)
                rewsize = VR['settings'][0][0][0][0][4] * gainf # reward zone is 5th element
                ypos = np.hstack(ybinned * gainf)
                velocity = forwardvel
                dffs_cp = []
                indtemp = 0

                rngopto = range(eps[opto_ep[dyind] - 1], eps[opto_ep[dyind]])
                rngpreopto = range(eps[opto_ep[dyind] - 2], eps[opto_ep[dyind] - 1])
                # normalize to rew loc?
                yposopto = ypos[rngopto]
                ypospreopto = ypos[rngpreopto]
                # mask for activity before reward loc
                # yposoptomask = np.hstack(yposopto < rewloc[opto_ep[dyind] - 1] - rewsize-10)
                # ypospreoptomask = np.hstack(ypospreopto < rewloc[opto_ep[dyind] - 2] - rewsize-10)
                # yposoptomask = np.hstack(yposopto > rewloc[opto_ep[dyind] - 1] - rewsize-10)
                # ypospreoptomask = np.hstack(ypospreopto > rewloc[opto_ep[dyind] - 2] - rewsize-10)
                # get entire tuning curve
                yposoptomask = (np.ones_like(np.hstack(yposopto))*True).astype(bool)
                ypospreoptomask = (np.ones_like(np.hstack(ypospreopto))*True).astype(bool)
                trialoptomask = trialnum[rngopto] > 10
                trialpreoptomask = trialnum[rngpreopto] > 10
                cp = cells_to_plot[dyind] # just get 1 cell        
                # try:
                #     if len(cp)>1:
                #         cp = cp[0]
                # except:
                #     print('e')
                dffopto = dFF[rngopto, :]
                dffpreopto = dFF[rngpreopto, :]
                dffs_cp.append([np.nanmean(dffopto[:, [cp]],axis=1), 
                                np.nanmean(dffpreopto[:, [cp]],axis=1)])
                # Initialize arrays for tuning curves
                # opto_tuning = np.ones(nbins)*np.nan
                # prevep_tuning = np.ones(nbins)*np.nan

                # Extract dFF arrays for the corresponding conditions
                optodff = np.nanmean(dffopto[:, [cp]],axis=1)
                prevepdff = np.nanmean(dffpreopto[:, [cp]],axis=1)
                # get tuning curve
                # # Process for 'opto' condition
                # # Create an index array from 0 to len(timedFF(rngopto)) - 1
                # time_moving = np.arange(len(timedFF[rngopto]))
                # ypos_mov = yposopto[time_moving]
                # # Filter by the mask conditions (y position and trial number)
                # time_moving = time_moving[yposoptomask & trialoptomask]
                # ypos_mov = ypos_mov[yposoptomask & trialoptomask]

                # # Bin the data
                # time_in_bin_opto = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

                # # Process for 'pre opto' condition
                # time_moving = np.arange(len(timedFF[rngpreopto]))
                # ypos_mov = ypospreopto[time_moving]
                # time_moving = time_moving[ypospreoptomask & trialpreoptomask]
                # ypos_mov = ypos_mov[ypospreoptomask & trialpreoptomask]

                # # Bin the data
                # time_in_bin_pre = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

                # # Compute the mean for each bin and populate the tuning curves
                # for bin_ in range(nbins):
                #     if len(time_in_bin_opto[bin_]) > 0:
                #         opto_tuning[bin_] = np.nanmean(optodff[time_in_bin_opto[bin_]])
                #     if len(time_in_bin_pre[bin_]) > 0:
                #         prevep_tuning[bin_] = np.nanmean(prevepdff[time_in_bin_pre[bin_]])
                # peri reward activity
                normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(optodff, rewards[rngopto], 
                        timedFF[rngopto], range_val, binsize)
                normmeanrewdFF, meanrewdFF_ctrl, normrewdFF, \
                rewdFF_ctrl = eye.perireward_binned_activity(prevepdff, rewards[rngpreopto], 
                        timedFF[rngpreopto], range_val, binsize)

                dffs_cp_dys.append([meanrewdFF_opto, meanrewdFF_ctrl])
                indtemp += 1
                dyind += 1
#%%
# plot tuning curve before and during opto
dffarr = np.array(dffs_cp_dys)
# delete outliers?
# dffarr = np.delete(dffarr,2,0)
# dffarr = np.delete(dffarr,5,0)
dyrng = dffarr.shape[0]
sq = int(np.ceil(np.sqrt(dyrng)))
fig, axes = plt.subplots(nrows=sq,ncols=sq, figsize=(14,9))
r=0;c=0
for ii,dy in enumerate(range(dyrng)):
        ax = axes[r,c]
        meantc = dffarr[dy,1,:]
        ax.plot(meantc, color='k')   
        xmin,xmax = ax.get_xlim()     
        # ax.fill_between(np.arange(0,(range_val*2)/binsize), 
        #         meantc-scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'),
        #         meantc+scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'), 
        #         color = 'k', alpha=0.2)        
        ax.set_xlabel('Time from reward (s)')
        ax.set_ylabel('dF/F')
        ax.axvline(int(range_val/binsize), color='slategray', linestyle='--')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(f'Day {dy+1}, e216')
        if r<int(np.ceil(np.sqrt(dyrng)))-1: r+=1
        else: c+=1; r=0
fig.tight_layout()
savedst = r'C:\Users\Han\Box\Han_lab_dropbox\UndergradRA\Maggie\2p_images\figures_from_zahra'
plt.savefig(os.path.join(savedst, 'vip_cck_per_day.svg'), bbox_inches='tight')

#%%
# plot overlay of days
fig, ax = plt.subplots(figsize=(8,5))

for ii,dy in enumerate(range(dyrng)):
        meantc = dffarr[dy,1,:]
        ax.plot(meantc, label=f'Day {dy+1}')   
xmin,xmax = ax.get_xlim()     
# ax.fill_between(np.arange(0,(range_val*2)/binsize), 
#         meantc-scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'),
#         meantc+scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'), 
#         color = 'k', alpha=0.2)        
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel('dF/F')
ax.legend(bbox_to_anchor=(1.00, 1.00))
ax.axvline(int(range_val/binsize), color='slategray', linestyle='--')
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 1))
ax.spines[['top','right']].set_visible(False)
ax.set_title(f'e216')        
fig.tight_layout()

plt.savefig(os.path.join(savedst, 'vip_cck_across_days.svg'), bbox_inches='tight')
# %%
