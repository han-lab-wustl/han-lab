from scipy.io import loadmat
import os, scipy
import glob, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

# Definitions and setups
mice = ["e216", "e217", "e218"]
dys_s = [[37, 41, 57, 60], [14, 26, 27], [35, 38, 41, 44, 47, 50]]
opto_ep_s = [[2, 3, 2, 3], [2, 3, 3], [3, 2, 3, 2, 3, 2]]
cells_to_plot_s = [[135,1655,780,2356], [16,6,9], 
    [[453,63,26,38], [301, 17, 13, 320], [17, 23, 36, 10], 
    [6, 114, 11, 24], [49, 47, 6, 37], [434,19,77,5]]]
src = "X:/vipcre"
dffs_cp_dys = []
mind = 0

# Processing loop
for m, mouse_name in enumerate(mice):
    days = dys_s[m]
    cells_to_plot = cells_to_plot_s[m]
    opto_ep = opto_ep_s[m]
    dyind = 0
    for dy in days:
        daypath = glob.glob(os.path.join(src, mouse_name, str(dy), '**/*Fall.mat'), recursive=True)[0]
        data = loadmat(daypath)
        dFF = data['dFF']
        changeRewLoc = data['changeRewLoc'].flatten()
        VR = data['VR']
        ybinned = data['ybinned'].flatten()
        forwardvel = data['forwardvel'].flatten()
        timedFF = data['timedFF'].flatten()
        rewards = data['rewards']
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
        yposopto = ypos[rngopto]
        ypospreopto = ypos[rngpreopto]
        yposoptomask = np.hstack(yposopto < rewloc[opto_ep[dyind] - 1] - rewsize-10)
        ypospreoptomask = np.hstack(ypospreopto < rewloc[opto_ep[dyind] - 2] - rewsize-10)
        trialoptomask = trialnum[rngopto] > 10
        trialpreoptomask = trialnum[rngpreopto] > 10
        cp = cells_to_plot[dyind] # just get 1
        try:
            if len(cp)>0:
                cp = cp[0]
        except Exception as e:
            print(e)
        dffopto = dFF[rngopto, :]
        dffpreopto = dFF[rngpreopto, :]
        dffs_cp.append([dffopto[:, cp], dffpreopto[:, cp]])
                
        # Define the number of bins and the size of each bin
        nbins = 90
        bin_size = 3

        # Initialize arrays for tuning curves
        opto_tuning = np.ones(nbins)*np.nan
        prevep_tuning = np.ones(nbins)*np.nan

        # Extract dFF arrays for the corresponding conditions
        optodff = dffs_cp[0][0]
        prevepdff = dffs_cp[0][1]

        # Process for 'opto' condition
        # Create an index array from 0 to len(timedFF(rngopto)) - 1
        time_moving = np.arange(len(timedFF[rngopto]))
        ypos_mov = yposopto[time_moving]
        # Filter by the mask conditions (y position and trial number)
        time_moving = time_moving[yposoptomask & trialoptomask]
        ypos_mov = ypos_mov[yposoptomask & trialoptomask]

        # Bin the data
        time_in_bin_opto = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

        # Process for 'pre opto' condition
        time_moving = np.arange(len(timedFF[rngpreopto]))
        ypos_mov = ypospreopto[time_moving]
        time_moving = time_moving[ypospreoptomask & trialpreoptomask]
        ypos_mov = ypos_mov[ypospreoptomask & trialpreoptomask]

        # Bin the data
        time_in_bin_pre = [time_moving[(ypos_mov >= (i * bin_size)) & (ypos_mov < ((i + 1) * bin_size))] for i in range(nbins)]

        # Compute the mean for each bin and populate the tuning curves
        for bin_ in range(nbins):
            if len(time_in_bin_opto[bin_]) > 0:
                opto_tuning[bin_] = np.nanmean(optodff[time_in_bin_opto[bin_]])
            if len(time_in_bin_pre[bin_]) > 0:
                prevep_tuning[bin_] = np.nanmean(prevepdff[time_in_bin_pre[bin_]])

        dffs_cp_dys.append([prevep_tuning, opto_tuning])
        indtemp += 1
        dyind += 1

#%%
# plot tuning curve before and during opto
dffarr = np.array(dffs_cp_dys)

dffarr = np.delete(dffarr,2,0)
dffarr = np.delete(dffarr,5,0)
meantc = np.nanmean(dffarr[:,0,:],axis=0)
fig, ax = plt.subplots()
ax.plot(meantc, color='k', label='LED off')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,nbins), 
        meantc-scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'),
        meantc+scipy.stats.sem(dffarr[:,0,:],axis=0,nan_policy='omit'), color = 'k', alpha=0.2)        
meantc = np.nanmean(dffarr[:,1,:],axis=0)
ax.plot(meantc, color='r', label='LED on')   
xmin,xmax = ax.get_xlim()     
ax.fill_between(range(0,nbins), 
        meantc-scipy.stats.sem(dffarr[:,1,:],axis=0,nan_policy='omit'),
        meantc+scipy.stats.sem(dffarr[:,1,:],axis=0,nan_policy='omit'), color = 'r', alpha=0.2)        
ax.set_xlabel('Spatial bins')
ax.set_ylabel('dF/F')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'
plt.savefig(os.path.join(savedst, 'vip_during_opto.svg'), bbox_inches='tight')