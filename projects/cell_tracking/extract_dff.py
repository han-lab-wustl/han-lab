# Zahra
# extract dff and cluster
import numpy as np, os
from scipy.io import loadmat
import h5py, scipy
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = scipy.cluster.hierarchy.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


src = r'Y:\sstcre_analysis\celltrack\e201_week4789\Results'
mat = os.path.join(src,'dff_per_day.mat')
f = h5py.File(mat)
# extract from h5py
dff = []
for i in range(len(f['dff'][:])):
    dff.append(f[f['dff'][i][0]][:])
commoncells = os.path.join(src,'commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat')
cc = loadmat(commoncells)['cellmap2dayacrossweeks'].astype(int)
cc=cc-1 # subtract from matlab ind
# load fall example
# day 41
daypth = r'Z:\sstcre_imaging\e201\41\230413_ZD_000_000'
fallpth = os.path.join(daypth,'suite2p', 'plane0', 'Fall.mat')
fall = loadmat(fallpth)
epoch = np.where(fall['changeRewLoc']>1)
for ep in range(len(epoch[1])):
    print(ep)
    epoch_start,epoch_stop = epoch[1][ep],epoch[1][ep+1]
    rewloc = np.unique(fall['changeRewLoc'])[ep+1]
    trials = max(max(fall['trialnum'][:, epoch_start:epoch_stop])) # total number of trials
    dff_av = []
    mask=cc[:,17] # 17 is the dark num
    dff_day = dff[17].T[mask][mask>-1]
    for trial in range(trials-10,trials): #only first epoch, 10 trials
        print(trial)
        if trial > 0:
            trialind = np.where(fall['trialnum']==trial)[1]
            trialind = trialind[epoch_start <= trialind]
            trialind = trialind[trialind <= epoch_stop] # first trial of first epoch?
            # dff structure = each item of list is a day
            # 18 days 
            # days=[14,15,16,17,18,27, 28, 29, 30, 31, 32, 33,36,38,39,40,41]
            # cluster dff 1 day   
            # bin into 500 frames
            dff_trial = dff_day[:,trialind].T
            dff_binned = np.zeros((270,dff_day.T[trialind].shape[1])) # init
            for i in range(dff_trial.shape[1]): # bin per cell
                llen = dff_trial.shape[0] # get frames per trial
                data= np.linspace(1,llen,llen)  
                bins=np.linspace(1,llen,271)     
                dig = np.digitize(data,bins)  
                bin_means = [np.nanmean(dff_trial[:,i][dig == ii]) for ii in range(1,
                            len(bins))]
                dff_binned[:,i]=bin_means
            dff_av.append(dff_binned) #mean across all frames
            print(dff_binned.shape)

    dff_av_arr = np.mean(np.array(dff_av),axis=0).T        
    #https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    Z_columns = scipy.cluster.hierarchy.linkage(dff_av_arr, method='centroid')
    max_d = 7

    fancy_dendrogram(
        Z_columns,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        max_d=max_d # max distance
    )
  
    clusters = scipy.cluster.hierarchy.fcluster(Z_columns, max_d, 
                                                criterion='distance')

    bigmap = np.zeros_like(dff_av_arr)
    for cl in np.unique(clusters):
        heatmap=dff_av_arr[clusters==cl]
        cellids = np.array(range(dff_av_arr.shape[0]))[clusters==cl]
        plt.figure()
        sns.heatmap(heatmap, cmap='viridis',yticklabels=cellids)
        plt.ylabel("# of cells")
        plt.xlabel("track length (cm)")
        plt.title(f"cluster {cl}")
        plt.axvline(x=rewloc,color='white')
        plt.savefig(rf'Y:\sstcre_analysis\clustering\epoch{ep}_cluster{cl}_maxd{max_d}.pdf',
        bbox_inches='tight'
        )

        plt.close()    