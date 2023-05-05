# Zahra
# extract dff and cluster
# TODO: convert to functions and for loop
import numpy as np, os, sys
from scipy.io import loadmat
import h5py, scipy
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import pickle
from sklearn.decomposition import PCA
import math
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import makedir

def plot_som_series_dba_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") # I changed this part
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()

src = r'Y:\sstcre_analysis\celltrack\e201_week4789\Results'
# mat = os.path.join(src,'dff_per_day.mat')
# f = h5py.File(mat)
# # extract from h5py
# dff = [] # takes a long time, if saved extract from pickle
# for i in range(len(f['dff'][:])):
#     dff.append(f[f['dff'][i][0]][:])

# with open(os.path.join(src,"dff_per_day.p"), "wb") as fp:   #Pickling
#    pickle.dump(dff, fp)
with open(os.path.join(src,"dff_per_day.p"), "rb") as fp: #unpickle
    dff = pickle.load(fp)
# need only tracked cells
commoncells = os.path.join(src,'commoncells_atleastoneactivedayperweek_4weeks_week2daymap.mat')
cc = loadmat(commoncells)['cellmap2dayacrossweeks'].astype(int)
cc=cc-1 # subtract from matlab ind
# load fall example
# day 41
# tracked day 17
#%%
day = 41
tracked_day = 17
daypth = rf'Z:\sstcre_imaging\e201\{day}'
daypth = [os.path.join(daypth, xx) for xx in os.listdir(daypth) if "ZD" in xx][0]
fallpth = os.path.join(daypth,'suite2p', 'plane0', 'Fall.mat')
fall = loadmat(fallpth)
epoch = np.where(fall['changeRewLoc']>1)
mask=cc[:,tracked_day] # 17 is the dark num
# get cell ids 
cellids = mask[mask>-1]
dff_day = dff[tracked_day].T#[mask][mask>-1] # only get tracked cells
dst = rf'Y:\sstcre_analysis\clustering\day{day}';makedir(dst)
dst = os.path.join(dst, 'across_epochs');makedir(dst)
# run for each epoch
dff_av = []

for ep in range(len(epoch[1])):
    print(ep)
    try:
        epoch_start,epoch_stop = epoch[1][ep],epoch[1][ep+1]
    except:
        epoch_start,epoch_stop = epoch[1][ep],len(fall['trialnum'][0]) #else set 
        # to end of RECORDING
    rewloc = np.unique(fall['changeRewLoc'])[ep+1]
    trials = max(max(fall['trialnum'][:, epoch_start:epoch_stop])) # total number of trials
    
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
            dff_binned = np.zeros((270,dff_day.T[trialind].shape[1])) # init binning by track length
            for i in range(dff_trial.shape[1]): # bin per cell
                llen = dff_trial.shape[0] # get frames per trial
                # TODO: how to speed this up?
                data= np.linspace(1,llen,llen)  
                bins=np.linspace(1,llen,271)     
                dig = np.digitize(data,bins)  
                bin_means = [np.nanmean(dff_trial[:,i][dig == ii]) for ii in range(1,
                            len(bins))]
                dff_binned[:,i]=bin_means
            dff_av.append(dff_binned) #mean across all frames
            print(dff_binned.shape)

dff_av_arr_ep1 = np.mean(np.array(dff_av)[:10],axis=0).T # gets all cells at this stage        
dff_av_arr_ep2 = np.mean(np.array(dff_av)[10:],axis=0).T
dff_av_arr = np.zeros((dff_av_arr_ep1.shape[0]*2,dff_av_arr_ep1.shape[1]))
dff_av_arr[:dff_av_arr_ep1.shape[0]]=dff_av_arr_ep1
dff_av_arr[dff_av_arr_ep1.shape[0]:]=dff_av_arr_ep2
dff_av_norm=MinMaxScaler().fit_transform(dff_av_arr.T).T 
#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
#https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering
    # make the self organizing maps
som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(dff_av_norm))))
som = MiniSom(som_x, som_y,len(dff_av_norm[0]), sigma=0.3, learning_rate = 0.1)
som.random_weights_init(dff_av_norm)
som.train(dff_av_norm, 50000)
win_map = som.win_map(dff_av_norm)
# Returns the mapping of the winner nodes and inputs

plot_som_series_dba_center(som_x, som_y, win_map)
cluster_c = []
cluster_n = []
for x in range(som_x):
    for y in range(som_y):
        cluster = (x,y)
        if cluster in win_map.keys():
            cluster_c.append(len(win_map[cluster]))
        else:
            cluster_c.append(0)
        cluster_number = x*som_y+y+1
        cluster_n.append(f"Cluster {cluster_number}")

plt.figure(figsize=(25,5))
plt.title("Cluster Distribution for SOM")
plt.bar(cluster_n,cluster_c)
plt.show()
cluster_map = []
for idx in range(len(dff_av_norm)):
    winner_node = som.winner(dff_av_norm[idx])
    cluster_map.append((idx,f"Cluster {winner_node[0]*som_y+winner_node[1]+1}"))

cluster_df = pd.DataFrame(cluster_map,columns=["Series","Cluster"]).sort_values(by="Cluster")
cluster_df.to_csv(os.path.join(dst, f'clusters_all_cells_epoch{ep+1}.csv'), 
                    index = None)
    ##########################HIEARCHICAL CLUSTERING, DID NOT WORK##########################
    #     Z_columns = scipy.cluster.hierarchy.linkage(dff_av_arr, method='centroid')
    #     max_d = 5

    #     fancy_dendrogram(
    #         Z_columns,
    #         truncate_mode='lastp',  # show only the last p merged clusters
    #         p=12,  # show only the last p merged clusters
    #         leaf_rotation=90.,
    #         leaf_font_size=12.,
    #         show_contracted=True,  # to get a distribution impression in truncated branches
    #         max_d=max_d # max distance
    #     )
    
    #     clusters = scipy.cluster.hierarchy.fcluster(Z_columns, max_d, 
    #                                                 criterion='distance')

    # for cl in np.unique(clusters):
    #     heatmap=dff_av_arr[clusters==cl]
    #     cellids = np.array(range(dff_av_arr.shape[0]))[clusters==cl]
    #     plt.figure()
    #     sns.heatmap(heatmap, cmap='viridis',yticklabels=cellids)
    #     sns.set(font_scale=0.3)
    #     plt.ylabel("# of cells")
    #     plt.xlabel("track length (cm)")
    #     plt.title(f"cluster {cl}")
    #     plt.axvline(x=rewloc,color='white')
    #     plt.savefig(rf'Y:\sstcre_analysis\clustering\epoch{ep}_cluster{cl}_maxd{max_d}.pdf',
    #     bbox_inches='tight'
    #     )

    #     plt.close()        
tracked_cells_in_cluster = []
for cl in cluster_df['Cluster'].unique():
    iid = cluster_df.loc[cluster_df.Cluster == cl, 'Series']
    tracked_iid = [ii for ii in iid if ii in cellids] #only tracked cells heatmap
    weekinds = [np.where(cc[:,17]==cell)[0][0] for cell in tracked_iid]
    tracked_cells_in_cluster.append((tracked_iid, weekinds, cl))
    if len(tracked_iid)> 0: # only if cluster contains tracked cells
        heatmap = dff_av_norm[tracked_iid]
        plt.figure()
        sns.heatmap(heatmap, cmap='viridis',yticklabels=weekinds)
        sns.set(font_scale=0.5)
        plt.ylabel("week cell ID")
        plt.xlabel("track length (cm)")
        plt.title(f"epoch{ep+1}, {cl}")
        plt.axvline(x=rewloc,color='white')            
        # plt.savefig(os.path.join(dst, f'epoch{ep+1}_{cl}.pdf'),
        #     bbox_inches='tight'
        #     )
        # plt.close()
            
tracked_cell_df = pd.DataFrame(tracked_cells_in_cluster)
tracked_cell_df.columns = ['day_cellid', 'cluster']
tracked_cell_df.to_pickle(os.path.join(dst, 'across_epochs_clusters_tracked_cells.p'))
#%%
# apply clusters to diff day
tracked_days = np.arange(5,17)
days = [27, 28, 29, 30, 31, 32, 33, 34, 36,38,39,40]

for di,day in enumerate(days):
    tracked_day = tracked_days[di] # 0 index bc python list
    daypth = rf'Z:\sstcre_imaging\e201\{day}'
    daypth = [os.path.join(daypth, xx) for xx in os.listdir(daypth) if "ZD" in xx][0]
    fallpth = os.path.join(daypth,'suite2p', 'plane0', 'Fall.mat')
    fall = loadmat(fallpth)
    epoch = np.where(fall['changeRewLoc']>1)
    mask=cc[:,tracked_day] # 17 is the dark num
    # get cell ids 
    cellids = mask[mask>-1]
    dff_day = dff[tracked_day].T#[mask][mask>-1] # only get tracked cells
    dst = rf'Y:\sstcre_analysis\clustering\day{day}';makedir(dst)
    with open(r'Y:\sstcre_analysis\clustering\clusters_tracked_cells.p', "rb") as fp: #unpickle
        tracked_cell_clusters_day41 = pickle.load(fp)
    tracked_cell_clusters = []
    for cl in tracked_cell_clusters_day41.cluster.unique():
        day41_iid = tracked_cell_clusters_day41.loc[tracked_cell_clusters_day41.cluster == cl, 
                            'day_cellid'].values[0]
        cl_iid = []; weekinds = []
        if len(day41_iid)>0:
            for cell in day41_iid:
                weekind = np.where(cc[:,17]==cell)[0][0]; weekinds.append(weekind)
                cl_iid.append(cc[weekind, tracked_day])
            cl_iid_org = np.array(cl_iid)
            cl_iid = cl_iid_org[cl_iid_org>-1]
            weekinds = np.array(weekinds)
            weekinds = weekinds[cl_iid_org>-1] # mask both day and week ids
            tracked_cell_clusters.append((cl_iid, weekinds, cl))
    tracked_cell_clusters = pd.DataFrame(tracked_cell_clusters,
                             columns = ['day_cellid', 'week_cellid', 'cluster'])
    # save cluster ids
    tracked_cell_clusters.to_pickle(os.path.join(dst, 'clusters_tracked_cells.p'))

    for ep in range(len(epoch[1])):
        print(ep)
        try:
            epoch_start,epoch_stop = epoch[1][ep],epoch[1][ep+1]
        except:
            epoch_start,epoch_stop = epoch[1][ep],len(fall['trialnum'][0]) #else set 
            # to end of RECORDING
        rewloc = np.unique(fall['changeRewLoc'])[ep+1]
        trials = max(max(fall['trialnum'][:, epoch_start:epoch_stop])) # total number of trials
        dff_av = []
        
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
                dff_binned = np.zeros((270,dff_day.T[trialind].shape[1])) # init binning by track length
                for i in range(dff_trial.shape[1]): # bin per cell
                    llen = dff_trial.shape[0] # get frames per trial
                    # TODO: how to speed this up?
                    data= np.linspace(1,llen,llen)  
                    bins=np.linspace(1,llen,271)     
                    dig = np.digitize(data,bins)  
                    bin_means = [np.nanmean(dff_trial[:,i][dig == ii]) for ii in range(1,
                                len(bins))]
                    dff_binned[:,i]=bin_means
                dff_av.append(dff_binned) #mean across all frames
                print(dff_binned.shape)

        dff_av_arr = np.mean(np.array(dff_av),axis=0).T # gets all cells at this stage        
        dff_av_norm=MinMaxScaler().fit_transform(dff_av_arr.T).T 
            
        for cl in tracked_cell_clusters.cluster:
            tracked_iid = tracked_cell_clusters.loc[tracked_cell_clusters.cluster == cl,
                        'day_cellid'].values[0]
            if len(tracked_iid)>0:
                heatmap = dff_av_norm[tracked_iid]
                week_cellid = tracked_cell_clusters.loc[tracked_cell_clusters.cluster == cl,
                        'week_cellid'].values[0]
                plt.figure()
                sns.heatmap(heatmap, cmap='viridis',yticklabels=week_cellid)
                sns.set(font_scale=0.5)
                plt.ylabel("week cell ID")
                plt.xlabel("track length (cm)")
                plt.title(f"epoch{ep+1}, {cl}")
                plt.axvline(x=rewloc,color='white')            
                plt.savefig(os.path.join(dst, f'epoch{ep+1}_{cl}.pdf'),
                    bbox_inches='tight'
                    )
                plt.close()