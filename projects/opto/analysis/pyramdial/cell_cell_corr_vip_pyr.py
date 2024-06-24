"""
cell-cell correlation with VIPs during optogenetics
zahra
june 2024
"""


import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.backends.backend_pdf
import matplotlib as mpl
from scipy.stats import pearsonr
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
from placecell import get_cosine_similarity
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data\vip_inhibition'
savepth = os.path.join(savedst, 'goal_cells_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

# Definitions and setups
mice = ["e216", "e217", "e218"]
dys_s = [[48], [14, 26, 27], [35, 38, 41, 44, 47, 50]]
opto_ep_s = [[2], [2, 3, 3], [3, 2, 3, 2, 3, 2]]
cells_to_plot_s = [[2231], [16,6,9], 
        [[453,63,26,38], [301, 17, 13, 320], [17, 23, 36, 10], 
        [6, 114, 11, 24], [49, 47, 6, 37], [434,19,77,5]]]
# Processing loop
for m, mouse_name in enumerate(mice):
    days = dys_s[m]
    cells_to_plot = cells_to_plot_s[m]
    opto_ep = opto_ep_s[m]
    dyind = 0
    for ii,day in enumerate(days):    
        plane=0 #TODO: make modular  
        params_pth = rf"Y:\analysis\fmats\{mouse_name}\days\{mouse_name}_day{day:03d}_plane{plane}_Fall.mat"
        # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
        #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'dFF', 'tuning_curves_early_trials',
            'tuning_curves_late_trials', 'coms', 'coms_early_trials', 'trialnum', 'rewards',
            'ybinned', 'forwardvel', 'timedFF', 'VR', 'iscell', 'Fc3', 'licks', 'bordercells'])  
        dFF = fall['dFF']; Fc3 = fall['Fc3']      
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum = fall['trialnum'][0]; rewards = fall['rewards'][0]
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf; timedFF = fall['timedFF'][0]
        forwardvel = fall['forwardvel'][0]
        licks = fall['licks'][0]
        eptest = opto_ep[ii]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))    
        # exclude last ep if too little trials
        lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
        if lastrials<8:
            eps = eps[:-1]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]    
        # just 1 cell for now
        vipcellind = cells_to_plot[ii][0]
        # cell-cell correlation
        # same criteria as the one in tuning curves
        pyr = Fc3[:, ((fall['iscell'][:,0].astype(bool)) & (~fall['bordercells'][0].astype(bool)))]        
        df = pd.DataFrame(pyr)
        df[-1] = dFF[:, vipcellind]
        correlations = df.corrwith(df[-1])        
        # Plot the raw data of the target cell and the most similar cells (top 10%)
        similar_cells_neg = sorted_correlations.tail(int(np.ceil(pyr.shape[1]*0.1))).index  # Including the target cell itself
        similar_cells_pos = sorted_correlations.head(int(np.ceil(pyr.shape[1]*0.1))).index  # Including the target cell itself
        similar_cells_pos_no_vip = np.array([xx for xx in similar_cells_pos if xx!='vip'])
        similar_cells_pos_no_vip_tc = tcs_late[opto_ep[ii]-2][similar_cells_pos_no_vip,:] # prev ep of opto
        similar_cells_neg_tc = tcs_late[opto_ep[ii]-2][np.array(list(similar_cells_neg)),:] # prev ep of opto
        plt.imshow(similar_cells_pos_no_vip_tc)
        plt.imshow(similar_cells_neg_tc)
        similar_cells_pos_no_vip_tc = tcs_late[opto_ep[ii]-1][similar_cells_pos_no_vip,:] # prev ep of opto
        similar_cells_neg_tc = tcs_late[opto_ep[ii]-1][np.array(list(similar_cells_neg)),:] # prev ep of opto
        plt.imshow(similar_cells_pos_no_vip_tc)
        plt.imshow(similar_cells_neg_tc)     
        similar_cells_pos_no_vip_tc = tcs_late[opto_ep[ii]-3][similar_cells_pos_no_vip,:] # prev ep of opto
        similar_cells_neg_tc = tcs_late[opto_ep[ii]-3][np.array(list(similar_cells_neg)),:] # prev ep of opto
        plt.imshow(similar_cells_pos_no_vip_tc)
        plt.imshow(similar_cells_neg_tc)   

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import KernelPCA

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.T)

        # Apply PCA
        kpca = KernelPCA(kernel='rbf', gamma=15, n_components=2)
        kpca_result = kpca.fit_transform(scaled_data)

        plt.scatter(kpca_result[:, 0], kpca_result[:, 1])
        plt.title('Kernel PCA')

        pca_result = pca.fit_transform(scaled_data)  # Transpose to get cells as rows
        df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df.columns)
        
        # Plot PCA result
        plt.figure(figsize=(10, 7))
        plt.scatter(df_pca['PC1'], df_pca['PC2'])
        # for cell in df_pca.index:
        #     plt.annotate(cell, (df_pca.loc[cell, 'PC1'], df_pca.loc[cell, 'PC2']))
        plt.title('PCA of Calcium Imaging Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(df.T)  # Transpose to get cells as rows
        df_tsne = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'], index=df.columns)

        # Plot t-SNE result
        plt.figure(figsize=(10, 7))
        plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'])
        # for cell in df_tsne.index:
        #     plt.annotate(cell, (df_tsne.loc[cell, 'TSNE1'], df_tsne.loc[cell, 'TSNE2']))
        plt.title('t-SNE of Calcium Imaging Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

        corr = d.corr()
        d = pd.DataFrame(data=Fc3[:,iind])
        # Compute the correlation matrix
        corr = d.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        plt.rc('font', size=20)      
        # Draw the heatmap with the mask and correct aspect ratio
        clustergrid = sns.clustermap(np.array(corr),cmap=cmap,vmax=.5,
                    metric='correlation')    
        
        linkage_matrix = clustergrid.dendrogram_row.linkage
        # Define a threshold to cut the dendrogram into clusters
        threshold = .5
        from scipy.cluster.hierarchy import fcluster

        # Assign cluster labels to each row
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        for cl in np.unique(cluster_labels):
            plt.figure()
            plt.plot(Fc3[:, np.where(d.columns==cl)[0]])
            plt.title(f'cluster: {cl}')

        # Add cluster labels to the DataFrame
        d.columns = cluster_labels

        # Display the cluster labels
        print(df[['Cluster']])





