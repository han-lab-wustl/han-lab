# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:13:27 2023

@author: Han
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np, seaborn as sns
import os
import scipy.ndimage
import pickle
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def collect_clustering_vars(df,matfl):
    #cleanup
    df = pd.read_csv(df)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns = ["Unnamed: 0"])
    with open(matfl,'rb') as fp: #unpickle
        mat = pickle.load(fp)
    forwardvelocity = mat['forwardvel']
    # plt.plot(forwardvelocity)
    # plt.axhline(y=75, color='r', linestyle='-')
    # bin every other row
    idx = len(df) - 1 if len(df) % 2 else len(df)
    df = df[:idx].groupby(df.index[:idx] // 2).mean()
    poses = df.columns[1:]
    eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']
    #plot blinks
    #here i think y pos starts from above
    # plt.plot(df['EyeNorth_y'].astype('float32').values - df['EyeSouth_y'].astype('float32').values)
    # plt.ylabel('y position (pixels)')
    # plt.ylim(-50, 10)
    # plt.xlabel('frames')

    #plot nose movement
    # plt.plot(np.mean(df[['NoseTopPoint_y', 'NoseBottomPoint_y', 'NoseTip_y']].astype('float32').values,1))
    # plt.ylabel('nose y position (pixels)')
    # plt.xlabel('frames')

    #plot tongue1 movement
    #assign to nans/0
    #TODO: need to clearn up, apply filter to all?
    df['TongueTip_x'][df['TongueTip_likelihood'].astype('float32') < 0.9] = 0
    df['TongueTip_y'][df['TongueTip_likelihood'].astype('float32') < 0.9] = 0
    df['TongueTop_x'][df['TongueTop_likelihood'].astype('float32') < 0.9] = 0
    df['TongueTop_y'][df['TongueTop_likelihood'].astype('float32') < 0.9] = 0
    df['TongueBottom_x'][df['TongueBottom_likelihood'].astype('float32') < 0.9] = 0
    df['TongueBottom_y'][df['TongueBottom_likelihood'].astype('float32') < 0.9] = 0
    # plt.plot(df['TongueTip_y'].astype('float32'))
    # plt.plot(df['TongueTop_y'].astype('float32'))
    # plt.plot(df['TongueBottom_y'].astype('float32'))
    #whisker
    df['WhiskerUpper1_x'][df['WhiskerUpper1_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerUpper1_y'][df['WhiskerUpper1_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerUpper_x'][df['WhiskerUpper_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerUpper_y'][df['WhiskerUpper_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerUpper3_x'][df['WhiskerUpper3_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerUpper3_y'][df['WhiskerUpper3_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower_x'][df['WhiskerLower_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower_y'][df['WhiskerLower_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower1_x'][df['WhiskerLower1_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower1_y'][df['WhiskerLower1_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower3_x'][df['WhiskerLower3_likelihood'].astype('float32') < 0.9] = 0
    df['WhiskerLower3_y'][df['WhiskerLower3_likelihood'].astype('float32') < 0.9] = 0
    # plt.plot(df['WhiskerUpper_x'].astype('float32'))
    # plt.plot(df['WhiskerLower_x'].astype('float32'))

    #paw
    df['PawTop_x'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
    df['PawTop_y'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
    df['PawMiddle_x'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
    df['PawMiddle_y'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
    df['PawBottom_x'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0
    df['PawBottom_y'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0
    # plt.plot(df['PawTop_y'].astype('float32'))
    # plt.plot(df['PawMiddle_x'].astype('float32'))
    # plt.plot(df['PawBottom_x'].astype('float32'))

    #eye centroids
    centroids_x = []; centroids_y = []
    for i in range(len(df)):
        eye_x = np.array([df[xx+"_x"].iloc[i] for xx in eye])
        eye_y = np.array([df[xx+"_y"].iloc[i] for xx in eye])
        eye_coords = np.array([eye_x, eye_y])
        centroid_x, centroid_y = centeroidnp(eye_coords)
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)

    # plt.plot(centroids[1000:])
    #blinks
    blinks=scipy.ndimage.gaussian_filter(df['EyeNorth_y'].astype('float32').values - df['EyeSouth_y'].astype('float32').values,sigma=3)
    #tongue movement
    tongue=df[['TongueTip_x','TongueTop_x','TongueBottom_x']].astype('float32').mean(axis=1, skipna=False)
    #nose
    nose=df[['NoseTopPoint_y', 'NoseBottomPoint_y', 'NoseTip_y']].astype('float32').mean(axis=1, skipna=False).astype('float32').values
    whiskerUpper = df[['WhiskerUpper_x', 'WhiskerUpper1_x', 'WhiskerUpper3_x']].astype('float32').mean(axis=1)
    whiskerLower = df[['WhiskerLower_x','WhiskerLower3_x']].astype('float32').mean(axis=1)
    paw = df[['PawTop_y','PawBottom_y','PawMiddle_y']].astype('float32').mean(axis=1)

    datadf = [[os.path.basename(matfl)[:4]]*len(blinks), 
              [os.path.basename(matfl)[5:-15]]*len(blinks),
              blinks, centroids_x, centroids_y, tongue, nose, whiskerUpper,whiskerLower, paw, 
              forwardvelocity, mat['ybinned'], mat['licks'], mat['lickVoltage'],
              mat['changeRewLoc'],
              [mat['experiment']]*len(blinks)]
    datadf = np.array([np.ravel(np.array(xx)) for xx in datadf])
    return pd.DataFrame(datadf.T, columns = ["animal","data",
                "blinks", "eye_centroid_x", "eye_centroid_y","tongue", "nose", "whiskerUpper",
                "whiskerLower", "paw", 
              "forwardvelocity",'ybinned', 'licks', 'lickVoltage',
              'changeRewLoc','experiment'])

#%%
def run_pca(dfkmeans,columns):
    # PCA and kmeans
    #https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
    #things to feed into kmeasn
    
    dfkmeans[columns] = dfkmeans[columns].astype(float)
    # drops a few last frames with na fron vr??
    dfkmeans = dfkmeans.dropna()
    #classify blinks, sniffs, licks?
    dfkmeans['blinks_lbl'] = dfkmeans['blinks']>dfkmeans["blinks"].mean()+dfkmeans["blinks"].std()*3
    dfkmeans['eye_centroid_xlbl'] = dfkmeans['eye_centroid_x']>dfkmeans["eye_centroid_x"].mean()+dfkmeans["eye_centroid_x"].std()*3
    dfkmeans['eye_centroid_ylbl'] = dfkmeans['eye_centroid_y']>dfkmeans["eye_centroid_y"].mean()+dfkmeans["eye_centroid_y"].std()*3
    dfkmeans['sniff_lbl'] =  dfkmeans['nose']>dfkmeans["nose"].mean()+dfkmeans["nose"].std()*3
    dfkmeans['tongue_lbl'] =  dfkmeans['tongue']>0
    dfkmeans['grooms'] = dfkmeans['paw']>0
    # dfkmeans['whisking'] = dfkmeans['whiskerUpper']<dfkmeans["whiskerUpper"].mean()+dfkmeans["whiskerUpper"].std()*3
    dfkmeans['fastruns'] =  dfkmeans['forwardvelocity']>dfkmeans["forwardvelocity"].mean()+dfkmeans["forwardvelocity"].std()*3 #arbitrary thres
    #stopped for 5 seconds around the cell
    dfkmeans['stops'] =  [True if sum(dfkmeans["forwardvelocity"].iloc[xx-5:xx+5])==0 else False for xx in range(len(dfkmeans["forwardvelocity"]))]

    X_scaled=StandardScaler().fit_transform(dfkmeans[columns])#,'mouth_open1','mouth_open2']])
    #https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(X_scaled)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))
    #convert to df...
    X_scaled = pd.DataFrame(X_scaled, columns=columns)#,'mouth_open1','mouth_open2'])

    dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=X_scaled.columns, index=['PC_1', 'PC_2'])
    print('\n\n', dataset_pca)

    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())   
    print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
    print("\n******************************************************************")

    # 4. Hyperparameter tuning using the silhouette score method
    # Apart from the curse of dimensionality issue, KMeans also has this problem where we need to explicitly inform the KMeans model about the number of clusters we want our data to be categorised in, this hit and trial can be daunting, so we are using silhouette score method. Here you give a list of probable candidates and the metrics.silhouette_score method calculates a score by applying the KMeans model to our data considering one value (number of clusters) at a time. For eg., if we want to check how good our model will be if we ask it to form 2 clusters out of our data, we can check the silhouette score for clusters=2.

    # Silhouette score value ranges from 0 to 1, 0 being the worst and 1 being the best.

    # # candidate values for our number of cluster
    # parameters = np.linspace(2,10,9).astype(int)
    # # instantiating ParameterGrid, pass number of clusters as input
    # parameter_grid = sk.model_selection.ParameterGrid({'n_clusters': parameters})
    # best_score = -1
    # kmeans_model = KMeans()     # instantiating KMeans model
    # silhouette_scores = []
    # # evaluation based on silhouette_score
    # for p in parameter_grid:
    #     kmeans_model.set_params(**p)    # set current hyper parameter
    #     kmeans_model.fit(X_scaled)          # fit model on wine dataset, this will find clusters based on parameter p
    #     ss = sk.metrics.silhouette_score(X_scaled, kmeans_model.labels_)   # calculate silhouette_score
    #     silhouette_scores += [ss]       # store all the scores
    #     print('Parameter:', p, 'Score', ss)
    #     # check p which has the best score
    #     if ss > best_score:
    #         best_score = ss
    #         best_grid = p
    # # plotting silhouette score
    # plt.figure()
    # plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    # plt.xticks(range(len(silhouette_scores)), list(parameters))
    # plt.title('Silhouette Score', fontweight='bold')
    # plt.xlabel('Number of Clusters')
    # plt.show()

    return X_scaled, pca_2_result, dfkmeans

def run_kmeans(X_scaled, pca_2_result, dfkmeans, n_clusters=4):
    # fitting KMeans    
    kmeans = KMeans(n_clusters=n_clusters)    
    kmeans.fit(X_scaled)
    label = kmeans.fit_predict(X_scaled)
   
    return pca_2_result, label, dfkmeans
    # #only get cluster 2 frames
    # cluster2=df[label==3]
    # #here i think y pos starts from above
    # plt.plot(cluster2['eyeLidBottom_y'].astype('float32').values - cluster2['eyeLidTop_y'].astype('float32').values)
    # plt.ylabel('eyelidbottom-eyelidtop y position (pixels)')
    # plt.xlabel('frames')
    # plt.axhline(y=17, color='r', linestyle='-')

    # #plot nose movement
    # plt.plot(cluster2['nose_y'].astype('float32').values)
    # plt.ylabel('nose y position (pixels)')
    # plt.xlabel('frames')
    # plt.axhline(y=58, color='r', linestyle='-')

    # #get frames
    # cluster4frames=np.arange(39998)[label==3]
    #visualize cross correlation
    #https://www.kaggle.com/code/sanikamal/principal-component-analysis-with-kmeans
    # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(12, 10))
    # plt.title('Pearson Correlation of Movie Features')
    # # Draw the heatmap using seaborn
    # sns.heatmap(X_scaled.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
