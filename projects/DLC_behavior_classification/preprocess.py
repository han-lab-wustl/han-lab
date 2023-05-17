import os, sys, pickle, pandas as pd, numpy as np
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from kmeans import collect_clustering_vars, run_pca, run_kmeans
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def preprocess(step,vrdir, dlcfls,columns=False,
               only_add_experiment=False):
    if step == 0:
    #vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    #dlcfls = r'G:\dlc_mixedmodel2' # h5 and csv files from dlc
        mouse_df = preprocessing.copyvr_dlc(vrdir, dlcfls)          
        mouse_df.to_pickle(os.path.join(dlcfls,"mouse_df.p"))  

        return mouse_df

    if step == 1: # copy vr files from  usb matching dlc data
        with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
            df = pickle.load(fp)
        [preprocessing.VRalign(os.path.join(dlcfls,row["VR"]), 
                    os.path.join(dlcfls,row["DLC"]),only_add_experiment=only_add_experiment) for i,row in df.iterrows()]
            
    if step == 2: # align to vr
        bigdf = []
        with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp)
        for i,row in mouse_df.iterrows():
            vrfl_p = os.path.join(dlcfls,row["VR"][:16]+"_vr_dlc_align.p")    
            print(row["mouse"], row["VR"])
            bigdf.append(collect_clustering_vars(os.path.join(dlcfls,row["DLC"]),vrfl_p))                    

        big_df = pd.concat(bigdf)
        big_df.to_pickle(os.path.join(dlcfls,'all_mice_dlc_vr_df.p'))
        
        return big_df
    
    if step == 3: # get clustering vars and cluster
        with open(os.path.join(dlcfls,'all_mice_dlc_vr_df.p'),'rb') as fp: #unpickle
                df = pickle.load(fp)
        cluster_output = {}
        #novr; clusters = 4; default
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment=='no_Random_Rewards_no_VR_Sol2_CS'],
                                    columns)
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        cluster_output['novr'] = [pca,lbl,df_kmeans]
        #pavlovian
        tasks = ['Random_Rewards_no_VR_Sol2_CS', 
                 'Random_Rewards_no_VR_Sol2_CS_Omit_Experiments']
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment.isin(tasks)], 
                                    columns)
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        cluster_output['pavlovian'] = [pca,lbl,df_kmeans]
        #hrz
        tasks = ['M3_M4_altered_dim_HRZ_norewards__MM_sol2', 
                 'M3_M4_altered_dim_HRZ_double_probe_middle_5cmRL_GM_sol2']
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment.isin(tasks)],
                                    columns)
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        cluster_output['hrz'] = [pca,lbl,df_kmeans]

        return cluster_output


if __name__ == "__main__":
    vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\DLC\dlc_mixedmodel2' # h5 and csv files from dlc
    # df = preprocess(0,vrdir,dlcfls)
    # now need to fix vr mat files separately in matlab, lol
    # will not work otherwise!!!
    # uncomment below if you don't want to re-run step 0 but want the mouse df 

    with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
        df = pickle.load(fp)
    # preprocess(1,vrdir,dlcfls)
    # dfs = preprocess(2,vrdir,dlcfls)
    columns = ['blinks', 'eye_centroid_x', 'eye_centroid_y', 
        'tongue', 'nose', 'paw', 'forwardvelocity']#, 'whiskerUpper',
    #    'whiskerLower', 'ybinned', 'licks',
       #'lickVoltage']
    
    cluster_output = preprocess(3,vrdir,dlcfls,columns=columns)
    #pca_2_result, label, dfkmeans
       
#%%
# custom analysis, can make into function
%matplotlib inline
experiments = ['novr', 'pavlovian', 'hrz']
for experiment in experiments:
    label = cluster_output[experiment][1]
    dfkmeans = cluster_output[experiment][2]
    pca_2_result= cluster_output[experiment][0]
    plt.figure()
    # plot pc components
    uniq = np.unique(label)
    for i in uniq:
        plt.scatter(pca_2_result[label == i, 0] , pca_2_result[label == i , 1] , label = i)

    #plot behaviors
    pca_2_result_bl=pca_2_result[dfkmeans['tongue_lbl']]
    plt.scatter(pca_2_result_bl[:, 0] , pca_2_result_bl[: , 1] , color='k', 
                marker='o', facecolors='none')
    # pca_2_result_sn=pca_2_result[dfkmeans['eye_centroid_ylbl']]
    # plt.scatter(pca_2_result_sn[:, 0] , pca_2_result_sn[: , 1] , 
    #             color='k', marker='o')
    # pca_2_result_lk=pca_2_result[dfkmeans['whisking']]
    # plt.scatter(pca_2_result_lk[:, 0] , pca_2_result_lk[: , 1] , 
    #             color='k', marker='o', facecolors='none')
    # pca_2_result_mo=pca_2_result[dfkmeans['mouth_mov']]
    # plt.scatter(pca_2_result_mo[:, 0] , pca_2_result_mo[: , 1] , 
    #             color='k', marker='d', facecolors='none')
    # pca_2_result_fast=pca_2_result[dfkmeans['fastruns']]
    # plt.scatter(pca_2_result_fast[:, 0] , pca_2_result_fast[: , 1] , 
    #             color='k', marker='s', facecolors='none')
    # pca_2_result_stop=pca_2_result[dfkmeans['stops']]
    # plt.scatter(pca_2_result_stop[:, 0] , pca_2_result_stop[: , 1] , 
    #             color='k', marker='|')

    # plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'
    #             ])
    plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4',
                'tongue'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-means, n = {len(dfkmeans.animal.unique())}, experiment: \n{dfkmeans.experiment.unique()[0]}")

# %%
