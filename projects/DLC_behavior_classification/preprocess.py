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
        for i,row in mouse_df[:50].iterrows():
            vrfl_p = os.path.join(dlcfls,row["VR"][:16]+"_vr_dlc_align.p")    
            if os.path.exists(vrfl_p):
                print(row["mouse"], row["VR"])
                bigdf.append(collect_clustering_vars(os.path.join(dlcfls,
                        row["DLC"]),vrfl_p))                    

        big_df = pd.concat(bigdf)
        big_df.to_pickle(os.path.join(dlcfls,'all_mice_dlc_vr_df.p'))
        
        return big_df
    
    if step == 3: # get clustering vars and cluster
        with open(os.path.join(dlcfls,'all_mice_dlc_vr_df.p'),'rb') as fp: #unpickle
                df = pickle.load(fp)
        cluster_output = {}
        #novr
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
    vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\DLC\dlc_mixedmodel2' # h5 and csv files from dlc
    df = preprocess(0,vrdir,dlcfls)
    # need to fix vr mat files separately in matlab, lol
    # will not work otherwise!!!
    # you would want to run `fix_vr_data...` on the whole VR_data folder to avoid 
    # errors
    # uncomment below if you don't want to re-run step 0 but want the mouse df 
    # if you just want to remake the mouse df after deleting mice
    # you can run step 0 and go directly to step 2
    # (do not need to remake vr_align.p as you have just deleted some mice)
    with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
        df = pickle.load(fp)
    # makes vr align pickle
    preprocess(1,vrdir,dlcfls)
    # gets clustering vars
    dfs = preprocess(2,vrdir,dlcfls)
    columns = ['tongue', 'nose', 'paw', 
               'whiskerUpper', 'whiskerLower']
        # , 'forwardvelocity']#, 'whiskerUpper',
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
    pca_2_result_bl=pca_2_result[dfkmeans['grooms']]
    plt.scatter(pca_2_result_bl[:, 0] , pca_2_result_bl[: , 1] , color='k', 
                marker='o', facecolors='none')
    # pca_2_result_sn=pca_2_result[dfkmeans['tongue_lbl']]
    # plt.scatter(pca_2_result_sn[:, 0] , pca_2_result_sn[: , 1] , 
    #             color='k', marker='x')
    pca_2_result_lk=pca_2_result[dfkmeans['sniff_lbl']]
    plt.scatter(pca_2_result_lk[:, 0] , pca_2_result_lk[: , 1] , 
                color='k', marker='*', facecolors='none')
    # pca_2_result_mo=pca_2_result[dfkmeans['sniff_lbl']]
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
                'paw', 'sniff'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-means, n = {len(dfkmeans.animal.unique())}, experiment: \n{dfkmeans.experiment.unique()[0]}")

# %%
test = dfkmeans[(dfkmeans.animal=='E189') & (dfkmeans.data=='18_Apr_2023')]

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
ax3 = ax1.twinx()

ax1.plot(test.lickVoltage.astype(float).values,  
             color='r')
ax2.scatter(np.arange(len(test))[test.licks.astype(float).values>0],
        test.licks.astype(float).values[test.licks.astype(float).values>0]*.995,
        marker = 'o', s=40, facecolors='none', color = 'k')
ax3.scatter(np.arange(len(test))[test.tongue_lbl.astype(float).values>0],
        test.tongue_lbl.astype(float).values[test.tongue_lbl.astype(float).values>0]*1.01,
        marker = 'o', s=40, facecolors='none', color = 'b')

ax2.set_ylim(0.98, 1.02) #Define limit/scale for primary Y-axis
ax1.set_ylim(-.1, -.02) #Define limit/scale for secondary Y-axis
ax3.set_ylim(0.98, 1.02) #Define limit/scale for primary Y-axis

plt.show()

plt.plot()
plt.scatter(np.arange(len(test))[test.licks.astype(float).values>0],
        test.licks.astype(float).values[test.licks.astype(float).values>0],
        marker = 'o', s=2)
