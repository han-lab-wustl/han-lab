import os, sys, pickle, pandas as pd, numpy as np
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from kmeans import collect_clustering_vars, run_pca, run_kmeans
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def preprocess(step,vrdir, dlcfls,only_add_experiment=False):
    if step == 0:
    #vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    #dlcfls = r'G:\dlc_mixedmodel2' # h5 and csv files from dlc
        mouse_df = preprocessing.copyvr_dlc(vrdir, dlcfls)          
        mouse_df.to_pickle(os.path.join(dlcfls,"mouse_df.p"))  

        return mouse_df

    if step == 1:
        with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
            df = pickle.load(fp)
        [preprocessing.VRalign(os.path.join(dlcfls,row["VR"]), 
                    os.path.join(dlcfls,row["DLC"]),only_add_experiment=only_add_experiment) for i,row in df.iterrows()]
            
    if step == 2: 
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
    
    if step == 3:
        with open(os.path.join(dlcfls,'all_mice_dlc_vr_df.p'),'rb') as fp: #unpickle
                df = pickle.load(fp)
        pcadf = []
        #novr
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment=='no_Random_Rewards_no_VR_Sol2_CS'])
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        #pavlovian
        tasks = ['Random_Rewards_no_VR_Sol2_CS', 
                 'Random_Rewards_no_VR_Sol2_CS_Omit_Experiments']
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment.isin(tasks)])
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        #hrz
        tasks = ['M3_M4_altered_dim_HRZ_norewards__MM_sol2', 
                 'M3_M4_altered_dim_HRZ_double_probe_middle_5cmRL_GM_sol2']
        X_scaled, pca_2_result, df_kmeans = run_pca(df[df.experiment.isin(tasks)])
        pca, lbl, df_kmeans = run_kmeans(X_scaled, pca_2_result, df_kmeans)
        
        return 


if __name__ == "__main__":
    vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\DLC\dlc_mixedmodel2' # h5 and csv files from dlc
    # step=0
    # df = preprocess(step,vrdir,dlcfls)
    # now need to fix vr mat files separately in matlab, lol
    # will not work otherwise!!!
    # uncomment below if you don't want to re-run step 0 but want the mouse df 

    # with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
    #     df = pickle.load(fp)
    # preprocess(1,vrdir,dlcfls)
    # dfs = preprocess(2,vrdir,dlcfls)
    X_scaled, pcadf = preprocess(2,vrdir,dlcfls)