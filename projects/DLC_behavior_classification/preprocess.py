import os, sys, pickle, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from kmeans import collect_clustering_vars
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#TODO: find a way to append df to add more mice

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
        

        return bigdf

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
    # step=1
    # preprocess(step,vrdir,dlcfls)
    step=2
    dfs = preprocess(step,vrdir,dlcfls)