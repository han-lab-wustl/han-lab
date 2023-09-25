import os, sys, pickle, pandas as pd, numpy as np
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from kmeans import collect_clustering_vars, run_pca, run_kmeans
from preprocessing import fixcsvcols
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
dlcfls = r'Y:\DLC\dlc_mixedmodel2'

with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp) 

for i,row in mouse_df.iterrows():
    dfpth = os.path.join(dlcfls, row['DLC']) #.values[0]
    matfl = os.path.join(dlcfls,row["VR"][:16]+"_vr_dlc_align.p")    
    
    with open(matfl,'rb') as fp: #unpickle
        mat = pickle.load(fp)
    eps = np.where(mat['changeRewLoc']>0)[0]    
    eps = np.hstack([list(eps), len(mat['changeRewLoc'])])    
    # at least 2 epochs, rewarded hrz
    if 'HRZ' in mat['experiment'] and sum(mat['rewards']>0) and len(eps)>2: # only for hrz
        df = pd.read_csv(dfpth)
        # if 'bodyparts' not in df.columns: ## this was missing some dfs
        try:  
            df = fixcsvcols(dfpth)
        except Exception as e:
            print(e)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns = ["Unnamed: 0"])
        idx = len(df) - 1 if len(df) % 2 else len(df)
        df = df[:idx].groupby(df.index[:idx] // 2).mean()
        #paw
        df['PawTop_x'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
        df['PawTop_y'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
        df['PawMiddle_x'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
        df['PawMiddle_y'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
        df['PawBottom_x'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0
        df['PawBottom_y'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0
        paw = df[['PawTop_y','PawBottom_y','PawMiddle_y']].astype('float32').mean(axis=1)
        if sum(paw.values)>0:
            plt.figure(); plt.scatter(np.where(paw>0)[0], mat['ybinned'][paw>0],c='y', marker = 'D', label='paw'); 
            plt.scatter(np.where(mat['rewards']>0.5)[0], mat['ybinned'][mat['rewards']>0.5],c='g', marker = '*',s=10**2,
                        label='reward'); 
            plt.scatter(np.where(mat['licks'])[0], mat['ybinned'][mat['licks']>0],c='r',s=4, label = 'lick'); 
            plt.plot(mat['ybinned'],
                        label='ypos')
            plt.title(f'mouse = {row.mouse}, date = {row.date}')
            plt.legend()