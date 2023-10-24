import os, sys, pickle, pandas as pd, numpy as np, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
from kmeans import collect_clustering_vars, run_pca, run_kmeans
from preprocessing import fixcsvcols
#analyze videos and copy vr files before this step
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
from math import ceil 
import datetime

vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
dlcfls = r'Y:\DLC\dlc_mixedmodel2'#\for_analysis'

with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp) 

groom_start_stops = []
mice = []
hrz_summary = True
# d = datetime.datetime(2023, 5, 1)
for i,row in mouse_df.iterrows():
    # if row.mouse == 'E201': # if only analyzing particular mouse       
        datetime_str = row.date
        # filter by date of behavior
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y-%m-%d')
        if True: #datetime_object>d: # only late hrz days
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
                paw_y = df[['PawTop_y','PawBottom_y','PawMiddle_y']].astype('float32').mean(axis=1)
                paw_x = df[['PawTop_x','PawBottom_x','PawMiddle_x']].astype('float32').mean(axis=1)
                # if there is any grooming
                if sum(paw_y.values)>0:
                    paw_gf = scipy.ndimage.gaussian_filter(paw_y.values,3)
                    paw_gf_x = scipy.ndimage.gaussian_filter(paw_x.values,3)
                    diffs = np.diff((paw_gf>0).astype(int),axis=0)
                    starts = np.argwhere(diffs == 1).T[0]    
                    stops = np.argwhere(diffs == -1).T[0]                   
                    
                    if len(stops)<len(starts):
                        stops_ = np.zeros(len(stops)+1)
                        stops_[:-1] = stops
                        stops_[-1] = len(paw_gf)
                    else:
                        stops_ = stops
                    start_stop = stops_-starts
                    # filter by long grooms
                    starts = starts[start_stop>75]
                    stops_ = stops_[start_stop>75]
                    if len(starts)>0:
                        licks = mat['lickVoltage']<-0.07
                        ybin_paw = mat['ybinned'][:-1] # remove 1 for diff
                        rewz = mat['changeRewLoc'][mat['changeRewLoc']>0]
                        # categories for periods of grooming
                        rewzgrs = []; darktimegrs = []
                        beforerewgrs = []; afterrewgrs = []
                        for ep in range(len(eps)-1):
                            rng = np.arange(eps[ep],eps[ep+1])
                            trialnumep = np.hstack(mat['trialnum'][rng])
                            rng = rng[trialnumep>3] # no probes
                            trialnumep = trialnumep[trialnumep>3]
                            rewep = (mat['rewards']==1)[rng]
                            # only successful trials
                            s_tr = []
                            for tr in np.unique(trialnumep[trialnumep>3]):
                                if sum(rewep[trialnumep==tr])>0:
                                    s_tr.append(tr)
                            trm = np.isin(trialnumep,s_tr)    
                            # categorize by ypos
                            gr_ = [xx in rng for xx in starts]
                            gr_ = starts[gr_]
                            yposgr = [ceil(xx) for xx in mat['ybinned'][gr_]]
                            rewzrng = np.arange(rewz[ep]-5, rewz[ep]+6)
                            rewzgr = [xx in rewzrng for xx in yposgr]
                            if len(yposgr)>0:
                                rewzgrs.append(gr_[rewzgr])
                                darktimegrs.append(gr_[[xx<3 for xx in yposgr]])
                                beforerewgrs.append(gr_[(yposgr<min(rewzrng)) & [xx>=3 for xx in yposgr]])
                                afterrewgrs.append(gr_[yposgr>max(rewzrng)])
                        beforerewgrs = np.hstack(beforerewgrs)
                        afterrewgrs = np.hstack(afterrewgrs)
                        darktimegrs = np.hstack(darktimegrs)
                        rewzgrs = np.hstack(rewzgrs)
                        if hrz_summary:    
                            # plots hrz behavior with grooms
                            plt.figure()
                            plt.plot(mat['ybinned'], color='slategray',
                                    linewidth=0.5)
                            plt.scatter(np.argwhere(mat['rewards']==0.5).T[0], mat['ybinned'][mat['rewards']==0.5], color='b', marker='o')
                            plt.scatter(np.argwhere(licks).T[0], mat['ybinned'][licks], color='r', marker='o',
                                        s = 2**2)
                            plt.scatter(starts, ybin_paw[starts], color='y', marker='*',
                                        s = 20**2)
                            plt.title(os.path.basename(matfl))
                            plt.savefig(rf'Y:\DLC\dlc_mixedmodel2\figures\{os.path.basename(matfl)}_grooming_beh.pdf')
                            
                            fig, ax = plt.subplots()
                            cat = ['dark time', 'before rew', 'after rew', 'rew zone']
                            counts = [len(darktimegrs), len(beforerewgrs), len(afterrewgrs),
                                    len(rewzgrs)]
                            ax.bar(cat, counts)
                            ax.set_title(os.path.basename(matfl))
                            ax.set_ylabel('number of grooming bouts')
                            plt.savefig(rf'Y:\DLC\dlc_mixedmodel2\figures\{os.path.basename(matfl)}_grooming_beh_quant.pdf')

                        groom_start_stops.append((starts,stops_))
                        
                        mice.append(row)
