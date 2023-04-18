from pathlib import Path
import numpy as np, os, matplotlib.pyplot as plt, scipy.io, seaborn as sns
import h5py

src = r'X:\sstcre_imaging\e201'
weeks = np.arange(2,5)
# weeks 5,6,7,8 were run with different params
metrics_plt = []
mapp = r'Y:\sstcre_analysis\celltrack\e201_week1-6\Results\commoncells.mat'
for nweek,week in enumerate(weeks):
    #if matlab save data use ‘-v7.3’，scipy.io.loadmat api load may lead mistake ，thus can use like：    
    try:
        f = h5py.File(mapp, 'r')
        print(f.keys()) # <KeysViewHDF5 ['X', 'y']>  
        mapp_arr = f['commoncells'][:]
    except:
        mapp_arr = scipy.io.loadmat(mapp)['commoncells'].T
    # ordered as week 5,6,7,8 as col 1,2,3,4

    for path in Path(src).rglob(f'week{week}/stat.npy'):
        stat = np.load(path,allow_pickle=True)
    # skewdcells = find(skewness(l.F,1,2)<2); 
    # %looks at skewness of cells, <2 --> interneurons; if omitted, gets all cells
    skews = [stat[xx]['skew'] for i,
            xx in enumerate(range(len(stat))) if i in mapp_arr[nweek+1,:]]
    compacts = [stat[xx]['compact'] for i,
            xx in enumerate(range(len(stat))) if i in mapp_arr[nweek+1,:]]
    radius = [stat[xx]['radius'] for i,
            xx in enumerate(range(len(stat))) if i in mapp_arr[nweek+1,:]]
    skew_plt.append(skews)
    compact_plt.append(compacts)
    radii.append(radius)
    
src = r'X:\sstcre_imaging\e201'
weeks = np.arange(5,9)
# weeks 5,6,7,8 were run with different params
mapp = r'Y:\sstcre_analysis\celltrack\e201_week4-8\Results\commoncells.mat'
for nweek,week in enumerate(weeks):
    #if matlab save data use ‘-v7.3’，scipy.io.loadmat api load may lead mistake ，thus can use like：    
    f = h5py.File(mapp, 'r')
    print(f.keys()) # <KeysViewHDF5 ['X', 'y']>  
    mapp_arr = f['commoncells'][:]
    # ordered as week 5,6,7,8 as col 1,2,3,4

    for path in Path(src).rglob(f'week{week}/stat.npy'):
        stat = np.load(path,allow_pickle=True)
    # skewdcells = find(skewness(l.F,1,2)<2); 
    # %looks at skewness of cells, <2 --> interneurons; if omitted, gets all cells
    skews = [stat[xx]['skew'] for i,
            xx in enumerate(range(len(stat))) if i in mapp_arr[nweek,:]]
    compacts = [stat[xx]['compact'] for i,
            xx in enumerate(range(len(stat))) if i in mapp_arr[nweek,:]]
    skew_plt.append(skews)
    compact_plt.append(compacts)

import pandas as pd
# loading seaborn dataset tips
df = pd.DataFrame(skew_plt).T 
df.columns = ['week2','week3','week4','week5_lowthres',
              'week6_lowthres','week7_lowthres','week8_lowthres']
df['cell_ind'] = df.index
# convert columns to row
df=df.melt(id_vars= ["cell_ind"], value_name="skew")

# creating boxplot
sns.boxplot(x='variable', y='skew', data=df, showfliers = False)
plt.xticks(rotation='vertical')
# adding data points
sns.stripplot(x='variable', y='skew', data=df,
              color='k',size=2, alpha=0.5)
# display plot
plt.show()

df = pd.DataFrame(compact_plt).T 
df.columns = ['week2','week3','week4','week5_lowthres',
              'week6_lowthres','week7_lowthres','week8_lowthres']
df['cell_ind'] = df.index
# convert columns to row
df=df.melt(id_vars= ["cell_ind"], value_name="compact")

sns.stripplot(x='variable', y='compact', data=df,
              color='k',size=2, alpha=0.5)
plt.xticks(rotation='vertical')
# display plot
plt.show()
