"""PCA analysis
sept 2024
"""
#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your data
# Assume 'data' is a pandas DataFrame with rows as neurons and columns as activity features.
# Assume 'labels' is a pandas Series with the neuron class labels.

import os, sys, scipy, imageio, pandas as pd
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf

# Add custom path for MATLAB functions
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') 

# Formatting for figures
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)


# Define save path for PDF
condition = 'drd2'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'
savepth = os.path.join(savedst, f'{condition}.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# Define source directory and mouse name
src = r'Y:\drd'
# days = [3,4,5,6,7,9]
# days = [3,4,5,6,7,8,9,10,12]
days = [7,8,9, 13,14,15, 2,3,4,6,7,8]#,6,7,8, 10,11,12, 
mice = ['e253','e253','e253','e256','e256','e256',
        # 'e254','e254','e254', 'e255','e255','e255', 
        'e261','e261','e261','e262','e262','e262']
        # ]
        # 'e261','e261','e261','e262','e262','e262']
condition = ['drd2','drd2','drd2','drd2','drd2','drd2',
            'drd2ko','drd2ko','drd2ko','drd2ko','drd2ko',
            'drd2ko']
            # 'drd1','drd1','drd1','drd1','drd1','drd1', 
            
range_val, binsize = 5 , 0.2 # seconds
meanrew_dff_all_days = []
# Iterate through specified days
for ii,dy in enumerate(days):
    day_dir = os.path.join(src, mice[ii], str(dy))
    postrew_dff_all_planes = []; perirew_all_planes = []
    for root, dirs, files in os.walk(day_dir):
        for file in files:
            if 'plane' in root and file.endswith('roibyclick_F.mat'):
                f = loadmat(os.path.join(root, file))
                print(os.path.join(root, file))
                try:
                    plane = int(root.split("plane")[1])
                except:
                    plane = int(root.split("plane")[1][0])

                # Filename pattern to match
                target_filename = 'masks.jpg'
                blotches_file_path = None
                
                # Check for the blotches.jpg file
                for file in os.listdir(root):
                    if os.path.isfile(os.path.join(root, file)) and file.endswith(target_filename):
                        blotches_file_path = os.path.join(root, file)
                        break

                eps = np.where(f['changeRewLoc'] > 0)[1]
                eps = np.append(eps, len(f['changeRewLoc'][0]))
                rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                plane = int(root.split("plane")[1][0])
                dFF_iscell = f['dFF']
                F_iscell = f['F']
                means = np.nanmean(F_iscell, axis=0)
                # remove dim cells
                dFF_iscell = dFF_iscell[:, means>450]
                dFF_iscell_filtered = dFF_iscell.T
                dff_res = []
                perirew = []

                # Iterate through cells
                for cll in range(dFF_iscell_filtered.shape[0]):
                    X = np.array([f['forwardvel'][0]]).T 
                    X = sm.add_constant(X)
                    y = dFF_iscell_filtered[cll, :]
                    
                    model = sm.GLM(y, X, family=sm.families.Gaussian())
                    result = model.fit()
                    dff_res.append(result.resid_pearson)
                    
                    dff = result.resid_pearson
                    dffdf = pd.DataFrame({'dff': dff})
                    dff = np.hstack(dffdf.rolling(5).mean().values)
                
                    normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
                        dff, 
                        f['solenoid2'][0].astype(int), 
                        f['timedFF'][0], 
                        f['trialnum'][0], 
                        range_val, 
                        binsize,
                    )
                    perirew.append(meanrewdFF)
                perirew_all_planes.append(perirew)
    meanrew_dff_all_days.append(perirew_all_planes)
#%%            
# 2. Standardize the data
from sklearn.preprocessing import StandardScaler
## removes 0 cell planes
pcadata = [np.vstack([xxx for xxx in xx if len(xxx)>0]) for xx in meanrew_dff_all_days]
lblnum = [xx.shape[0] for xx in pcadata]

pcadata = np.vstack(pcadata)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pcadata)
label = np.concatenate([[cond]*lblnum[ii] for ii, cond in enumerate(condition)])
# 3. Apply PCA
pca = PCA(n_components=12)  # Adjust the number of components as needed
pca_result = pca.fit_transform(scaled_data)

# 4. Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4',
            'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])
pca_df['cell_type'] = label

# 5. Plot the PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC7', y='PC8', hue='cell_type',data=pca_df,s=200)
plt.title('PCA of Peri-reward activity')
plt.xlabel('Principal Component 4')
plt.ylabel('Principal Component 5')
plt.legend(title='Neuron Class')
plt.show()
#%%
from sklearn.cluster import KMeans

# Fit KMeans on the PCA result
labels = np.array(label)
result = pca_result[:,2:4]

kmeans = KMeans(n_clusters=len(np.unique(labels)))
kmeans.fit(pcadata)
clusters = kmeans.predict(pcadata)

# Create a DataFrame for the clustering results
cluster_df = pd.DataFrame(data=result, columns=['PC1', 'PC2'])
cluster_df['Cluster'] = clusters
cluster_df['cell_type'] = label

# Plot the clusters
ax = sns.scatterplot(x='PC1', y='PC2', hue='Cluster',
        data=cluster_df,s=200,palette='colorblind')
plt.title('K-means Clustering on PCA')
plt.xlabel(f'Principal Component 4')
plt.ylabel(f'Principal Component 5')
plt.show()

plt.figure()
ax = sns.scatterplot(x='PC1', y='PC2', hue='cell_type',
        data=cluster_df,s=200,palette='colorblind')
plt.figure()
ax = sns.countplot(data=cluster_df,x='cell_type', hue='Cluster',
                palette='colorblind') 
#%%
import umap

# Apply UMAP
umap_model = umap.UMAP(n_components=4, n_neighbors=20, 
        min_dist=0.05)
umap_result = umap_model.fit_transform(scaled_data)
#%%
# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(data=umap_result)
umap_df['Class'] = label
umap_df.columns = np.array(umap_df.columns).astype(str)

# Plot the UMAP results
fig, ax = plt.subplots(figsize=(6,5))
sns.scatterplot(x='0', y='1', hue='Class', data=umap_df, 
        palette=sns.color_palette('colorblind')[1:],s=100)
# ax.set_title('UMAP of Peri-reward activity across 3 days')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

plt.savefig(os.path.join(savedst,'umap_drdko_with_all_mice.svg'))