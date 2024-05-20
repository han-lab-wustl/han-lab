"""
cluster inactivated cells based on their activity
"""
#%%
import numpy as np, pickle, os, pandas as pd, matplotlib.pyplot as plt, scipy, random
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

# import raw data
with open("Z:\dcts_com_opto_inference_wcomp.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
print(len(dcts))
print(len(conddf))
plt.close('all')
for dd in range(len(conddf)):
    bin_size = 3 # cm
    animal = conddf.animals.values[dd]
    intype = conddf.in_type.values[dd]
    day = conddf.days.values[dd]
    if intype=='vip':
        dct = dcts[dd]
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'forwardvel', 'ybinned', 'iscell',
                                    'trialnum', 'bordercells', 'changeRewLoc', 'licks',
                                    'coms', 'changeRewLoc', 'tuning_curves_early_trials',
                                    'tuning_curves_late_trials', 'coms_early_trials'])
        inactive = dcts[dd]['inactive']
        changeRewLoc = np.hstack(fall['changeRewLoc']) 
        eptest = conddf.optoep.values[dd]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]
        eps = np.append(eps, len(changeRewLoc)) 
        trialnum = np.hstack(fall['trialnum'])
        comp = dct['comp'] # eps to compare  
        other_ep = [xx for xx in range(len(eps)-1) if xx not in comp]
        # filter iscell
        coms = fall['coms'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]

        tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0]]]))
        tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1]]]))
        tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0]]]))
        tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1]]]))    
        
        if len(inactive)>1:
            coms1 = np.hstack(coms[comp[0]])[inactive]
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[1].imshow(np.concatenate([np.squeeze(tc1_late[inactive,:][np.argsort(coms1)]),
                np.squeeze(tc2_late[inactive,:][np.argsort(coms1)])]))
            axes[1].axhline(tc1_late[inactive,:].shape[0], color='y')
            axes[1].axvline(rewlocs[comp[0]]/bin_size, color='y', linestyle='--')
            axes[1].axvline(rewlocs[comp[1]]/bin_size, color='y')
            axes[1].set_title(f'{animal}, {day}, opto: {eptest}, late')
            axes[0].imshow(np.concatenate([np.squeeze(tc1_early[inactive,:][np.argsort(coms1)]),
                np.squeeze(tc2_early[inactive,:][np.argsort(coms1)])]))
            axes[0].axhline(tc1_late[inactive,:].shape[0], color='y')
            axes[0].axvline(rewlocs[comp[0]]/bin_size, color='y', linestyle='--')
            axes[0].axvline(rewlocs[comp[1]]/bin_size, color='y')
            axes[0].set_title(f'{animal}, {day}, opto: {eptest}, early')
#%%
y = tc2_late[inactive,:]
nans, x= nan_helper(y)
y[nans]= np.interp(x(nans), x(~nans), y[~nans])
# cluster 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(y)

# Choose the number of components
n_components = 2# Example: 5 components

# Create and fit the GMM
gmm = GaussianMixture(n_components=n_components, random_state=0)
gmm.fit(data_scaled)
clusters = gmm.predict(data_scaled)


# Visualize the results
plt.figure(figsize=(12, 8))
for i in range(n_components):
    # Plot each cluster
    plt.subplot(2, 3, i + 1)
    cluster_data = y[clusters == i]
    for trace in cluster_data:
        plt.plot(trace, alpha=0.4)
    plt.title(f'Cluster {i+1}')
plt.show()

print("AIC:", gmm.aic(data_scaled))
print("BIC:", gmm.bic(data_scaled))