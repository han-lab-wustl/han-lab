"""
cluster inactivated cells based on their activity
"""
#%%
import numpy as np, pickle, os, pandas as pd, matplotlib.pyplot as plt, scipy, random
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=12)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

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
with open(r"Z:\dcts_com_opto_inference_wcomp.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
plt.close('all')
coms_start_of_track = []
for dd in range(len(conddf)):
    bin_size = 3 # cm
    animal = conddf.animals.values[dd]
    intype = conddf.in_type.values[dd]
    day = conddf.days.values[dd]
    if True:#intype=='vip':
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
            coms2 = np.hstack(coms[comp[1]])[inactive]
            coms_start_of_track.append([animal, day, dd, intype, 
                eptest, np.diff(rewlocs[comp])[0],
                coms2]) # get coms of early firing cells
#%%

df_ = pd.DataFrame(coms_start_of_track, columns = ['animal', 'day', 'dd', 
    'intype', 'optoep', 'rewloc_shift', 'coms_early'])
vipopto = [np.nanmean(xx) for xx in df_[(df_.optoep>1) & (df_.intype=='vip')].coms_early.values]
anvipopto = [xx for xx in df_[(df_.optoep>1) & (df_.intype=='vip')].animal.values]
vipoff = [np.nanmean(xx) for xx in df_[(df_.optoep<2)&(df_.intype=='vip')].coms_early.values]
anvipoff = [xx for xx in df_[(df_.optoep<2) & (df_.intype=='vip')].animal.values]
ctrlopto = [np.nanmean(xx) for xx in df_[(df_.optoep>1) & (df_.intype!='vip')].coms_early.values]
anctrlopto = [xx for xx in df_[(df_.optoep>1) & (df_.intype!='vip')].animal.values]
ctrloff = [np.nanmean(xx) for xx in df_[(df_.optoep<2)&(df_.intype!='vip')].coms_early.values]
anctrloff = [xx for xx in df_[(df_.optoep<2) & (df_.intype!='vip')].animal.values]

df = pd.DataFrame(np.concatenate([vipopto, vipoff, ctrlopto, ctrloff]), columns = ['coms'])
df['condition'] = np.concatenate([['VIP LED on']*len(vipopto), ['VIP LED off']*len(vipoff),
                                ['Control LED on']*len(ctrlopto), ['Control LED off']*len(ctrloff)])
df['animal'] = np.concatenate([anvipopto,anvipoff,anctrlopto,anctrloff])
df=df.groupby(['animal', 'condition']).mean(numeric_only=True)
df.sort_values(by=['condition'])

fig, ax = plt.subplots(figsize=(5,6))
ax = sns.boxplot(x='condition', y='coms', data=df, hue='condition', fill=False,
            palette={'VIP LED off': "slategray", 'VIP LED on': "red",
                    'Control LED on': "coral", 'Control LED off': "lightgray"})
ax = sns.stripplot(x='condition', y='coms', data=df, hue='condition',
            palette={'VIP LED off': "slategray", 'VIP LED on': "red",
                    'Control LED on': "coral", 'Control LED off': "lightgray"},
            s = 7)
ax.spines[['top','right']].set_visible(False)

vipopto = df.loc[df.index.get_level_values('condition')=='VIP LED on', 'coms'].values
vipoff = df.loc[df.index.get_level_values('condition')=='VIP LED off', 'coms'].values
ctrlopto = df.loc[df.index.get_level_values('condition')=='Control LED on', 'coms'].values
ctrloff = df.loc[df.index.get_level_values('condition')=='Control LED off', 'coms'].values
scipy.stats.f_oneway(vipopto, vipoff, ctrlopto, ctrloff)
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([vipopto, vipoff, ctrlopto, ctrloff])#,p_adjust='holm-sidak')
print(p_values)

# fig, ax = plt.subplots(figsize=(2.5,6))
# ax = sns.barplot(x='condition', y='rewloc_shift', data=df, hue='condition', fill=False)
# ax = sns.stripplot(x='condition', y='rewloc_shift', data=df, hue='condition')


#%%
nans, x= nan_helper(y)
y[nans]= np.interp(x(nans), x(~nans), y[~nans])
# cluster 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(y)

# Choose the number of components
n_components = 4# Example: 5 components

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