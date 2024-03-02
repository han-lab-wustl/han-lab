"""zahra's analysis for clustering/dimensionality reduction of pyramidal cell data

"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
#%%

days_cnt_an1 = 10; days_cnt_an2=9; days_cnt_an3=8
animals = np.hstack([['e218']*(days_cnt_an1), ['e216']*(days_cnt_an2), \
                    ['e201']*(days_cnt_an3)])
days = np.array([20,21,22,23, 35, 38, 41, 44, 47,50,7,8,9,37, 41, 48, 50, 54,57,52,53,54,55,56,57,58,59])#[20,21,22,23]#
optoep = np.array([-1,-1,-1,-1, 3, 2, 3, 2,3, 2,-1,-1,-1,2, 3, 3, 2, 3,2,-1, -1, -1, 2, 3, 0, 2, 3])#[2,3,2,3]
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing
#%%
track_length = 270
# com shift analysis
dcts = []
for dd,day in enumerate(days):
    dct = {}
    animal = animals[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials'])
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = optoep[dd]
    if optoep[dd]<2: eptest = random.randint(2,3)    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    eps = np.append(eps, len(changeRewLoc))    
    if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    bin_size = 3    
    tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_early_trials'][0][comp[0]]]))
    tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_early_trials'][0][comp[1]]]))
    tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_late_trials'][0][comp[0]]]))
    tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_late_trials'][0][comp[1]]]))    
    # replace nan coms
    peak = np.nanmax(tc1_late,axis=1)
    coms1_max = np.array([np.where(tc1_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
    peak = np.nanmax(tc2_late,axis=1)
    coms2_max = np.array([np.where(tc2_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
    coms = fall['coms'][0]
    coms1 = np.hstack(coms[comp[0]])
    coms2 = np.hstack(coms[comp[1]])
    coms1[np.isnan(coms1)]=coms1_max[np.isnan(coms1)]
    coms2[np.isnan(coms2)]=coms2_max[np.isnan(coms2)]
    # take fc3 in area around com
    difftc1 = tc1_late-tc1_early
    coms1_bin = np.floor(coms1/bin_size).astype(int)
    difftc1 = np.array([np.nanmean(difftc1[ii,com-2:com+2]) for ii,com in enumerate(coms1_bin)])
    difftc2 = tc2_late-tc2_early
    coms2_bin = np.floor(coms2/bin_size).astype(int)
    difftc2 = np.array([np.nanmean(difftc2[ii,com-2:com+2]) for ii,com in enumerate(coms2_bin)])

    # # Define a threshold for differential inactivation (e.g., 0.2 difference in normalized mean activity)
    # threshold = 15   # 15 for e218? 
    # # Find differentially inactivated cells
    # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1[:, :int(rewlocs[comp[0]]/bin_size)-3], tc2[:, :int(rewlocs[comp[1]]/bin_size)-3], threshold, bin_size)
    # differentially_activated_cells = find_differentially_activated_cells(tc1[:, :int(rewlocs[comp[0]]/bin_size)-3], tc2[:, :int(rewlocs[comp[1]]/bin_size)-3], threshold, bin_size)
    # pre and post reward relative is diff
    rel_coms1 = np.hstack([(coms1[coms1<=rewlocs[comp[0]]]-rewlocs[comp[0]])/rewlocs[comp[0]],(coms1[coms1>rewlocs[comp[0]]]-rewlocs[comp[0]])/(track_length-rewlocs[comp[0]])])
    rel_coms2 = np.hstack([(coms2[coms2<=rewlocs[comp[1]]]-rewlocs[comp[1]])/rewlocs[comp[1]],(coms2[coms2>rewlocs[comp[1]]]-rewlocs[comp[1]])/(track_length-rewlocs[comp[1]])])
    # rel_coms2 = (coms2-rewlocs[comp[1]])/rewlocs[comp[1]]
    dct['rel_coms1'] = rel_coms1
    dct['rel_coms2'] = rel_coms2
    dct['learning_tc1'] = [tc1_early, tc1_late]
    dct['learning_tc2'] = [tc2_early, tc2_late]
    dct['difftc1'] = difftc1
    dct['difftc2'] = difftc2
    dct['rewzones_comp'] = rewzones[comp]
    dct['coms1'] = coms1
    dct['coms2'] = coms2
    # coms = fall['coms'][0] # keep original coms temp    
    # coms1 = np.hstack(coms[comp[0]])
    # coms2 = np.hstack(coms[comp[1]])
    dct['frac_place_cells_tc1'] = sum((coms1>(rewlocs[comp[0]]-(track_length*.07))) & (coms1<(rewlocs[comp[0]])+5))/len(coms1[(coms1>bin_size) & (coms1<(track_length/bin_size)-bin_size)])
    dct['frac_place_cells_tc2'] = sum((coms2>(rewlocs[comp[1]]-(track_length*.07))) & (coms2<(rewlocs[comp[1]])+5))/len(coms2[(coms2>bin_size) & (coms2<(track_length/bin_size)-bin_size)])
    # rewlocs[comp[0]]
    dcts.append(dct)
#%%
# plot fraction of place cells
dcts_opto = np.array(dcts)[optoep>1]
dfs=[]; dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    diff_rel_coms1=[dct['frac_place_cells_tc1']]
    diff_rel_coms2=[dct['frac_place_cells_tc2']]
    df = pd.DataFrame(np.hstack([diff_rel_coms1, diff_rel_coms2]), columns = ['frac_pc'])
    df['animal'] = animals[optoep>1][ii]
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['rewzones'] = np.hstack([[f'rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)    
bigdf.groupby(['animal','opto', 'rewzones']).mean()

plt.figure()
ax = sns.stripplot(x="condition", y="frac_pc", hue="opto", data=bigdf)
ax.tick_params(axis='x', labelrotation=90)

import scipy
from scipy.stats import ttest_rel, ttest_ind, ranksums
plt.figure()
ax = sns.stripplot(x="opto", y="frac_pc", hue='rewzones', data=bigdf)
scipy.stats.ttest_rel(bigdf[(bigdf.opto==True)].frac_pc.values, bigdf[(bigdf.opto==False)].frac_pc.values)
scipy.stats.mannwhitneyu(bigdf[(bigdf.opto==True) & (bigdf.rewzones=='rz_1.0')].frac_pc.values, bigdf[(bigdf.opto==False) &  (bigdf.rewzones=='rz_1.0')].frac_pc.values)

#%%
# plot coms of enriched cells    
dcts_opto = np.array(dcts)[optoep>1]

dfs=[]; dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    diff_rel_coms1=(dct['rel_coms1'][dct['difftc1']>1e-3])
    diff_rel_coms2=(dct['rel_coms2'][dct['difftc2']>1e-3])    
    df = pd.DataFrame(np.hstack([diff_rel_coms1, diff_rel_coms2]), columns = ['relative_com'])
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['rewzones'] = np.hstack([[f'rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['animal'] = animals[optoep>1][ii]
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)
# test 
ttest_ind(bigdf[(bigdf.opto==True) & (bigdf.rewzones=='rz_1.0')].relative_com.values, bigdf[(bigdf.opto==False) &  (bigdf.rewzones=='rz_1.0')].relative_com.values)
    

plt.figure()
ax = sns.stripplot(x="rewzones", y="relative_com", hue="opto", data=bigdf, size=1)
ax = sns.boxplot(x="rewzones", y="relative_com", hue="opto", data=bigdf, color='w')
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')

plt.figure()
ax = sns.stripplot(x="animal", y="relative_com", hue="opto", data=bigdf, size=1)
ax = sns.boxplot(x="animal", y="relative_com", hue="opto", data=bigdf, color='w')
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
#%%
# average enriched cells
dfs_diff = []
for ii,dct in enumerate(dcts):
    diff1=dct['difftc1'][dct['difftc1']>0]
    diff2=dct['difftc2'][dct['difftc1']>0]    
    df = pd.DataFrame(np.hstack([diff1, diff2]), columns = ['tc_diff'])
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff2)])
    if optoep[ii]>1:    
        df['opto'] = np.hstack([[False]*len(diff1),[True]*len(diff2)])
    else: 
        df['opto'] = [False]*len(df)
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff)    
# ax = sns.stripplot(x="condition", y="relative_com", hue="opto", data=bigdf, size=1)
plt.figure()
ax = sns.barplot(x="opto", y="tc_diff",data=bigdf)
ax.tick_params(axis='x', labelrotation=90)

# number of enriched cells
dfs_diff_o = np.array(dfs_diff)[np.array(optoep)>1]
bigdf = pd.concat(dfs_diff_o)    
plt.figure()
plt.bar([1,2],[bigdf.groupby("opto").count().tc_diff[0],bigdf.groupby("opto").count().tc_diff[1]])

#%%
optoep = np.array(optoep)
com_shift = np.array(com_shift)
rewloc_shift = np.array(rewloc_shift)
plt.scatter(com_shift[optoep<1, 0], rewloc_shift[optoep<1], label = 'Control Inactive')
plt.scatter(com_shift[optoep>=2, 0], rewloc_shift[optoep>=2], label = 'VIP Inactive')
plt.scatter(com_shift[optoep<1, 1], rewloc_shift[optoep<1], label = 'Control Active')
plt.scatter(com_shift[optoep>=2, 1], rewloc_shift[optoep>=2], label = 'VIP Active')
# plt.scatter(com_shift[optoep<2, 2], rewloc_shift[optoep<2], label = 'Control All')
# plt.scatter(com_shift[optoep>=2, 2], rewloc_shift[optoep>=2], label = 'VIP All')
plt.legend()
plt.ylabel('Change in Rew Loc (cm)')
plt.xlabel('Median COM Shift (cm)')
#%%
# active vs. inactive place cell fraction
plt.figure()
in_active_cells = np.array(in_active_cells)
optoep = np.array(optoep)
plt.boxplot(np.array(in_active_cells[optoep<0]))
plt.xticks(ticks = [1,2], labels = ['Active', 'Inactive'])
plt.ylabel('Fraction of Place Cells')
# plt.ylim(0, .2)
# plt.ylim(0, .12)
plt.figure()
plt.boxplot(np.array(in_active_cells[optoep>=2]))
plt.xticks(ticks = [1,2], labels = ['Active', 'Inactive'])
# plt.ylim(0, .2)
# plt.ylim(0, .12)
plt.ylabel('Fraction of Place Cells')
#%% 
# com shift clusters
inactive_coms2_days = np.array(inactive_coms2_days); inactive_coms1_days = np.array(inactive_coms1_days)
optoep = np.array(optoep)
for i in range(4):    
    inactive_coms2_days_opto = inactive_coms2_days[optoep>1]; inactive_coms1_days_opto = inactive_coms1_days[optoep>1]
    com_differences_opto = np.hstack(inactive_coms2_days_opto[i])-np.hstack(inactive_coms1_days_opto[i])
    inactive_coms2_days_ctrl = inactive_coms2_days[optoep<=1]; inactive_coms1_days_ctrl = inactive_coms1_days[optoep<=1]
    com_differences_ctrl = np.hstack(inactive_coms2_days_ctrl[i])-np.hstack(inactive_coms1_days_ctrl[i])
    # Cluster cells based on their center of mass changes using k-means
    num_clusters = 3  # You can choose this number based on domain knowledge or use methods like the elbow method to decide
    if i==0:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(com_differences_opto.reshape(-1, 1))
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        prop_opto = counts/len(kmeans.labels_)
    else:
        kmeans_lbls = kmeans.predict(com_differences_opto.reshape(-1, 1))
        unique, counts = np.unique(kmeans_lbls, return_counts=True)
        prop_opto = counts/len(kmeans_lbls)
    # Plotting the results
    # Let's visualize the mean center of mass change for each cluster
    plt.figure()
    for cluster_idx in range(num_clusters):
        cluster_member_indices = np.where(kmeans_lbls == cluster_idx)[0]
        plt.scatter(np.hstack(inactive_coms1_days_opto[i])[cluster_member_indices],np.hstack(inactive_coms2_days_opto[i])[cluster_member_indices], label=f'Cluster {cluster_idx + 1}')
    plt.legend()
    plt.xlabel('Prev Ep COM')
    plt.ylabel('Opto COM')
    plt.title('Inactivated Cells')
    plt.ylim(-1,1)
    plt.xlim(-1,1)

    kmeans_lbls_ctrl = kmeans.predict(com_differences_ctrl.reshape(-1, 1))
    unique, counts = np.unique(kmeans_lbls_ctrl, return_counts=True)
    prop_ctrl = counts/len(kmeans_lbls_ctrl)

    # Plotting the results
    # Let's visualize the mean center of mass change for each cluster
    plt.figure()
    for cluster_idx in range(num_clusters):
        cluster_member_indices = np.where(kmeans_lbls_ctrl == cluster_idx)[0]
        plt.scatter(np.hstack(inactive_coms1_days_ctrl[i])[cluster_member_indices],np.hstack(inactive_coms2_days_ctrl[i])[cluster_member_indices], label=f'Cluster {cluster_idx + 1}')
    plt.legend()
    plt.xlabel('Prev Ep COM')
    plt.ylabel('Opto COM')
    plt.title('Inactivated Cells - Control Days')
    plt.ylim(-1,1)
    plt.xlim(-1,1)

#%%
# com shift clusters
active_coms2_days = np.array(active_coms2_days); active_coms1_days = np.array(active_coms1_days)
optoep = np.array(optoep)
active_coms2_days_opto = active_coms2_days[optoep>1]; active_coms1_days_opto = active_coms1_days[optoep>1]
com_differences_opto = np.hstack(active_coms2_days_opto)-np.hstack(active_coms1_days_opto)
active_coms2_days_ctrl = active_coms2_days[optoep<1]; active_coms1_days_ctrl = active_coms1_days[optoep<1]
com_differences_ctrl = np.hstack(active_coms2_days_ctrl)-np.hstack(active_coms1_days_ctrl)

# Cluster cells based on their center of mass changes using k-means
num_clusters = 4  # You can choose this number based on domain knowledge or use methods like the elbow method to decide
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(com_differences_opto.reshape(-1, 1))
unique, counts = np.unique(kmeans.labels_, return_counts=True)
prop_opto = counts/len(kmeans.labels_)

# Plotting the results
# Let's visualize the mean center of mass change for each cluster
plt.figure()
for cluster_idx in range(num_clusters):
    cluster_member_indices = np.where(kmeans.labels_ == cluster_idx)[0]
    plt.scatter(np.hstack(active_coms1_days_opto)[cluster_member_indices],np.hstack(active_coms2_days_opto)[cluster_member_indices], label=f'Cluster {cluster_idx + 1}')
plt.legend()
plt.xlabel('Prev Ep COM')
plt.ylabel('Opto COM')
plt.title('Activated Cells')

kmeans_lbls_ctrl = kmeans.predict(com_differences_ctrl.reshape(-1, 1))
unique, counts = np.unique(kmeans_lbls_ctrl, return_counts=True)
prop_ctrl = counts/len(kmeans_lbls_ctrl)

# Plotting the results
# Let's visualize the mean center of mass change for each cluster
plt.figure()
for cluster_idx in range(num_clusters):
    cluster_member_indices = np.where(kmeans_lbls_ctrl == cluster_idx)[0]
    plt.scatter(np.hstack(active_coms1_days_ctrl)[cluster_member_indices],np.hstack(active_coms2_days_ctrl)[cluster_member_indices], label=f'Cluster {cluster_idx + 1}')
plt.legend()
plt.xlabel('Prev Ep COM')
plt.ylabel('Opto COM')
plt.title('Activated Cells - Control Days')
