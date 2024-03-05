"""zahra's analysis for clustering/dimensionality reduction of pyramidal cell data

"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, find_differentially_inactivated_cells

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
#%%

days_cnt_an1 = 10; days_cnt_an2=9; days_cnt_an3=24; days_cnt_an4=11; days_cnt_an5=7; days_cnt_an6=9
days_cnt_an7=19
animals = np.hstack([['e218']*(days_cnt_an1), ['e216']*(days_cnt_an2), \
                    ['e201']*(days_cnt_an3), ['e186']*(days_cnt_an4), ['e190']*(days_cnt_an5), ['e189']*(days_cnt_an6), \
                        ['e200']*days_cnt_an7])
in_type = np.hstack([['vip']*(days_cnt_an1), ['vip']*(days_cnt_an2), \
                    ['sst']*(days_cnt_an3), ['pv']*(days_cnt_an4), ['ctrl']*(days_cnt_an5), ['ctrl']*(days_cnt_an6), \
                        ['sst']*days_cnt_an7])
days = np.array([20,21,22,23, 35, 38, 41, 44, 47,50,7,8,9,37, 41, 48, \
                50, 54,57,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,\
                2,3,4,5,31,32,33,34,36,37,40,33,34,35,40,41,42,45,35,36,37,38,39,40,41,42,44, \
                    65,66,67,68,69,70,72,73,74,76,81,82,83,84,85,86,87,88,89])#[20,21,22,23]#
optoep = np.array([-1, -1, -1, -1,  3,  2,  3,  2,  3,  2, -1, -1, -1,  2,  3,  3,  2,
        3,  2, -1, -1, -1,  2,  3,  0,  2,  3,  0,  2,  3,  0,  2,  3,  0,
        2,  3,  0,  2,  3,  0,  2,  3,  3, -1, -1, -1, -1,  2,  3,  2,  3,
        2,  3,  2, -1, -1, -1,  3,  0,  1,  3, -1, -1, -1, -1,  2,  3,  2,
        0,  2,  2,  3,  0,  2,  3,  0,  3,  0,  2,  0,  2,  3,  0,  2,  3,
        0,  2,  3,  0])#[2,3,2,3]
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
    threshold = 15   # 15 for e218? 
    # Find differentially inactivated cells
    differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    rewloc_shift = rewlocs[comp[1]]-rewlocs[comp[0]]
    com_shift = [np.nanmean(coms[comp[1]][differentially_inactivated_cells]-coms[comp[0]][differentially_inactivated_cells]), \
                np.nanmean(coms[comp[1]][differentially_activated_cells]-coms[comp[0]][differentially_activated_cells]), \
                    np.nanmean(coms[comp[1]]-coms[comp[0]])]
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
    dct['frac_place_cells_tc1'] = sum((coms1>(rewlocs[comp[0]]-(track_length*.07))) & (coms1<(rewlocs[comp[0]])+5))/len(coms1[(coms1>bin_size) & (coms1<=(track_length/bin_size)-bin_size)])
    dct['frac_place_cells_tc2'] = sum((coms2>(rewlocs[comp[1]]-(track_length*.07))) & (coms2<(rewlocs[comp[1]])+5))/len(coms2[(coms2>bin_size) & (coms2<=(track_length/bin_size)-bin_size)])
    dct['rewloc_shift'] = rewloc_shift
    dct['com_shift'] = com_shift
    dct['inactive'] = differentially_inactivated_cells
    dct['active'] = differentially_activated_cells
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
    df['in_type'] = in_type[optoep>1][ii]
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['rewzones'] = np.hstack([[f'rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)    
bigdf.groupby(['in_type','opto', 'rewzones']).mean()

plt.figure()
ax = sns.stripplot(x="condition", y="frac_pc", hue="opto", data=bigdf)
ax.tick_params(axis='x', labelrotation=90)

in_type_cond = 'vip'
plt.figure()
ax = sns.stripplot(x="opto", y="frac_pc", hue='rewzones', data=bigdf[bigdf.in_type == in_type_cond])
bigdf = bigdf[bigdf.in_type == in_type_cond]
scipy.stats.ttest_rel(bigdf[(bigdf.opto==True)].frac_pc.values, \
            bigdf[(bigdf.opto==False)].frac_pc.values)
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
    df['in_type'] = in_type[optoep>1][ii]
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)
# test 
scipy.stats.ttest_ind(bigdf[(bigdf.opto==True) & (bigdf.rewzones=='rz_1.0')].relative_com.values, bigdf[(bigdf.opto==False) &  (bigdf.rewzones=='rz_1.0')].relative_com.values)
    
plt.figure()
ax = sns.stripplot(x="rewzones", y="relative_com", hue="opto", data=bigdf[bigdf.in_type == 'sst'], size=1)
ax = sns.boxplot(x="rewzones", y="relative_com", hue="opto",fill=False, data=bigdf[bigdf.in_type == 'sst'])
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')

plt.figure()
ax = sns.stripplot(x="animal", y="relative_com", hue="opto", data=bigdf, size=1)
ax = sns.boxplot(x="animal", y="relative_com", hue="opto", data=bigdf, fill=False)
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')


plt.figure()
ax = sns.stripplot(x="in_type", y="relative_com", hue="opto", data=bigdf, size=1)
ax = sns.boxplot(x="in_type", y="relative_com", hue="opto", data=bigdf, fill=False)
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
#%%
# average enrichment
dcts_opto = np.array(dcts)[optoep>1]

dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    diff1=dct['difftc1'][dct['difftc1']>1e-3]
    diff2=dct['difftc2'][dct['difftc1']>1e-3]
    df = pd.DataFrame(np.hstack([diff1, diff2]), columns = ['tc_diff'])
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff2)])
    df['animal'] = animals[optoep>1][ii]
    df['in_type'] = in_type[optoep>1][ii]
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff1),[True]*len(diff2)])
    # else: 
    # df['opto'] = [False]*len(df)
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff,ignore_index=False) 
bigdf.reset_index(drop=True, inplace=True)   
# ax = sns.stripplot(x="condition", y="relative_com", hue="opto", data=bigdf, size=1)
plt.figure()
ax = sns.barplot(x="opto", y="tc_diff", hue="in_type",data=bigdf)
ax.tick_params(axis='x', labelrotation=90)
#%%
# number of enriched cells
dfs_diff_o = np.array(dfs_diff)
bigdf = pd.concat(dfs_diff_o)   
bigdf = bigdf[bigdf.in_type == 'pv'] 
plt.figure()
plt.bar([1,2],[bigdf.groupby("opto").count().tc_diff[0],bigdf.groupby("opto").count().tc_diff[1]])

#%%
# com shift
in_type_cond = 'ctrl'
optoep_in = np.array([xx for ii,xx in enumerate(optoep) if in_type[ii]==in_type_cond])
com_shift = np.array([dct['com_shift'] for ii,dct in enumerate(dcts) if in_type[ii]==in_type_cond])
rewloc_shift = np.array([dct['rewloc_shift'] for ii,dct in enumerate(dcts) if in_type[ii]==in_type_cond ])
plt.scatter(com_shift[optoep_in<1, 0], rewloc_shift[optoep_in<1], label = 'Control Inactive')
plt.scatter(com_shift[optoep_in>=2, 0], rewloc_shift[optoep_in>=2], label = 'SST Inactive')
plt.scatter(com_shift[optoep_in<1, 1], rewloc_shift[optoep_in<1], label = 'Control Active')
plt.scatter(com_shift[optoep_in>=2, 1], rewloc_shift[optoep_in>=2], label = 'SST Active')
# plt.scatter(com_shift[optoep_in<2, 2], rewloc_shift[optoep_in<2], label = 'Control All')
# plt.scatter(com_shift[optoep_in>=2, 2], rewloc_shift[optoep_in>=2], label = 'SST All')

plt.legend()
plt.ylabel('Change in Rew Loc (cm)')
plt.xlabel('Mean COM Shift (cm)')
plt.title('Shift = Opto Ep - Previous Ep')
#%%
# inactivate cells tuning curves
dcts_opto = np.array(dcts)

dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    inactive_frac = len(dct['inactive'])/len(dct['coms1'])
    active_frac = len(dct['active'])/len(dct['coms1'])    
    df = pd.DataFrame(np.hstack([inactive_frac]), columns = ['inactive_frac'])
    df['active_frac'] = active_frac
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']])
    df['animal'] = animals[ii]
    df['in_type'] = in_type[ii]
    df['opto'] = bool(optoep[ii]>1) # true vs. false
    # if optoep[ii]>1:        
    # else: 
    # df['opto'] = [False]*len(df)
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff,ignore_index=False) 
bigdf.reset_index(drop=True, inplace=True)   
plt.figure()
ax = sns.barplot(x="opto", y="inactive_frac",hue='in_type', data=bigdf)
ax.tick_params(axis='x', labelrotation=90)
plt.figure()
ax = sns.barplot(x="opto", y="active_frac", hue='in_type', data=bigdf)
ax.tick_params(axis='x', labelrotation=90)

bigdf = bigdf[(bigdf.opto == True)]
scipy.stats.kruskal(bigdf.loc[bigdf.in_type=='vip', 'inactive_frac'].values, \
                    bigdf.loc[bigdf.in_type=='vip', 'active_frac'].values, \
                    bigdf.loc[bigdf.in_type!='vip', 'inactive_frac'].values, \
                    bigdf.loc[bigdf.in_type!='vip', 'active_frac'].values)
bi
import scikit_posthocs as sp
# using the posthoc_dunn() function
p_values= sp.posthoc_dunn([bigdf.loc[bigdf.in_type=='vip', 'inactive_frac'].values, \
                    bigdf.loc[bigdf.in_type=='vip', 'active_frac'].values, \
                    bigdf.loc[bigdf.in_type!='vip', 'inactive_frac'].values, \
                    bigdf.loc[bigdf.in_type!='vip', 'active_frac'].values], p_adjust = 'holm')

print(p_values)
# %%
