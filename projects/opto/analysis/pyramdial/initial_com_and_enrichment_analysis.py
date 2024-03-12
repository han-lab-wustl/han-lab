"""zahra's analysis for clustering/dimensionality reduction of pyramidal cell data

"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, find_differentially_inactivated_cells

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
#%%

# import condition df
conddf = pd.read_csv(r"Z:\conddf_neural.csv", index_col=None)

#%%
track_length = 270
# com shift analysis
dcts = []
for dd,day in enumerate(conddf.days.values):
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials'])
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
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
    threshold = 5   # 15 for e218? 
    # Find differentially inactivated cells
    # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    # differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
    differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late, tc2_late, threshold, bin_size)
    differentially_activated_cells = find_differentially_activated_cells(tc1_late, tc2_late, threshold, bin_size)
    
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
    dct['frac_place_cells_tc1'] = sum((coms1>(rewlocs[comp[0]]-(track_length*.07))) & (coms1<(rewlocs[comp[0]])+5))/len(coms1[(coms1>bin_size) & (coms1<=(track_length/bin_size))])
    dct['frac_place_cells_tc2'] = sum((coms2>(rewlocs[comp[1]]-(track_length*.07))) & (coms2<(rewlocs[comp[1]])+5))/len(coms2[(coms2>bin_size) & (coms2<=(track_length/bin_size))])
    dct['rewloc_shift'] = rewloc_shift
    dct['com_shift'] = com_shift
    dct['inactive'] = differentially_inactivated_cells
    dct['active'] = differentially_activated_cells
    dct['rewlocs_comp'] = rewlocs[comp]
    # rewlocs[comp[0]]
    dcts.append(dct)
#%%
# plot fraction of place cells
optoep = conddf.optoep.values; animals = conddf.animals.values; in_type = conddf.in_type.values
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

fig,ax = plt.subplots()
ax = sns.stripplot(x="condition", y="frac_pc", hue="opto", data=bigdf)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

in_type_cond = 'vip'
fig,ax = plt.subplots()
ax = sns.stripplot(x="opto", y="frac_pc", hue='rewzones', data=bigdf[bigdf.in_type == in_type_cond],
            palette='rocket')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

bigdf = bigdf[bigdf.in_type == in_type_cond]
scipy.stats.ttest_rel(bigdf[(bigdf.opto==True)].frac_pc.values, \
            bigdf[(bigdf.opto==False)].frac_pc.values)
scipy.stats.ranksums(bigdf[(bigdf.opto==True)].frac_pc.values, bigdf[(bigdf.opto==False)].frac_pc.values)

#%%
# plot coms of enriched cells    
dcts_opto = np.array(dcts)[optoep>1]

dfs=[]; dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    diff_rel_coms1=(dct['rel_coms1'][dct['difftc1']<0])
    diff_rel_coms2=(dct['rel_coms2'][dct['difftc2']<0])    
    df = pd.DataFrame(np.hstack([diff_rel_coms1, diff_rel_coms2]), columns = ['relative_com'])
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['rewzones'] = np.hstack([[f'rz_{dct["rewzones_comp"][0]}']*len(diff_rel_coms1), [f'rz_{dct["rewzones_comp"][1]}']*len(diff_rel_coms2)])
    df['animal'] = animals[optoep>1][ii]
    df['in_type'] = in_type[optoep>1][ii]
    if in_type[optoep>1][ii]=='vip':
        df['vip_cond'] = 'vip'
    else:
        df['vip_cond'] = "ctrl"
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)

intype = 'sst'
fig,ax = plt.subplots()
ax = sns.stripplot(x="rewzones", y="relative_com", hue="opto", 
    data=bigdf[bigdf.in_type == intype], size=1, palette={False: "slategray", True: "red"})
ax = sns.boxplot(x="rewzones", y="relative_com", hue="opto",
fill=False, data=bigdf[bigdf.in_type == intype], palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig,ax = plt.subplots()
ax = sns.stripplot(x="animal", y="relative_com", 
    hue="opto", data=bigdf, size=1, palette={False: "slategray", True: "red"})
ax = sns.boxplot(x="animal", y="relative_com", hue="opto", 
    data=bigdf, fill=False, palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig,ax = plt.subplots()
ax = sns.stripplot(x="in_type", y="relative_com", 
    hue="opto", data=bigdf, size=1, palette={False: "slategray", True: "red"})
ax = sns.barplot(x="in_type", y="relative_com", hue="opto", 
    data=bigdf, fill=False, palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# look at specific animals
an = 'e200'
fig,ax = plt.subplots()
ax = sns.stripplot(x="rewzones", y="relative_com", hue="opto", 
    data=bigdf[bigdf.animal == an], size=1, palette={False: "slategray", True: "red"})
ax = sns.boxplot(x="rewzones", y="relative_com", hue="opto",
fill=False, data=bigdf[bigdf.animal == an], palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.axhline(0, color = 'slategray', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

scipy.stats.ranksums(bigdf[(bigdf.opto==True) & (bigdf.vip_cond=='ctrl')].relative_com.values, 
bigdf[(bigdf.opto==False) &  (bigdf.vip_cond=='ctrl')].relative_com.values)

#%%
# # average enrichment
# dcts_opto = np.array(dcts)[optoep>1]

# dfs_diff = []
# for ii,dct in enumerate(dcts_opto):
#     diff1=dct['difftc1'][dct['difftc1']>1e-3]
#     diff2=dct['difftc2'][dct['difftc1']>1e-3]
#     df = pd.DataFrame(np.hstack([diff1, diff2]), columns = ['tc_diff'])
#     df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']*len(diff1), [f'day{ii}_tc2_rz_{dct["rewzones_comp"][1]}']*len(diff2)])
#     df['animal'] = animals[optoep>1][ii]
#     df['in_type'] = in_type[optoep>1][ii]
#     # if optoep[ii]>1:    
#     df['opto'] = np.hstack([[False]*len(diff1),[True]*len(diff2)])
#     # else: 
#     # df['opto'] = [False]*len(df)
#     dfs_diff.append(df)
# bigdf = pd.concat(dfs_diff,ignore_index=False) 
# bigdf.reset_index(drop=True, inplace=True)   
# # ax = sns.stripplot(x="condition", y="relative_com", hue="opto", data=bigdf, size=1)
# plt.figure()
# ax = sns.barplot(x="opto", y="tc_diff", hue="in_type",data=bigdf)
# # ax = sns.stripplot(x="opto", y="tc_diff", hue="in_type",data=bigdf)
# ax.tick_params(axis='x', labelrotation=90)
#%%
# com shift
optoep = conddf.optoep.values
in_type = conddf.in_type.values
in_type_cond = 'vip'
optoep_in = np.array([xx for ii,xx in enumerate(optoep) if in_type[ii]==in_type_cond])
com_shift = np.array([dct['com_shift'] for ii,dct in enumerate(dcts) if in_type[ii]==in_type_cond])
rewloc_shift = np.array([dct['rewloc_shift'] for ii,dct in enumerate(dcts) if in_type[ii]==in_type_cond ])
fig, ax = plt.subplots()
ax.scatter(com_shift[optoep_in<0, 0], rewloc_shift[optoep_in<0], label = 'Control Inactive', color = 'gold')
ax.scatter(com_shift[optoep_in>=2, 0], rewloc_shift[optoep_in>=2], label = 'VIP Inactive', color = 'red')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[optoep_in>=2, 0][~np.isnan(com_shift[optoep_in>=2, 0])],rewloc_shift[optoep_in>=2][~np.isnan(com_shift[optoep_in>=2, 0])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
# ax.plot(x_vals, y_vals, color = 'r')
ax.scatter(com_shift[optoep_in<0, 1], rewloc_shift[optoep_in<0], label = 'Control Active', color = 'darkgoldenrod')
ax.scatter(com_shift[optoep_in>=2, 1], rewloc_shift[optoep_in>=2], label = 'VIP Active', color = 'maroon')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[optoep_in>=2, 1][~np.isnan(com_shift[optoep_in>=2, 1])],rewloc_shift[optoep_in>=2][~np.isnan(com_shift[optoep_in>=2, 1])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
# ax.plot(x_vals, y_vals, color = 'maroon')
# ax.set_ylim(-150,200)
# plt.scatter(com_shift[optoep_in<2, 2], rewloc_shift[optoep_in<2], label = 'Control All')

# plt.scatter(com_shift[optoep_in>=2, 2], rewloc_shift[optoep_in>=2], label = 'SST All')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(bbox_to_anchor=(1.1, 1.1))
ax.set_ylabel('Change in Rew Loc (cm)')
ax.set_xlabel('Mean COM Shift (cm)')
ax.set_title('Shift = Opto Epoch - Previous Epoch')
#%%
# inactivate cells 
dcts_opto = np.array(dcts)

dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    inactive_frac = len(dct['inactive'])/len(dct['coms1'])
    active_frac = len(dct['active'])/len(dct['coms1'])    
    df = pd.DataFrame(np.hstack([inactive_frac]), columns = ['inactive_frac'])
    df['active_frac'] = active_frac
    df['condition'] = np.hstack([[f'day{ii}_tc1_rz_{dct["rewzones_comp"][0]}']])
    df['animal'] = conddf.animals.values[ii]
    df['in_type'] = conddf.in_type.values[ii]    
    df['opto'] = bool(conddf.optoep.values[ii]>1) # true vs. false
    if df['in_type'].values[0] =='vip':
        df['vip_cond'] = 'vip'
    elif df['in_type'].values[0] =='sst':
        df['vip_cond'] = 'ctrl'
    # if optoep[ii]>1:        
    # else: 
    # df['opto'] = [False]*len(df)
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff,ignore_index=False) 
bigdf.reset_index(drop=True, inplace=True)   
#%%
# plot fraction of inactivated vs. activated cells
plt.figure()
ax = sns.barplot(x="opto", y="inactive_frac",hue='vip_cond', data=bigdf,fill=False,
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y="inactive_frac",hue='vip_cond', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"})

ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.figure()
ax = sns.barplot(x="opto", y="active_frac", hue='vip_cond', data=bigdf,fill=False,
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y="active_frac",hue='vip_cond', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# sig
scipy.stats.mannwhitneyu(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'inactive_frac'].values,
                    bigdf.loc[(bigdf.vip_cond=='ctrl') & (bigdf.opto==True), 'inactive_frac'].values)

scipy.stats.mannwhitneyu(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'active_frac'].values,
                    bigdf.loc[(bigdf.vip_cond=='ctrl') & (bigdf.opto==True), 'active_frac'].values)

# %%
# check inactive vs. active cell tuning
dd = 4
day = conddf.days.values[dd]
animal = conddf.animals.values[dd]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
    'tuning_curves_late_trials', 'coms_early_trials'])
changeRewLoc = np.hstack(fall['changeRewLoc'])
eptest = conddf.optoep.values[dd]
if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
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

#%%
# compare early vs. late tuning
dct = dcts[dd]
arr_early = dct['learning_tc2'][0][dct['inactive']]
arr_late = dct['learning_tc2'][1][dct['inactive']]
arr_early = arr_early[np.argsort(dct['coms2'][dct['inactive']])]
arr_late = arr_late[np.argsort(dct['coms2'][dct['inactive']])]
fig, axes = plt.subplots(2,1)
axes[0].imshow(arr_early); axes[0].axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w')
axes[0].axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w', linestyle='--'); 
axes[1].imshow(arr_late); axes[1].axvline(dct['rewlocs_comp'][1]/bin_size, color = 'w'); 
axes[1].axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w', linestyle='--'); 
axes[1].set_xlabel('Spatial bins')
axes[0].set_ylabel('Cells')

# comparse previous ep vs. next ep tuning (late)
arr_prev_ep_late = dct['learning_tc1'][1][dct['inactive']]
arr_prev_ep_late = arr_prev_ep_late[np.argsort(dct['coms1'][dct['inactive']])]
fig, axes = plt.subplots(2,1)
axes[0].imshow(arr_prev_ep_late); axes[0].axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w')
axes[1].imshow(arr_late); axes[1].axvline(dct['rewlocs_comp'][1]/bin_size, color = 'w'); 
axes[1].set_xlabel('Spatial bins')
axes[0].set_ylabel('Cells')

#%%
# enriched cells
arr_early = dct['learning_tc2'][0][dct['active']]
arr_late = dct['learning_tc2'][1][dct['active']]
arr_early = arr_early[np.argsort(dct['coms2'][dct['active']])]
arr_late = arr_late[np.argsort(dct['coms2'][dct['active']])]
plt.figure(); plt.imshow(arr_early); plt.axvline(dct['rewlocs_comp'][1]/bin_size, color = 'w'); plt.axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w', linestyle='--'); 
plt.figure(); plt.imshow(arr_late); plt.axvline(dct['rewlocs_comp'][1]/bin_size, color = 'w'); plt.axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w', linestyle='--'); 
# prev epoch
arr_early = dct['learning_tc1'][0][dct['active']]
arr_late = dct['learning_tc1'][1][dct['active']]
arr_early = arr_early[np.argsort(dct['coms1'][dct['active']])]
arr_late = arr_late[np.argsort(dct['coms1'][dct['active']])]
plt.figure(); plt.imshow(arr_early); plt.axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w'); 
plt.figure(); plt.imshow(arr_late); plt.axvline(dct['rewlocs_comp'][0]/bin_size, color = 'w'); 

#%%
other_ep_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_early_trials'][0][0]]))
other_ep_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_late_trials'][0][0]]))    
# other epoch
arr_early = other_ep_early[dct['active']]
arr_late = other_ep_late[dct['active']]
arr_early = arr_early[np.argsort(np.hstack(coms[0][dct['active']]))]
arr_late = arr_late[np.argsort(np.hstack(coms[0][dct['active']]))]
plt.figure(); plt.imshow(arr_early); plt.axvline(rewlocs[0]/bin_size, color = 'w'); 
plt.figure(); plt.imshow(arr_late); plt.axvline(rewlocs[0]/bin_size, color = 'w'); 


# %%
