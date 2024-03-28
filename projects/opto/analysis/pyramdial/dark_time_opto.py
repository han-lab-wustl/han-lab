"""
zahra's analysis for vip dark time experiments
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto
import matplotlib.backends.backend_pdf

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\conddf_dark_time_only.csv", index_col=None)

#%%

dcts = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    threshold = 10
    dct = get_pyr_metrics_opto(conddf, dd, day, threshold=threshold)
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
    df['rewzones_transition'] = f'rz_{dct["rewzones_comp"][0].astype(int)}-{dct["rewzones_comp"][1].astype(int)}'
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff_rel_coms1),[True]*len(diff_rel_coms2)])
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)    

bigdf.groupby(['in_type','opto', 'rewzones']).mean()

in_type_cond = 'vip'
fig,ax = plt.subplots()
ax = sns.stripplot(x="opto", y="frac_pc", hue='rewzones_transition', data=bigdf[bigdf.in_type == in_type_cond],
            palette='rocket')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

bigdf = bigdf[bigdf.in_type == in_type_cond]
scipy.stats.ttest_rel(bigdf[(bigdf.opto==True)].frac_pc.values, \
            bigdf[(bigdf.opto==False)].frac_pc.values)
scipy.stats.ranksums(bigdf[(bigdf.opto==True)].frac_pc.values, bigdf[(bigdf.opto==False)].frac_pc.values)

# %%
# average enrichment
dcts_opto = np.array(dcts)[optoep>1]

dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    diff1=dct['difftc1'][dct['difftc1']>1e-5]
    diff2=dct['difftc2'][dct['difftc1']>1e-5]
    df = pd.DataFrame(np.hstack([diff1, diff2]).astype(float), columns = ['tc_diff'])
    df['animal'] = animals[optoep>1][ii]
    # if optoep[ii]>1:    
    df['opto'] = np.hstack([[False]*len(diff1),[True]*len(diff2)])
    # else: 
    # df['opto'] = [False]*len(df)
    if in_type[optoep>1][ii] =='vip':
        df['vip_cond'] = 'vip'
    elif in_type[optoep>1][ii] !='pv':
        df['vip_cond'] = 'ctrl'
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff,ignore_index=False) 
bigdf.reset_index(drop=True, inplace=True)   
bigdf['vip_cond'] = bigdf['vip_cond'].astype(str)
bigdf['opto']= bigdf['opto'].astype(bool)
bigdf['animal'] = bigdf['animal'].astype(str)
# ax = sns.stripplot(x="condition", y="relative_com", hue="opto", data=bigdf, size=1)
# ax = sns.stripplot(x="opto", y="tc_diff", hue="in_type",data=bigdf)
# ax.tick_params(axis='x', labelrotation=90)
bigdf_test = bigdf.groupby(['animal', 'vip_cond', 'opto']).mean()
comp1 = bigdf_test[(bigdf_test.index.get_level_values('opto')==True) & (bigdf_test.index.get_level_values('vip_cond')=='vip')].tc_diff.values; comp1=comp1[~np.isnan(comp1)]
comp2 = bigdf_test[(bigdf_test.index.get_level_values('opto')==False) &  (bigdf_test.index.get_level_values('vip_cond')=='vip')].tc_diff.values; comp2=comp2[~np.isnan(comp2)]
diff_offon_vip = comp1-comp2
comp1 = bigdf_test[(bigdf_test.index.get_level_values('opto')==True) & (bigdf_test.index.get_level_values('vip_cond')=='ctrl')].tc_diff.values; comp1=comp1[~np.isnan(comp1)]
comp2 = bigdf_test[(bigdf_test.index.get_level_values('opto')==False) &  (bigdf_test.index.get_level_values('vip_cond')=='ctrl')].tc_diff.values; comp2=comp2[~np.isnan(comp2)]
diff_offon_ctrl = comp1-comp2
t,pval=scipy.stats.ranksums(diff_offon_vip, diff_offon_ctrl)

import itertools
plt.figure()
df = pd.DataFrame(np.concatenate([diff_offon_vip, diff_offon_ctrl]), columns = ['tc_diff_ledoff-on'])
cond = [['vip']*len(diff_offon_vip), ['ctrl']*len(diff_offon_ctrl)]
df['condition']=np.array(list(itertools.chain(*cond)))
ax = sns.barplot(x="condition", y="tc_diff_ledoff-on",data=df, fill=False, color='k')
sns.stripplot(x="condition", y="tc_diff_ledoff-on",data=df, color='k')
plt.title(f"p-value = {pval:03f}")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

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
    df['animal'] = conddf.animals.values[ii]
    df['opto'] = bool(conddf.optoep.values[ii]>1) # true vs. false
    df['rewzones_transition'] = f'rz_{dct["rewzones_comp"][0].astype(int)}-{dct["rewzones_comp"][1].astype(int)}'
    if conddf.in_type.values[ii]     =='vip':
        df['vip_cond'] = 'vip'
    # else:
    #     df['vip_cond'] = 'ctrl'

    elif (conddf.in_type.values[ii]     =='sst') or conddf.in_type.values[ii]     =='pv':
        df['vip_cond'] = 'ctrl'
    # if optoep[ii]>1:        
    # else: 
    # df['opto'] = [False]*len(df)
    dfs_diff.append(df)
bigdf_org = pd.concat(dfs_diff,ignore_index=False) 
bigdf_org.reset_index(drop=True, inplace=True)   

# plot fraction of inactivated vs. activated cells
bigdf_test = bigdf_org.groupby(['animal', 'vip_cond', 'opto']).quantile(.75,numeric_only=True)
bigdf = bigdf_org[bigdf_org.vip_cond=='vip'].groupby(['animal', 'opto', 'rewzones_transition']).quantile(.75,numeric_only=True)

plt.figure()
ax = sns.barplot(x="opto", y="inactive_frac",hue='vip_cond', data=bigdf_test,fill=False)
ax = sns.stripplot(x="opto", y="inactive_frac",hue='vip_cond', data=bigdf_test)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


plt.figure()
ax = sns.barplot(x="opto", y="inactive_frac",hue='rewzones_transition', data=bigdf,fill=False)
ax = sns.stripplot(x="opto", y="inactive_frac",hue='rewzones_transition', data=bigdf)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figure()
ax = sns.barplot(x="opto", y="active_frac", hue='rewzones_transition', data=bigdf,fill=False)
ax = sns.stripplot(x="opto", y="active_frac",hue='rewzones_transition', data=bigdf)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# # sig
# scipy.stats.ttest_ind(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'inactive_frac'].values,
#                     bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==False), 'inactive_frac'].values)
# scipy.stats.ttest_ind(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'inactive_frac'].values,
#                     bigdf.loc[(bigdf.vip_cond=='ctrl') & (bigdf.opto==True), 'inactive_frac'].values)
# scipy.stats.ttest_ind(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'active_frac'].values,
#                     bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==False), 'active_frac'].values)
# scipy.stats.ttest_ind(bigdf.loc[(bigdf.vip_cond=='vip') & (bigdf.opto==True), 'active_frac'].values,
#                     bigdf.loc[(bigdf.vip_cond=='ctrl') & (bigdf.opto==True), 'active_frac'].values)


# sig # per animal
scipy.stats.ttest_ind(bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='vip') & (bigdf_test.index.get_level_values('opto')==True), 'inactive_frac'].values,
                    bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='ctrl') & (bigdf_test.index.get_level_values('opto')==True), 'inactive_frac'].values)
# vip led off vs. on
scipy.stats.ttest_ind(bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='vip') & (bigdf_test.index.get_level_values('opto')==True), 'inactive_frac'].values,
                    bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='vip') & (bigdf_test.index.get_level_values('opto')==False), 'inactive_frac'].values)
# active cells
scipy.stats.ttest_ind(bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='vip') & (bigdf_test.index.get_level_values('opto')==True), 'active_frac'].values,
                    bigdf_test.loc[(bigdf_test.index.get_level_values('vip_cond')=='vip') & (bigdf_test.index.get_level_values('opto')==False), 'active_frac'].values)

# %%
# understand inactive cell tuning
dy=8 

bin_size = 3
dct = dcts[dy]
print(conddf.iloc[dy])
arr = dct['learning_tc2'][1][dct['inactive']]
tc2 = arr[np.argsort(dct['coms2'][dct['inactive']])]
arr = dct['learning_tc1'][1][dct['inactive']]
tc1 = arr[np.argsort(dct['coms1'][dct['inactive']])]
plt.imshow(np.concatenate([tc1,tc2]))
plt.axvline(dct['rewlocs_comp'][0]/bin_size, color='w', linestyle='--')
plt.axvline(dct['rewlocs_comp'][1]/bin_size, color='w')
plt.axhline(tc1.shape[0], color='yellow')
plt.title('TC1 (top) vs. TC2 (bottom), last 5 trials')
plt.ylabel('Cells')
plt.xlabel('Spatial bins (3cm)')
#%%
pdf = matplotlib.backends.backend_pdf.PdfPages(r'Z:\inactive_cells_tuning_curves_per_animal.pdf')

bin_size = 3
figcom, axcom = plt.subplots() 
for dy,dct in enumerate(dcts):   
    if conddf.in_type.values[dy]=='vip':
        # TODO plot with circular coms (import here)
    # dct = dcts[dy]
        arr = dct['learning_tc2'][1][dct['active']]
        tc2 = arr[np.argsort(dct['coms2'][dct['active']])]
        arr = dct['learning_tc1'][1][dct['active']]
        tc1 = arr[np.argsort(dct['coms1'][dct['active']])]
        # fig, ax = plt.subplots()
        # plt.imshow(np.concatenate([tc1,tc2]))
        # plt.axvline(dct['rewlocs_comp'][0]/bin_size, color='w', linestyle='--')
        # plt.axvline(dct['rewlocs_comp'][1]/bin_size, color='w')
        # plt.axhline(tc1.shape[0], color='yellow')
        # plt.title(f'TC1 (top) vs. TC2 (bottom), last 5 trials \n Animal {conddf.iloc[dy].animals}, Opto Ep {conddf.iloc[dy].optoep}, Day {conddf.iloc[dy].days}')
        # plt.ylabel('Cells')
        # plt.xlabel('Spatial bins (3cm)')
        # pdf.savefig(fig)
        if conddf.optoep.values[dy]>1:
            axcom.scatter(dct['coms1'][dct['inactive']]-dct['rewlocs_comp'][0], dct['coms2'][dct['inactive']]-dct['rewlocs_comp'][1], s=5, color='red')       
        # if conddf.optoep.values[dy]>1:
        #     axcom.scatter(dct['coms1'][dct['inactive']]-dct['rewlocs_comp'][0], dct['coms2'][dct['inactive']]-dct['rewlocs_comp'][1], s=5, color='red')       
        # if conddf.optoep.values[dy]<2:
        #     axcom.scatter(dct['coms1'][dct['inactive']]-dct['rewlocs_comp'][0], dct['coms2'][dct['inactive']]-dct['rewlocs_comp'][1], s=5, color='k')
        # else:
        #     axcom.scatter(dct['coms1'][dct['inactive']]-dct['rewlocs_comp'][0], dct['coms2'][dct['inactive']]-dct['rewlocs_comp'][1], s=5, color='red')
        # if conddf.optoep.values[dy]<2:
        #     axcom.scatter(dct['coms1'][dct['active']]-dct['rewlocs_comp'][0], dct['coms2'][dct['active']]-dct['rewlocs_comp'][1], s=5, color='k')

        # else:
        #     axcom.scatter(dct['coms1'][dct['active']]-dct['rewlocs_comp'][0], dct['coms2'][dct['active']]-dct['rewlocs_comp'][1], s=5, color='red')
axcom.plot(axcom.get_xlim(), axcom.get_ylim(), color='slategray', linestyle='--')
axcom.spines['top'].set_visible(False)
axcom.spines['right'].set_visible(False)
axcom.set_ylabel('COM - Reward Loc, LED on')
axcom.set_xlabel('COM - Reward Loc, LED off')

    # plt.close('all')

pdf.close()