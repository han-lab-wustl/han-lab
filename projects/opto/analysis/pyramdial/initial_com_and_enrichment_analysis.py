"""
zahra's analysis for clustering/dimensionality reduction of pyramidal cell data
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto
import matplotlib.backends.backend_pdf
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\conddf_neural.csv", index_col=None)

#%%
dcts = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    threshold = 10
    pc = True
    dct = get_pyr_metrics_opto(conddf, dd, day, threshold=threshold, pc=pc)
    dcts.append(dct)
#%%
# plot fraction of cells near reward
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
    if in_type[optoep>1][ii]=="vip":
        df['vip_ctrl']='vip'
    else:
        df['vip_ctrl']='ctrl'
    # else: 
    #     df['opto'] = [False]*len(df)
    dfs.append(df)
bigdf = pd.concat(dfs)    

bigdf=bigdf.groupby(['animal', 'vip_ctrl','opto', 'in_type']).median(numeric_only=True)

in_type_cond = 'vip'
fig,ax = plt.subplots()
ax = sns.barplot(x="opto", y="frac_pc", hue = 'vip_ctrl', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"},
                ci=68, fill=False)
ax = sns.stripplot(x="opto", y="frac_pc", hue = 'vip_ctrl', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

bigdf = bigdf[bigdf.index.get_level_values('in_type') == in_type_cond]
scipy.stats.ttest_rel(bigdf[(bigdf.index.get_level_values('opto')==True)].frac_pc.values, \
            bigdf[(bigdf.index.get_level_values('opto')==False)].frac_pc.values)
scipy.stats.ranksums(bigdf[(bigdf.opto==True)].frac_pc.values, bigdf[(bigdf.opto==False)].frac_pc.values)
# %%
# average enrichment
# not as robust effect with 3 mice
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
    if df['in_type'].values[0] =='vip':
        df['vip_cond'] = 'vip'
    elif df['in_type'].values[0] !='pv':        
        df['vip_cond'] = 'ctrl'
    dfs_diff.append(df)
bigdf = pd.concat(dfs_diff,ignore_index=False) 
bigdf.reset_index(drop=True, inplace=True)   
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

plt.figure()
df = pd.DataFrame(np.concatenate([diff_offon_vip, diff_offon_ctrl]), columns = ['tc_diff_ledoff-on'])
df['condition']=np.concatenate(np.array([['vip']*len(diff_offon_vip), ['ctrl']*len(diff_offon_ctrl)]))
ax = sns.barplot(x="condition", y="tc_diff_ledoff-on",data=df, fill=False, color='k')
sns.stripplot(x="condition", y="tc_diff_ledoff-on",data=df, color='k')
plt.title(f"p-value = {pval:03f}")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
scipy.stats.t.interval(alpha=0.95, df=len(df[df.condition=='vip'])-1, 
            loc=df[df.condition=='vip'].mean().values[0], 
            scale=scipy.stats.sem(df.loc[df.condition=='vip', 'tc_diff_ledoff-on'].values)) 
scipy.stats.t.interval(alpha=0.95, df=len(df[df.condition=='ctrl'])-1, 
            loc=df[df.condition=='ctrl'].mean().values[0], 
            scale=scipy.stats.sem(df.loc[df.condition=='ctrl', 'tc_diff_ledoff-on'].values)) 

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
    df['rewzones_transition'] = f'rz_{dct["rewzones_comp"][0].astype(int)}-{dct["rewzones_comp"][1].astype(int)}'
    if df['in_type'].values[0] =='vip':
        df['vip_cond'] = 'vip'
    # else:
    #     df['vip_cond'] = 'ctrl'
    elif (df['in_type'].values[0] =='sst') or df['animal'].values[0] =='e190':
        df['vip_cond'] = 'ctrl'

    dfs_diff.append(df)
bigdf_org = pd.concat(dfs_diff,ignore_index=False) 
bigdf_org.reset_index(drop=True, inplace=True)   

# plot fraction of inactivated vs. activated cells
bigdf_test = bigdf_org.groupby(['animal', 'vip_cond', 'opto']).mean()
bigdf = bigdf_org.groupby(['animal', 'vip_cond','opto']).mean()
fig, ax = plt.subplots()
ratio = (bigdf.loc[bigdf.index.get_level_values('opto')==True, 'inactive_frac'].values)-(bigdf.loc[bigdf.index.get_level_values('opto')==False, 'inactive_frac'].values)
conditions = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('vip_cond'))
animals = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('animal'))
df = pd.DataFrame(np.array([ratio, conditions, animals]).T, columns=['ledoff-on', 'condition', 'animal'])
ax = sns.barplot(x="condition", y="ledoff-on", hue='condition',data=df,fill=False,
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="condition", y="ledoff-on", hue='condition', data=df,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

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
scipy.stats.ranksums(df.loc[(df.condition=='vip'), 'ledoff-on'].astype(float).values, df.loc[(df.condition=='ctrl'), 'ledoff-on'].astype(float).values)


# %%
# understand inactive cell tuning
dy=3

bin_size = 3
dct = dcts[dy]
print(conddf.iloc[dy])
arr = dct['learning_tc2'][1][dct['inactive']]
tc2 = arr[np.argsort(dct['coms1'][dct['inactive']])]
arr = dct['learning_tc1'][1][dct['inactive']]
tc1 = arr[np.argsort(dct['coms1'][dct['inactive']])]
plt.imshow(np.concatenate([tc1,tc2]),cmap = 'jet')
plt.axvline(dct['rewlocs_comp'][0]/bin_size, color='w', linestyle='--')
plt.axvline(dct['rewlocs_comp'][1]/bin_size, color='w')
plt.axhline(tc1.shape[0], color='yellow')
plt.title('TC1 (top) vs. TC2 (bottom), last 5 trials')
plt.ylabel('Cells')
plt.xlabel('Spatial bins (3cm)')
plt.savefig(r'C:\Users\Han\Box\neuro_phd_stuff\classes_2024\presentations\tcs.svg', bbox_inches='tight')
#%%
# plot coms and collect tracked cells
tracked_cells_pth = r"Y:\analysis\celltrack\e218_daily_tracking\Results\commoncells_once_per_week.mat"
tracked_cells = scipy.io.loadmat(tracked_cells_pth)
tracked_cells = tracked_cells['commoncells_once_per_week'].astype(int)-1
pdf = matplotlib.backends.backend_pdf.PdfPages(r'Z:\inactive_cells_tuning_curves_per_animal.pdf')
tracked_days = np.arange(20,51)
bin_size = 3
figcom, axcom = plt.subplots() 
tracked_cell_inactive_inds = []
tcs = []; coms = []; s2p_ind = []
for dy,dct in enumerate(dcts):   
    animal = conddf.animals.values[dy]
    day = conddf.days.values[dy]
    if conddf.in_type.values[dy]=='vip': #animal == 'e218':
        # TODO plot with circular coms (import here)
    # dct = dcts[dy]
        # dyind = np.where(day==tracked_days)[0][0]
        # tracked_cells_day = tracked_cells[:,dyind]
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['pyr_tc_s2p_cellind'])
        pyr_tc_s2p_cellind = fall['pyr_tc_s2p_cellind'][0]
        assert len(dct['learning_tc1'][1])==(len(pyr_tc_s2p_cellind))
        s2p_ind.append(pyr_tc_s2p_cellind)
        # s2p indices of inactive cells
        real_cell_inactive_ind = pyr_tc_s2p_cellind[dct['inactive']]
        # check if these cells are tracked
        tracked_cells_inactive = [xx for xx in real_cell_inactive_ind if xx in tracked_cells_day]
        
        tracked_cell_inactive_ind = [np.where(xx==tracked_cells_day)[0][0] for xx in tracked_cells_inactive]
        tracked_cell_inactive_inds.append(tracked_cell_inactive_ind)
        tcs.append([dct['learning_tc1'][1],dct['learning_tc2'][1]])
        coms.append([dct['coms1'],dct['coms2']])
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
tracked_cell_inactive_inds_opto = np.array(tracked_cell_inactive_inds)[conddf.loc[conddf.animals=='e218', 'optoep']>1]
tcs_opto = np.array(tcs)[conddf.loc[conddf.animals=='e218', 'optoep']>1]
s2p_ind_opto = np.array(s2p_ind)[conddf.loc[conddf.animals=='e218', 'optoep']>1]
same_in_first_inactivation = [[xx for xx in tracked_cell_inactive_inds_opto[0] if xx in yy] for yy in tracked_cell_inactive_inds_opto]

# # find its ind in the tuning curve
# tc_ind = np.where(same_in_first_inactivation[1][0]==s2p_ind_opto[1])[0][0]
# plt.plot(tcs_opto[1][0][tc_ind])
# plt.plot(tcs_opto[1][1][tc_ind])

# tc_ind = np.where(same_in_first_inactivation[2][0]==s2p_ind_opto[2])[0][0]
# plt.plot(tcs_opto[2][0][tc_ind])
# plt.plot(tcs_opto[2][1][tc_ind])

# dy=1; el=1
# tc_ind = np.where(same_in_first_inactivation[dy][el]==s2p_ind_opto[dy])[0][0]
# plt.plot(tcs_opto[dy][0][tc_ind])
# plt.plot(tcs_opto[dy][1][tc_ind])
# dy=5
# tc_ind = np.where(same_in_first_inactivation[dy][el]==s2p_ind_opto[dy])[0][0]
# plt.plot(tcs_opto[dy][0][tc_ind])
# plt.plot(tcs_opto[dy][1][tc_ind])