
"""
zahra's analysis for initial com and enrichment of pyramidal cell data
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from sklearn.cluster import KMeans
import seaborn as sns
import placecell
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
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural.csv", index_col=None)

#%%
dcts = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    threshold = 7
    pc = False
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
                errorbar='se', fill=False)
ax = sns.stripplot(x="opto", y="frac_pc", hue = 'vip_ctrl', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

bigdf = bigdf[bigdf.index.get_level_values('in_type') == in_type_cond]
scipy.stats.ttest_rel(bigdf[(bigdf.index.get_level_values('opto')==True)].frac_pc.values, \
            bigdf[(bigdf.index.get_level_values('opto')==False)].frac_pc.values)
scipy.stats.ranksums(bigdf[(bigdf.index.get_level_values('opto')==True)].frac_pc.values, bigdf[(bigdf.index.get_level_values('opto')==False)].frac_pc.values)
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
bigdf_test = bigdf.groupby(['animal', 'vip_cond', 'opto']).mean(numeric_only=True)
comp1 = bigdf_test[(bigdf_test.index.get_level_values('opto')==True) & (bigdf_test.index.get_level_values('vip_cond')=='vip')].tc_diff.values; comp1=comp1[~np.isnan(comp1)]
comp2 = bigdf_test[(bigdf_test.index.get_level_values('opto')==False) &  (bigdf_test.index.get_level_values('vip_cond')=='vip')].tc_diff.values; comp2=comp2[~np.isnan(comp2)]
diff_offon_vip = comp1-comp2
comp1 = bigdf_test[(bigdf_test.index.get_level_values('opto')==True) & (bigdf_test.index.get_level_values('vip_cond')=='ctrl')].tc_diff.values; comp1=comp1[~np.isnan(comp1)]
comp2 = bigdf_test[(bigdf_test.index.get_level_values('opto')==False) &  (bigdf_test.index.get_level_values('vip_cond')=='ctrl')].tc_diff.values; comp2=comp2[~np.isnan(comp2)]
diff_offon_ctrl = comp1-comp2
t,pval=scipy.stats.ranksums(diff_offon_vip, diff_offon_ctrl)

plt.figure()
df = pd.DataFrame(np.concatenate([diff_offon_vip, diff_offon_ctrl]), columns = ['tc_diff_ledoff-on'])
df['condition']=np.concatenate([['vip']*len(diff_offon_vip), ['ctrl']*len(diff_offon_ctrl)])
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
# control vs. vip led on
# com_shift col 0 = inactive; 1 = active; 0 = all
optoep = conddf.optoep.values
in_type = conddf.in_type.values

optoep_in = np.array([xx for ii,xx in enumerate(optoep)])
com_shift = np.array([dct['com_shift'] for ii,dct in enumerate(dcts)])
rewloc_shift = np.array([dct['rewloc_shift'] for ii,dct in enumerate(dcts)])
fig, ax = plt.subplots()
ax.scatter(com_shift[((optoep_in>=2) & (in_type=='vip')), 0], rewloc_shift[((optoep_in>=2) & (in_type=='vip'))], label = 'VIP Inactive', color = 'red')
ax.scatter(com_shift[((optoep_in>=2) & (in_type=='sst')), 0], rewloc_shift[((optoep_in>=2) & (in_type=='sst'))], label = 'Control Inactive', color = 'gold')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[((optoep_in>=2) & (in_type=='vip')), 0][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='vip')), 0])],rewloc_shift[((optoep_in>=2) & (in_type=='vip'))][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='vip')), 0])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color = 'r')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[((optoep_in>=2) & (in_type=='sst')), 0][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='sst')), 0])],rewloc_shift[((optoep_in>=2) & (in_type=='sst'))][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='sst')), 0])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color = 'gold')

ax.scatter(com_shift[((optoep_in>=2) & (in_type=='vip')), 1], rewloc_shift[((optoep_in>=2) & (in_type=='vip'))], label = 'VIP Active', color = 'maroon')
ax.scatter(com_shift[((optoep_in>=2) & (in_type=='sst')), 1], rewloc_shift[((optoep_in>=2) & (in_type=='sst'))], label = 'Control Active', color = 'darkgoldenrod')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[((optoep_in>=2) & (in_type=='vip')), 1][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='vip')), 1])],rewloc_shift[((optoep_in>=2) & (in_type=='vip'))][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='vip')), 1])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color = 'maroon')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(com_shift[((optoep_in>=2) & (in_type=='sst')), 1][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='sst')), 1])],rewloc_shift[((optoep_in>=2) & (in_type=='sst'))][~np.isnan(com_shift[((optoep_in>=2) & (in_type=='sst')), 1])])
x_vals = np.array(ax.get_xlim())
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color = 'darkgoldenrod')
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
    elif (df['in_type'].values[0] =='sst') or (df['in_type'].values[0] =='ctrl'):
        df['vip_cond'] = 'ctrl'

    dfs_diff.append(df)
bigdf_org = pd.concat(dfs_diff,ignore_index=False) 
bigdf_org.reset_index(drop=True, inplace=True)   

# plot fraction of inactivated vs. activated cells
bigdf_test = bigdf_org.groupby(['animal', 'vip_cond', 'opto']).mean(numeric_only=True)
bigdf = bigdf_org.groupby(['animal', 'vip_cond','opto']).mean(numeric_only=True)
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
# save pickle of dcts
with open(r'Z:\dcts_com_opto.p', "wb") as fp:   #Pickling
    pickle.dump(dcts, fp)            
#%%
# get spatial info of inactive cells vs. all cells
track_length = 270

info_inactive = []
info_other = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    animal = conddf.animals.values[dd]
    if conddf.optoep.values[dd]>1 and conddf.in_type.values[dd]=='vip':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'forwardvel', 'ybinned', 'iscell',
                                    'bordercells', 'changeRewLoc'])
        inactive = dcts[dd]['inactive']
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[dd]
        if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))    
        if len(eps)<4: eptest = 2 # if no 3 epochs
        comp = [eptest-2,eptest-1] # eps to compare  
        # filter iscell
        Fc3 = fall['Fc3'][:,(fall['iscell'][:,0].astype(bool)) & (~fall['bordercells'][0].astype(bool))]
        thres = 5; ftol = 10; Fs = 31.25; nBins = 90 # hard coded thresholds
        fv = fall['forwardvel'][0]
        position = fall['ybinned'][0]*1.5
        info = []
        # gets spatial info across entire recording!
        # for i in range(Fc3.shape[1]):
        #     info.append(get_spatial_info_per_cell(Fc3[:,i], fv, thres, ftol, position, 
        #                         Fs, nBins, track_length))
        # only in opto ep
        for i in range(Fc3.shape[1]):
            info.append(get_spatial_info_per_cell(Fc3[eps[comp[0]]:eps[comp[1]+1],i], fv[eps[comp[0]]:eps[comp[0]+1]],
                thres, ftol, position[eps[comp[0]]:eps[comp[0]+1]], 
                Fs, nBins, track_length)-get_spatial_info_per_cell(Fc3[eps[comp[1]]:eps[comp[1]+1],i], fv[eps[comp[1]]:eps[comp[1]+1]],
                thres, ftol, position[eps[comp[1]]:eps[comp[1]+1]], 
                Fs, nBins, track_length))
        info = np.array(info)
        info_inactive.append(info[inactive])
        info_other.append(info[~inactive])
#%%
ina = np.concatenate(info_inactive)
# ina = info_inactive[16]
oth = np.concatenate(info_other)
# oth = info_other[16]
df = pd.DataFrame()
df['spatial_info'] = np.concatenate([ina, oth])
df['condition'] = np.concatenate([['inactive']*len(ina), ['other']*len(oth)])
ax = sns.stripplot(x='condition', y='spatial_info', data=df, color='k')
ax = sns.boxplot(x='condition', y='spatial_info', data=df, fill=False, color='k')

#%%
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(r'Z:\inactive_cell_tuning_per_animal.pdf')

pearsonr_per_day = []
# understand inactive cell tuning
for dd,day in enumerate(conddf.days.values):
    pearsonr_per_cell = []
    dct = dcts[dd]
    animal = conddf.animals.values[dd]
    if conddf.in_type.values[dd]=='vip':#and conddf.in_type.values[dd]=='vip':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
                    'tuning_curves_late_trials', 'coms_early_trials'])
        coms = fall['coms'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[dd]
        if conddf.optoep.values[dd]<2: 
            eptest = random.randint(2,3)    
            if len(eps)<4: eptest = 2 # if no 3 epochs
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))    
        
        comp = [eptest-2,eptest-1] # eps to compare    
        other_eps = [xx for xx in range(len(eps)-1) if xx not in comp]
        for other_ep in other_eps:
            tc_other = tcs_late[other_ep]
            coms_other = coms[other_ep]
            bin_size = 3
            # print(conddf.iloc[dy])
            arr = tc_other[dct['inactive']]
            tc3 = arr[np.argsort(dct['coms1'][dct['inactive']])] # np.hstack(coms_other)
            arr = dct['learning_tc2'][1][dct['inactive']]    
            tc2 = arr[np.argsort(dct['coms1'][dct['inactive']])]
            arr = dct['learning_tc1'][1][dct['inactive']]
            tc1 = arr[np.argsort(dct['coms1'][dct['inactive']])]
            fig, ax1 = plt.subplots()            
            if other_ep>comp[1]:
                ax1.imshow(np.concatenate([tc1,tc2,tc3]),cmap = 'jet')
                ax1.axvline(dct['rewlocs_comp'][0]/bin_size, color='w', linestyle='--')
                ax1.axvline(dct['rewlocs_comp'][1]/bin_size, color='w')
                ax1.axvline(rewlocs[other_ep]/bin_size, color='w', linestyle='dotted')
                ax1.axhline(tc1.shape[0], color='yellow')
                ax1.axhline(tc1.shape[0]+tc2.shape[0], color='yellow')
                ax1.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n previous (top) vs. opto (middle) vs. after opto (bottom), inactive cells, last 5 trials')
                ax1.set_ylabel('Cells')
                ax1.set_xlabel('Spatial bins (3cm)')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)                
                for cl,cell in enumerate(dct['inactive']):         
                    fig, ax = plt.subplots()           
                    ax.plot(tc1[cl,:],color='k',label='previous_ep')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='after_ledon')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    pearsonr_per_cell.append(r)
                    # ax.set_axis_off()  
                    ax.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n r={r}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()
                    pdf.savefig(fig)
                    plt.close(fig)
            else:
                ax1.imshow(np.concatenate([tc3,tc1,tc2]),cmap = 'jet')
                ax1.axvline(dct['rewlocs_comp'][0]/bin_size, color='w', linestyle='--')
                ax1.axvline(dct['rewlocs_comp'][1]/bin_size, color='w')
                ax1.axvline(rewlocs[other_ep]/bin_size, color='w', linestyle='dotted')
                ax1.axhline(tc3.shape[0], color='yellow')
                ax1.axhline(tc3.shape[0]+tc1.shape[0], color='yellow')
                ax1.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n previous (top) x 2 vs. opto (bottom), inactive cells, last 5 trials')
                ax1.set_ylabel('Cells')
                ax1.set_xlabel('Spatial bins (3cm)')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                r=0; c=0
                for cl,cell in enumerate(dct['inactive']):
                    fig, ax = plt.subplots()         
                    ax.plot(tc1[cl,:],color='k',label='previous_ep')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='ep1')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    pearsonr_per_cell.append(r)
                    ax.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n r={r}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()                   
                    pdf.savefig(fig)
                    plt.close(fig)
        pearsonr_per_day.append(pearsonr_per_cell)
pdf.close()
