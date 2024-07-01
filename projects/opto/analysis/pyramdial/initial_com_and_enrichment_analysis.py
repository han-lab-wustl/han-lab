
"""
zahra's analysis for initial com and enrichment of pyramidal cell data
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math,  matplotlib as mpl, matplotlib.backends.backend_pdf
from sklearn.cluster import KMeans
from placecell import get_pyr_metrics_opto, get_dff_opto
mpl.rcParams['svg.fonttype'] = 'none'; mpl.rcParams["xtick.major.size"] = 8; mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'
#%% - re-run dct making
dcts = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    threshold = 7
    pc = False
    dct = get_pyr_metrics_opto(conddf, dd, day, 
                threshold=threshold, pc=pc)
    dcts.append(dct)
# save pickle of dcts
# with open(r'Z:\dcts_modeling_wcomp.p', "wb") as fp:   #Pickling
#     pickle.dump(dcts, fp)   
#%%
# open previously saved dcts
# with open(r"Z:\dcts_com_opto_inference_wcomp.p", "rb") as fp: #unpickle
#         dcts = pickle.load(fp)

# #%%
# dff for opto vs. control
dffs = []
for dd,day in enumerate(conddf.days.values):
    dff_opto, dff_prev = get_dff_opto(conddf, dd, day)
    dffs.append([dff_opto, dff_prev])
    
# #%%
# plot
plt.rc('font', size=20)          # controls default text sizes
conddf['dff_target'] = np.array(dffs)[:,0]
conddf['dff_prev'] = np.array(dffs)[:,1]
conddf['dff_target-prev'] = conddf['dff_target']-conddf['dff_prev']
conddf['condition'] = ['VIP' if xx=='vip' else 'Control' for xx in conddf.in_type.values]
conddf['opto'] = conddf.optoep.values>1
df = conddf
df=df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize=(2.5,6))
ax = sns.barplot(x="opto", y="dff_target-prev", hue = 'condition', data=df,
                palette={'Control': "slategray", 'VIP': "red"},
                errorbar='se', fill=False)
ax = sns.stripplot(x="opto", y="dff_target-prev", hue = 'condition', data=df,
                palette={'Control': "slategray", 'VIP': "red"},
                s=10)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)

t,pval = scipy.stats.ranksums(df[(df.index.get_level_values('condition')=='VIP')]['dff_target-prev'].values, \
            df[(df.index.get_level_values('condition')=='Control')]['dff_target-prev'].values)
# plt.savefig(os.path.join(savedst, 'dff.jpg'), bbox_inches='tight')
#%%
# plot fraction of cells near reward
df = conddf
optoep = conddf.optoep.values; animals = conddf.animals.values; in_type = conddf.in_type.values
dcts = np.array(dcts)
df['frac_pc_prev_early'] = [dct['frac_place_cells_tc1_early_trials'] for dct in dcts]
df['frac_pc_opto_early'] = [dct['frac_place_cells_tc2_early_trials'] for dct in dcts]
df['frac_pc_prev_late'] = [dct['frac_place_cells_tc1_late_trials'] for dct in dcts]
df['frac_pc_opto_late'] = [dct['frac_place_cells_tc2_late_trials'] for dct in dcts]
df['frac_pc_prev'] = df['frac_pc_prev_late']-df['frac_pc_prev_early']
df['frac_pc_opto'] = df['frac_pc_opto_late']-df['frac_pc_opto_early']
df['opto'] = [True if xx>1 else False if xx==-1 else np.nan for xx in conddf.optoep.values]
df['opto'] = conddf.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in conddf.in_type.values]

bigdf=df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(4,6))
ax = sns.barplot(x="opto", y="frac_pc_prev", hue = 'condition', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se', fill=False)
ax = sns.stripplot(x="opto", y="frac_pc_prev", hue = 'condition', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top', 'right']].set_visible(False)
ax.get_legend().set_visible(False)
t,pval = scipy.stats.ranksums(bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='vip')].frac_pc_prev.values, \
            bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='ctrl')].frac_pc_prev.values)
ax.set_title(f'p = {pval:02f}')

fig,ax = plt.subplots(figsize=(4,6))
ax = sns.barplot(x="opto", y="frac_pc_opto", hue = 'condition', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se', fill=False)
ax = sns.stripplot(x="opto", y="frac_pc_opto", hue = 'condition', data=bigdf,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)
ax.spines[['top', 'right']].set_visible(False); ax.get_legend().set_visible(False)
t,pval = scipy.stats.ranksums(bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='vip')].frac_pc_opto.values, \
            bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='ctrl')].frac_pc_opto.values)
ax.set_title(f'p = {pval:02f}')
# # plt.savefig(os.path.join(savedst, 'frac_pc.svg'), bbox_inches='tight')

# average enrichment
# not as robust effect with 3 mice
df['enrichment_prev'] = [np.nanmean(dct['difftc1']) for dct in dcts]
df['enrichment_opto'] = [np.nanmean(dct['difftc2']) for dct in dcts]

# ax.tick_params(axis='x', labelrotation=90)
bigdf=df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)

plt.figure(figsize=(2.5,6))
bigdf = bigdf.sort_values('condition')
ax = sns.barplot(x="opto", y="enrichment_opto",hue='condition',data=bigdf, fill=False,
                palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se')
sns.stripplot(x="opto", y="enrichment_opto",hue='condition',data=bigdf,
            palette={'ctrl': "slategray", 'vip': "red"},s=10)
vip = bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='vip')].enrichment_opto.values
ctrl = bigdf[(bigdf.index.get_level_values('opto')==True) &  (bigdf.index.get_level_values('condition')=='ctrl')].enrichment_opto.values
t,pval=scipy.stats.ranksums(vip, ctrl)
ax.spines[['top', 'right']].set_visible(False); ax.get_legend().set_visible(False)
plt.title(f"p-value = {pval:03f}")

# plt.savefig(os.path.join(savedst, 'tuning_curve_enrichment.svg'), bbox_inches='tight')
#%%
# com shift
# control vs. vip led on
# com_shift col 0 = inactive; 1 = active; 2 = all
optoep = conddf.optoep.values
in_type = conddf.in_type.values
optoep_in = np.array([xx for ii,xx in enumerate(optoep)])
com_shift = np.array([dct['com_shift'] for ii,dct in enumerate(dcts)])
rewloc_shift = np.array([dct['rewloc_shift'] for ii,dct in enumerate(dcts)])
animals = conddf.animals.values
df = pd.DataFrame(com_shift[:,0], columns = ['com_shift_inactive'])
df['com_shift_active'] = com_shift[:,1]
df['rewloc_shift'] = rewloc_shift
df['animal'] = animals
condition = []
df['vipcond'] = ['vip' if (xx == 'e216') | (xx == 'e217') | (xx == 'e218') else 'ctrl' for xx in animals]
df = df[(df.animal!='e189')&(df.animal!='e200')]
dfagg = df.groupby(['animal', 'vipcond']).mean(numeric_only=True)

fig, ax = plt.subplots()
ax = sns.scatterplot(x = 'com_shift_inactive', y = 'rewloc_shift', hue = 'vipcond', data = dfagg, 
        palette={'ctrl': "slategray", 'vip': "red"},s=150)
ax = sns.scatterplot(x = 'com_shift_inactive', y = 'rewloc_shift', hue = 'vipcond', data = df, 
        palette={'ctrl': "slategray", 'vip': "red"}, s=150,alpha=0.2)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
ax.set_title('Shift = VIP Inhibition-Before Inhibition')
# plt.savefig(os.path.join(savedst, 'scatterplot_comshift.svg'), bbox_inches='tight')
# active
fig, ax = plt.subplots()
ax = sns.scatterplot(x = 'com_shift_active', y = 'rewloc_shift', hue = 'vipcond', data = dfagg, 
        palette={'ctrl': "slategray", 'vip': "red"},s=50)
ax = sns.scatterplot(x = 'com_shift_active', y = 'rewloc_shift', hue = 'vipcond', data = df, 
        palette={'ctrl': "slategray", 'vip': "red"}, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(bbox_to_anchor=(1.1, 1.1))
ax.set_title('Shift = VIP Inhibition-Before Inhibition')

#%%
# bar plot of shift
dfagg = dfagg.sort_values('vipcond')
fig, ax = plt.subplots(figsize=(2.5,6))
ax = sns.barplot(x = 'vipcond', y = 'com_shift_inactive', hue = 'vipcond', data=dfagg, fill=False,
                palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se')
ax = sns.stripplot(x = 'vipcond', y = 'com_shift_inactive', hue = 'vipcond', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
vipshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='vip', 'com_shift_inactive'].values
ctrlshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='ctrl', 'com_shift_inactive'].values
t,pval=scipy.stats.ranksums(vipshift, ctrlshift)
plt.title(f"p-value = {pval:03f}")
# plt.savefig(os.path.join(savedst, 'barplot_comshift.svg'), bbox_inches='tight')

# active
dfagg = dfagg.sort_values('vipcond')
fig, ax = plt.subplots(figsize=(2.5,6))
ax = sns.barplot(x = 'vipcond', y = 'com_shift_active', hue = 'vipcond', data=dfagg, fill=False,
                palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se')
ax = sns.stripplot(x = 'vipcond', y = 'com_shift_active', hue = 'vipcond', data=dfagg,
                palette={'ctrl': "slategray", 'vip': "red"},
                s=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
vipshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='vip', 'com_shift_active'].values
ctrlshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='ctrl', 'com_shift_active'].values
t,pval=scipy.stats.ranksums(vipshift, ctrlshift)
plt.title(f"p-value = {pval:03f}")
# plt.savefig(os.path.join(savedst, 'barplot_active_comshift.svg'), bbox_inches='tight')

#%%
# proportion of inactivate cells 
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
    elif (df['in_type'].values[0] =='sst') or (df['animal'].values[0] =='e190') or (df['animal'].values[0] =='z8'):        
        df['vip_cond'] = 'ctrl'

    dfs_diff.append(df)
bigdf_org = pd.concat(dfs_diff,ignore_index=False) 
bigdf_org.reset_index(drop=True, inplace=True)   

# plot fraction of inactivated vs. activated cells
bigdf_test = bigdf_org.groupby(['animal', 'vip_cond', 'opto']).mean(numeric_only=True)
bigdf = bigdf_org.groupby(['animal', 'vip_cond','opto']).mean(numeric_only=True)

fig, ax = plt.subplots(figsize=(2.5,6))
ratio = (bigdf.loc[bigdf.index.get_level_values('opto')==True, 'inactive_frac'].values)-(bigdf.loc[bigdf.index.get_level_values('opto')==False, 'inactive_frac'].values)
conditions = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('vip_cond'))
animals = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('animal'))
df = pd.DataFrame(np.array([ratio, conditions, animals]).T, columns=['inactivated_cells_proportion_LEDon-off', 'condition', 'animal'])
ax = sns.barplot(x="condition", y="inactivated_cells_proportion_LEDon-off", 
                hue='condition',data=df,fill=False,
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="condition", y="inactivated_cells_proportion_LEDon-off", 
                hue='condition', s=10,data=df,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# sig # per animal
stat,pval = scipy.stats.ranksums(df.loc[(df.condition=='vip'), 'inactivated_cells_proportion_LEDon-off'].astype(float).values, 
            df.loc[(df.condition=='ctrl'), 'inactivated_cells_proportion_LEDon-off'].astype(float).values)
ax.set_title(f'p={np.round(pval, 4)}')
# plt.savefig(os.path.join(savedst, 'inactive_prop.svg'), bbox_inches='tight')
#%%
# activated cells
fig, ax = plt.subplots(figsize=(3,6))
ratio = (bigdf.loc[bigdf.index.get_level_values('opto')==True, 'active_frac'].values)-(bigdf.loc[bigdf.index.get_level_values('opto')==False, 'active_frac'].values)
conditions = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('vip_cond'))
animals = (bigdf[bigdf.index.get_level_values('opto')==True].index.get_level_values('animal'))
df = pd.DataFrame(np.array([ratio, conditions, animals]).T, columns=['activated_cells_proportion_LEDon-off', 'condition', 'animal'])
ax = sns.barplot(x="condition", y="activated_cells_proportion_LEDon-off", hue='condition',data=df,fill=False,
                palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="condition", y="activated_cells_proportion_LEDon-off", hue='condition', s=7,data=df,
                palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

stat,pval = scipy.stats.ranksums(df.loc[(df.condition=='vip'), 'activated_cells_proportion_LEDon-off'].astype(float).values, 
            df.loc[(df.condition=='ctrl'), 'activated_cells_proportion_LEDon-off'].astype(float).values)
ax.set_title(f'p={np.round(pval, 4)}')

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
    if True:#conddf.in_type.values[dd]=='vip':#and conddf.in_type.values[dd]=='vip':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
                    'tuning_curves_late_trials', 'coms_early_trials'])
        coms = fall['coms'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[dd]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))   
        if conddf.optoep.values[dd]<2: 
            eptest = random.randint(2,3)    
            if len(eps)<4: eptest = 2 # if no 3 epochs

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
                # fig, ax1 = plt.subplots() 
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
                    fig, ax = plt.subplots(figsize=(5,4))           
                    ax.plot(tc1[cl,:],color='k',label='previous_epoch')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='after_ledon')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    r = np.round(r,2)
                    pearsonr_per_cell.append(r)
                    # ax.set_axis_off()  
                    ax.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n r={r}, cell: {cell,cl}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()
                    ax.set_ylabel('dF/F')
                    ax.set_xlabel('Spatial bins')
                    plt.savefig(os.path.join(savedst, f'cell{cl}.svg'), bbox_inches='tight')
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
                    fig, ax = plt.subplots(figsize=(5,4))         
                    ax.plot(tc1[cl,:],color='k',label='previous_ep')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='ep1')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    r = np.round(r,2)
                    pearsonr_per_cell.append(r)
                    ax.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n r={r}, cell: {cell,cl}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()
                    plt.savefig(os.path.join(savedst, f'cell{cl}.svg'), bbox_inches='tight')                   
                    pdf.savefig(fig)
                    plt.close(fig)
        pearsonr_per_day.append(pearsonr_per_cell)
pdf.close()
#%%
# active cells
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(r'Z:\active_cell_tuning_per_animal.pdf')

pearsonr_per_day = []
# understand inactive cell tuning
for dd,day in enumerate(conddf.days.values):
    pearsonr_per_cell = []
    dct = dcts[dd]
    animal = conddf.animals.values[dd]
    if True:#conddf.in_type.values[dd]=='vip':#and conddf.in_type.values[dd]=='vip':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
                    'tuning_curves_late_trials', 'coms_early_trials'])
        coms = fall['coms'][0]
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[dd]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))   
        if conddf.optoep.values[dd]<2: 
            eptest = random.randint(2,3)    
            if len(eps)<4: eptest = 2 # if no 3 epochs

        comp = [eptest-2,eptest-1] # eps to compare    
        other_eps = [xx for xx in range(len(eps)-1) if xx not in comp]
        for other_ep in other_eps:
            tc_other = tcs_late[other_ep]
            coms_other = coms[other_ep]
            bin_size = 3
            # print(conddf.iloc[dy])
            arr = tc_other[dct['active']]
            tc3 = arr[np.argsort(dct['coms1'][dct['active']])] # np.hstack(coms_other)
            arr = dct['learning_tc2'][1][dct['active']]    
            tc2 = arr[np.argsort(dct['coms1'][dct['active']])]
            arr = dct['learning_tc1'][1][dct['active']]
            tc1 = arr[np.argsort(dct['coms1'][dct['active']])]
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
                for cl,cell in enumerate(dct['active']):         
                    fig, ax = plt.subplots()           
                    ax.plot(tc1[cl,:],color='k',label='previous_ep')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='after_ledon')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    r = np.round(r,2)
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
                for cl,cell in enumerate(dct['active']):
                    fig, ax = plt.subplots()         
                    ax.plot(tc1[cl,:],color='k',label='previous_ep')
                    ax.plot(tc2[cl,:],color='red',label='led_on')
                    ax.plot(tc3[cl,:],color='slategray',label='ep1')
                    ax.axvline(dct['rewlocs_comp'][0]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(dct['rewlocs_comp'][1]/bin_size,color='red', linestyle='dotted')
                    ax.axvline(rewlocs[other_ep]/bin_size,color='slategray', linestyle='dotted')
                    r, pval = scipy.stats.pearsonr(tc1[cl,:][~np.isnan(tc1[cl,:])], tc2[cl,:][~np.isnan(tc2[cl,:])])
                    r = np.round(r,2)
                    pearsonr_per_cell.append(r)
                    ax.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[dd]}\n r={r}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()                   
                    pdf.savefig(fig)
                    plt.close(fig)
        pearsonr_per_day.append(pearsonr_per_cell)
pdf.close()