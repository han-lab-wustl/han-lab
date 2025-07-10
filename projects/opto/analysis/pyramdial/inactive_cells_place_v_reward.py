
"""
zahra's analysis for initial com and enrichment of pyramidal cell data
updated aug 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math,  matplotlib as mpl, matplotlib.backends.backend_pdf
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.analysis.pyramdial.placecell import get_pyr_metrics_opto, get_dff_opto
mpl.rcParams['svg.fonttype'] = 'none'; mpl.rcParams["xtick.major.size"] = 10; mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
#%% - re-run dct making
# uses matlab tuning curves
dcts = []
for dd in range(len(conddf.days.values)):
    # define threshold to detect activation/inactivation
    day = conddf.days.values[dd]
    if dd%10==0: print(f'{dd}/{len(conddf)}')
    if dd!=202:
        threshold = 5
        pc = False
        dct = get_pyr_metrics_opto(conddf, dd, day, 
                    threshold=threshold, pc=pc)
        dcts.append(dct)
    # save pickle of dcts
# saved new version for r21 2/25/25
with open(r'Z:\dcts_com_opto_inference_wcomp.p', "wb") as fp:   #Pickling
    pickle.dump(dcts, fp)   
    
#%%
# get inactivated cells distribution
# fc3 for opto vs. control
# pre reward
# pcs only
dffs = []
for dd,day in enumerate(conddf.days.values):
    dff_opto, dff_prev = get_dff_opto(conddf, dd, day,pc=True)
    if dd%10==0: print(dd)
    dffs.append([dff_opto, dff_prev])
    
#%%
s =12 # pointsize
# plot relative fc3 transients
plt.rc('font', size=20)          # controls default text sizes
df = conddf.copy()[:len(dffs)]
df['dff_target'] = np.array(dffs)[:,0]
df['dff_prev'] = np.array(dffs)[:,1]
df['dff_target-prev'] = df['dff_target']-df['dff_prev']
df['condition'] = ['VIP' if xx=='vip' else 'Control' for xx in df.in_type.values]
df['opto'] = df.optoep.values>1
# only initial days as controls
# df['opto'] = [True if xx>1 else False if xx==-1 else np.nan for xx in conddf.optoep.values]
df = df.loc[~((df.animals=='e217')&(df.days.isin([12,26,29])))] # exclude noisy days
df = df.loc[~((df.animals=='e190')|(df.animals=='e189'))] # exclude noisy days
df=df[df.optoep.values>1]
df=df.groupby(['animals', 'condition']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(2,5))
sns.barplot(x="condition", y="dff_target-prev", hue = 'condition', data=df,
                palette={'Control': "slategray", 'VIP': "red"},
                errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="dff_target-prev", hue = 'condition', data=df,
                palette={'Control': "slategray", 'VIP': "red"},
                s=s,ax=ax,dodge=True)

ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Norm. place cell activity')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xlabel('')
# pvalues needed
t,pval = scipy.stats.ranksums(df.loc[((df.index.get_level_values('condition')=='VIP')),
            'dff_target-prev'].values, \
            df.loc[((df.index.get_level_values('condition')=='Control')),
            'dff_target-prev'].values)

# statistical annotation    
fs=46
ii=1; y=.003; pshift=.0002
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

# plt.savefig(os.path.join(savedst, 'dff.svg'), bbox_inches='tight')
#%%
s=12 # pointsize
# plot fraction of cells near reward
# conddf=conddf.drop([202])
df = conddf.copy()
optoep = conddf.optoep.values; animals = conddf.animals.values; in_type = conddf.in_type.values
dcts = np.array(dcts)
df['frac_pc_prev_early'] = [dct['frac_place_cells_tc1_early_trials'] for dct in dcts]
df['frac_pc_opto_early'] = [dct['frac_place_cells_tc2_early_trials'] for dct in dcts]
df['frac_pc_prev_late'] = [dct['frac_place_cells_tc1_late_trials'] for dct in dcts]
df['frac_pc_opto_late'] = [dct['frac_place_cells_tc2_late_trials'] for dct in dcts]
df['frac_pc_prev'] = df['frac_pc_prev_late']-df['frac_pc_prev_early']
df['frac_pc_opto'] = df['frac_pc_opto_late']-df['frac_pc_opto_early']
df['opto'] = [True if xx>1 else False if xx<1 else np.nan for xx in conddf.optoep.values]

# df['opto'] = conddf.optoep.values>1
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in conddf.in_type.values]
# df = df.loc[(df.animals!='e189')&(df.animals!='e190')]
bigdf=df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)
bigdf = bigdf.reset_index()

fig,ax = plt.subplots(figsize=(4,6))
sns.barplot(data=bigdf,
            x='opto', 
            y="frac_pc_opto", 
            hue='condition',        
                # palette={'ctrl': "slategray", 'vip': "red"},
                errorbar='se',ax=ax,fill=False)
sns.stripplot(x="opto", y="frac_pc_opto", hue = 'condition',
                data=bigdf,
                # palette={'ctrl': "slategray", 'vip': "red"},
                dodge=True,
                s=s,ax=ax)
ax.spines[['top', 'right']].set_visible(False); 
ax.legend(bbox_to_anchor=(1.05, 1.05))

# ctrl v.s vip
t,pval = scipy.stats.ranksums(bigdf[(bigdf.opto==True) & (bigdf.condition=='vip')].frac_pc_opto.values, \
            bigdf[(bigdf.opto==True) & (bigdf.condition=='ctrl')].frac_pc_opto.values)

ax.set_ylabel(f'Fraction of place cells near rew. loc.')
ax.set_xticks([0,1], labels=['LED off', 'LED on'])
ax.set_xlabel('')
# statistical annotation    
fs=46
ii=1; y=.13; pshift=.02
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

# ctrl v.s vip
t,pval = scipy.stats.ttest_rel(bigdf[(bigdf.opto==True) & (bigdf.condition=='vip')].frac_pc_opto.values, \
            bigdf[(bigdf.opto==False) & (bigdf.condition=='vip')].frac_pc_opto.values)

ax.set_ylabel(f'Fraction of place cells near rew. loc.')
ax.set_xticks([0,1], labels=['LED off', 'LED on'])
ax.set_xlabel('')
# statistical annotation    
fs=46; ii=0.5
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)
from statsmodels.stats.power import TTestPower

# Extract the relevant data
vip_on = bigdf[(bigdf.opto == True) & (bigdf.condition == 'vip')].frac_pc_opto.values
vip_off = bigdf[(bigdf.opto == False) & (bigdf.condition == 'vip')].frac_pc_opto.values

# Calculate the mean difference and pooled standard deviation
mean_diff = np.mean(vip_on) - np.mean(vip_off)
pooled_std = np.sqrt((np.std(vip_on, ddof=1) ** 2 + np.std(vip_off, ddof=1) ** 2) / 2)

# Calculate Cohen's d (effect size)
cohen_d = mean_diff / pooled_std

# Initialize power analysis object
analysis = TTestPower()

# Calculate the required sample size for 80% power
sample_size = analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, alternative='two-sided')

print(f"Estimated required sample size per group: {np.ceil(sample_size):.0f}")
# plt.savefig(os.path.join(savedst, 'frac_pc.svg'), bbox_inches='tight')
#%%
# average enrichment
# not as robust effect with 3 mice
df['enrichment_prev'] = [np.nanmean(dct['difftc1']) for dct in dcts]
df['enrichment_opto'] = [np.nanmean(dct['difftc2']) for dct in dcts]

# ax.tick_params(axis='x', labelrotation=90)
bigdf=df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)

plt.figure(figsize=(2.5,6))
bigdf = bigdf.sort_values('condition')
ax = sns.barplot(x="opto", y="enrichment_opto",hue='condition',data=bigdf, fill=False,errorbar='se')
sns.stripplot(x="opto", y="enrichment_opto",hue='condition',data=bigdf,dodge=True,s=10)
vip = bigdf[(bigdf.index.get_level_values('opto')==True) & (bigdf.index.get_level_values('condition')=='vip')].enrichment_opto.values
ctrl = bigdf[(bigdf.index.get_level_values('opto')==True) &  (bigdf.index.get_level_values('condition')=='ctrl')].enrichment_opto.values
t,pval=scipy.stats.ranksums(vip, ctrl)
ax.spines[['top', 'right']].set_visible(False); ax.get_legend().set_visible(False)
plt.title(f"p-value = {pval:03f}")

# plt.savefig(os.path.join(savedst, 'tuning_curve_enrichment.svg'), bbox_inches='tight')
#%%
#  com shift
# control vs. vip led on
# com_shift col 0 = inactive; 1 = active; 2 = all
optoep = conddf.optoep.values
in_type = conddf.in_type.values
optoep_in = np.array([xx for ii,xx in enumerate(optoep)])
com_shift = np.array([np.array(dct['com_shift']) for ii,dct in enumerate(dcts)])
rewloc_shift = np.array([dct['rewloc_shift'] for ii,dct in enumerate(dcts)])
animals = conddf.animals.values
df = pd.DataFrame(com_shift[:,0], columns = ['com_shift_inactive'])
df['com_shift_active'] = com_shift[:,1]
df['rewloc_shift'] = rewloc_shift
df['animal'] = animals
condition = []
df['vipcond'] = [xx if 'vip' in xx else 'ctrl' for xx in in_type]
df = df[(df.animal!='e189')&(df.animal!='e200')]
dfagg = df.groupby(['animal', 'vipcond']).mean(numeric_only=True)

fig, ax = plt.subplots()
ax = sns.scatterplot(x = 'com_shift_inactive', y = 'rewloc_shift', hue = 'vipcond', data = dfagg,s=150)
ax = sns.scatterplot(x = 'com_shift_inactive', y = 'rewloc_shift', hue = 'vipcond', data = df, s=150,alpha=0.2)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
ax.set_title('Shift = VIP Inhibition-Before Inhibition')
# plt.savefig(os.path.join(savedst, 'scatterplot_comshift.svg'), bbox_inches='tight')
# active
fig, ax = plt.subplots()
ax = sns.scatterplot(x = 'com_shift_active', y = 'rewloc_shift', hue = 'vipcond', data = dfagg ,s=50)
ax = sns.scatterplot(x = 'com_shift_active', y = 'rewloc_shift', hue = 'vipcond', data = df, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(bbox_to_anchor=(1.1, 1.1))
ax.set_title('Shift = VIP Inhibition-Before Inhibition')

#%%
# bar plot of shift
dfagg = dfagg.sort_values('vipcond')
dfagg = dfagg.reset_index()
fig, ax = plt.subplots(figsize=(2.5,5))
ax = sns.barplot(x = 'vipcond', y = 'com_shift_inactive', hue = 'vipcond', data=dfagg,
                errorbar='se', fill=False)
ax = sns.stripplot(x = 'vipcond', y = 'com_shift_inactive', hue = 'vipcond', data=dfagg,
                s=s)

ax.spines[['top','right']].set_visible(False)

vipshift = dfagg.loc[dfagg.vipcond=='vip', 'com_shift_inactive'].values
ctrlshift = dfagg.loc[dfagg.vipcond=='ctrl', 'com_shift_inactive'].values
t,pval=scipy.stats.ranksums(vipshift, ctrlshift)
ii=0.5; y= 35
ax.text(ii-0.5, y, f'p={pval:.3g}',fontsize=12)
ax.set_ylabel('Av. Inactive PC rel. center-of-mass')
ax.set_xticks([0,1], labels=['Control', 'VIP \nInhibition'])
ax.set_xlabel('')

from statsmodels.stats.power import TTestIndPower

# Means and standard deviations for each group
mean_vip = dfagg.loc[dfagg['vipcond'] == 'vip', 'com_shift_inactive'].mean()
mean_ctrl = dfagg.loc[dfagg['vipcond'] == 'ctrl', 'com_shift_inactive'].mean()
std_vip = dfagg.loc[dfagg['vipcond'] == 'vip', 'com_shift_inactive'].std()
std_ctrl = dfagg.loc[dfagg['vipcond'] == 'ctrl', 'com_shift_inactive'].std()
# Number of samples in each group
n_vip = dfagg[dfagg['vipcond'] == 'vip'].shape[0]
n_ctrl = dfagg[dfagg['vipcond'] == 'ctrl'].shape[0]

# Pooled standard deviation
pooled_std = np.sqrt(((n_vip - 1) * std_vip**2 + (n_ctrl - 1) * std_ctrl**2) / (n_vip + n_ctrl - 2))
# Calculate Cohen's d
cohen_d = (mean_vip - mean_ctrl) / pooled_std
# Perform power analysis
analysis = TTestIndPower()
power = analysis.power(effect_size=cohen_d, nobs1=n_vip, alpha=0.05, ratio=n_ctrl/n_vip, alternative='two-sided')

print(f"Cohen's d: {cohen_d:.3f}")
print(f"Power: {power:.3f}")
# plt.savefig(os.path.join(savedst, 'barplot_comshift.svg'), bbox_inches='tight')
#%%
# active
dfagg = dfagg.sort_values('vipcond')
fig, ax = plt.subplots(figsize=(2.5,6))
ax = sns.barplot(x = 'vipcond', y = 'com_shift_active', hue = 'vipcond', data=dfagg, fill=False,
                errorbar='se')
ax = sns.stripplot(x = 'vipcond', y = 'com_shift_active', hue = 'vipcond', data=dfagg,
                s=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
vipshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='vip', 'com_shift_active'].values
ctrlshift = dfagg.loc[dfagg.index.get_level_values('vipcond')=='ctrl', 'com_shift_active'].values
t,pval=scipy.stats.ranksums(vipshift, ctrlshift)
plt.title(f"p-value = {pval:03f}")
# plt.savefig(os.path.join(savedst, 'barplot_active_comshift.svg'), bbox_inches='tight')

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
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
    df['day'] = conddf.days.values[ii]
    df['in_type'] = conddf.in_type.values[ii]    
    df['opto'] = bool(conddf.optoep.values[ii]>1) # true vs. false
    df['rewzones_transition'] = f'rz_{dct["rewzones_comp"][0].astype(int)}-{dct["rewzones_comp"][1].astype(int)}'
    if 'vip' in df['in_type'].values[0]:
        df['vip_cond'] = df['in_type'].values[0] 
    else:
        df['vip_cond'] = 'ctrl'

    dfs_diff.append(df)
bigdf_org = pd.concat(dfs_diff,ignore_index=False) 
bigdf_org.reset_index(drop=True, inplace=True)   
bigdf_org = bigdf_org[(bigdf_org['animal']!='e190')&(bigdf_org['animal']!='e186')]

# remove outlier days
bigdf_org=bigdf_org[~((bigdf_org.animal=='z17')&((bigdf_org.day>12)))]
bigdf_org=bigdf_org[~((bigdf_org.animal=='z15')&((bigdf_org.day.isin([10,16]))))]
bigdf_org=bigdf_org[~((bigdf_org.animal=='e217')&((bigdf_org.day<9)|(bigdf_org.day.isin([18,17,11,29,30]))))]
bigdf_org=bigdf_org[~((bigdf_org.animal=='e189')&((bigdf_org.day.isin([35,41,42,44]))))]

# Compute per-animal LED-on minus off differences for inactive and active
bigdf = bigdf_org.groupby(['animal', 'vip_cond','opto']).mean(numeric_only=True).reset_index()
# Pivot to get ON/OFF split
pivoted = bigdf.pivot_table(index=['animal', 'vip_cond'], columns='opto', values=['inactive_frac', 'active_frac'])
pivoted.columns = [f"{feat}_{'on' if opto else 'off'}" for feat, opto in pivoted.columns]

# Compute difference per animal
pivoted['inactive_diff'] = pivoted['inactive_frac_on'] - pivoted['inactive_frac_off']
pivoted['active_diff'] = pivoted['active_frac_on'] - pivoted['active_frac_off']
pivoted = pivoted.reset_index()
# Melt to long format for plotting
df_long = pd.melt(pivoted, id_vars=['animal', 'vip_cond'], 
                  value_vars=['inactive_diff', 'active_diff'], 
                  var_name='cell_type', value_name='LED_on_minus_off')
# Rename for nicer plotting
df_long['cell_type'] = df_long['cell_type'].map({
    'inactive_diff': 'Inactivated',
    'active_diff': 'Activated'
})
df_long['LED_on_minus_off']=df_long['LED_on_minus_off']*100
# --- Plot ---
plt.figure(figsize=(5.5,5))
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax = sns.barplot(data=df_long, x='cell_type', y='LED_on_minus_off', hue='vip_cond', errorbar='se', fill=False,legend=False,palette=pl)
sns.stripplot(data=df_long, x='cell_type', y='LED_on_minus_off', hue='vip_cond',dodge=True, jitter=0.1, s=12, ax=ax,alpha=0.7,palette=pl)
# Clean up duplicate legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

# Format axes
ax.set_ylabel('$\Delta$ % Cells (LEDon-LEDoff)')
ax.set_xlabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# --- Statistical Annotations with Comparison Bars ---
conds = df_long['vip_cond'].unique()
cell_types = ['Inactivated', 'Activated']
y_max = df_long['LED_on_minus_off'].max()-10
y_step = 3
bar_idx = 0  # counter to stack comparison bars

for cell_type in cell_types:
    df_cell = df_long[df_long['cell_type'] == cell_type]

    for iii,cond in enumerate(conds):
        if cond == 'ctrl': continue
        # Get group data
        data1 = df_cell[df_cell['vip_cond'] == 'ctrl']['LED_on_minus_off'].astype(float)
        data2 = df_cell[df_cell['vip_cond'] == cond]['LED_on_minus_off'].astype(float)

        # Compute p-value
        stat, pval = stats.ranksums(data1, data2)
        text = '*' if pval < 0.05 else ''

        # Get x-locations
        x1 = cell_types.index(cell_type)-0.25  # ctrl
        x2 = cell_types.index(cell_type)-0.15+0.2*iii  # test group
        y = y_max + y_step * (bar_idx + 1)

        # Draw bar and annotate
        ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
        ax.text((x1 + x2) / 2, y-.01, text, ha='center', va='bottom', fontsize=40)
        ax.text((x1 + x2) / 2, y + y_step/3 - 0.01, f'p = {pval:.3g}', ha='center', va='bottom', fontsize=10)

        bar_idx += 1

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'active_v_inactive_all_opto.svg'))

#%%
# examples
track_length = 270
# dd=192
dd=52
# define threshold to detect activation/inactivation
animal = conddf.animals.values[dd]
day=conddf.days.values[dd]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
fall = scipy.io.loadmat(params_pth, variable_names=['dFF', 'Fc3','forwardvel', 'ybinned', 'iscell','bordercells', 'changeRewLoc'])
active = dcts[dd]['active']
inactive = dcts[dd]['inactive']
changeRewLoc = np.hstack(fall['changeRewLoc'])
eptest = conddf.optoep.values[dd]
if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
eps = np.where(changeRewLoc>0)[0]
rewlocs = changeRewLoc[eps]*1.5
eps = np.append(eps, len(changeRewLoc))    
if len(eps)<4: eptest = 2 # if no 3 epochs
comp = [eptest-2,eptest-1] # eps to compare  
# filter iscell
Fc3 = fall['dFF'][:,((fall['iscell'][:,0].astype(bool)) & ~(fall['bordercells'][0].astype(bool)))]
fv = fall['forwardvel'][0]
position = fall['ybinned'][0]*1.5
# z15
active=active[np.array([1,3,4,6,7,8,9,10,11,13,15])]
#%%
# plot
cells=inactive[25:35] # e216, 54
color='mediumturquoise'
cells=active
color='lightcoral'
fig, axes=plt.subplots(nrows=len(cells),sharex=True,figsize=(9,6))
for kk,cll in enumerate(cells):
    axes[kk].plot(Fc3[:,cll],color='k')
    patch_start = eps[1]
    patch_end = Fc3.shape[0]  # or patch_start + some_window
    # Draw patch (light red transparent rectangle)
    axes[kk].axvspan(patch_start, patch_end, color=color, alpha=0.3)
    if kk==0:
        axes[kk].set_ylabel('$\Delta F/F$')
    axes[kk].spines[['top','right']].set_visible(False)    
    # axes[kk].set_rasterized(True)  # <-- Rasterize this axis
    axes[kk].tick_params(axis='y', labelsize=12)  # or any size you prefer
axes[kk].set_xlabel('Time (minutes)')
axes[kk].set_xticks([0,Fc3.shape[0]/2,Fc3.shape[0]])
axes[kk].set_xticklabels([0,np.round(((Fc3.shape[0]/2)/31.25)/60,1),np.round(((Fc3.shape[0])/31.25)/60,1)])
# fig.tight_layout()
plt.savefig(os.path.join(savedst, f'{animal}_day{day}_active_cell_traces_opto.svg'))
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