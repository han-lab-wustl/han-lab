
"""
zahra
field width transitions
may 2025
dark time updates
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from width import get_pre_post_field_widths
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'width.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
goal_window_cm=20
dfs = []; lick_dfs = [] # licks and velocity
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        df, lick_df = get_pre_post_field_widths(params_pth,animal,day,ii,goal_window_cm=goal_window_cm)
        dfs.append(df)
        lick_dfs.append(lick_df)

#%%
# get all cells width cm 
bigdf = pd.concat(dfs)
# exclude some animals 
# bigdf = bigdf[(bigdf.animal!='e139') & (bigdf.animal!='e145') & (bigdf.animal!='e200')]

# transition from 1 to 3 only
plt.rc('font', size=20)
rzdf = bigdf[(bigdf.epoch.str.contains('epoch1_rz1') | bigdf.epoch.str.contains('epoch2_rz3'))]
rzdf2 = bigdf[(bigdf.epoch.str.contains('epoch2_rz1') | bigdf.epoch.str.contains('epoch3_rz3'))]
rzdf3 = bigdf[(bigdf.epoch.str.contains('epoch3_rz1') | bigdf.epoch.str.contains('epoch4_rz3'))]
rzdf4 = bigdf[(bigdf.epoch.str.contains('epoch4_rz1') | bigdf.epoch.str.contains('epoch5_rz3'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])

rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]
# fig, ax = plt.subplots(figsize=(4,5))
# sns.stripplot(x='epoch',y='width_cm',data=rzdf,hue='cell_type',alpha=0.05,dodge=True)
# sns.boxplot(x='epoch',y='width_cm',data=rzdf,fill=False,hue='cell_type')
# ax.spines[['top','right']].set_visible(False)

#%%
# per animal
s=10
hue_order = ['Pre','Post']
fig, ax = plt.subplots(figsize=(4,5))
# only get super close rz1 rewlocs
rzdf['rewloc_cm'] = [float(xx[-5:]) for xx in rzdf.rewloc.values]
rzdf = rzdf[((rzdf.epoch=='rz3')&(rzdf.rewloc_cm.values>210))|((rzdf.epoch=='rz1')&(rzdf.rewloc_cm.values<120))]
anrzdf = rzdf.groupby(['animal', 'epoch', 'cell_type']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="cell_type", ascending=False)
anrzdf = anrzdf.sort_values(by="epoch", ascending=True)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190')]
sns.stripplot(x='epoch',y='width_cm',data=anrzdf,hue='cell_type',alpha=0.7,dodge=True,s=s,
    palette='Dark2',hue_order=hue_order)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='width_cm',hue='cell_type',data=anrzdf,
        fill=False,palette='Dark2',hue_order=hue_order,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
ind = ['Pre','Post']
for j,ct in enumerate(ind):
    for i in range(len(ans)):    
        df_ = anrzdf[(anrzdf.animal==ans[i]) & (anrzdf.cell_type==ct)]
        df_ = df_.sort_values(by="epoch", ascending=True)
        color=sns.color_palette('Dark2')[j]
        ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='width_cm',
        data=df_,
        errorbar=None, color=color, linewidth=2, alpha=0.3,ax=ax)

# 3) remove whatever legend was just created
ax.legend_.remove()
# 4) re-add only the stripplot legend, placing it outside
ax.legend(
    h_strip, l_strip,
    title='Cell Type',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.
)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Field width (cm)')
ax.set_xlabel('')
ax.set_xticklabels(['Near', 'Far'],rotation=45)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
palette=sns.color_palette('Dark2')
# anrzdf has columns: ['animal','epoch','cell_type','width_cm']
# make sure epoch & cell_type are categorical
anrzdf['epoch'] = anrzdf['epoch'].astype('category')
anrzdf['cell_type'] = anrzdf['cell_type'].astype('category')

aov = AnovaRM(
    data=anrzdf,
    depvar='width_cm',
    subject='animal',
    within=['epoch','cell_type']
).fit()
print(aov)

results = []
for ct in ['Pre','Post']:
    sub = anrzdf[anrzdf['cell_type']==ct]
    near = sub[sub['epoch']=='rz1']['width_cm']
    far  = sub[sub['epoch']=='rz3']['width_cm']
    t, p = scipy.stats.ttest_rel(near, far)
    results.append({'cell_type':ct, 't_stat':t, 'p_uncorrected':p})

posthoc = pd.DataFrame(results)
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected']*len(posthoc), 1.0)
print(posthoc)
# mapping epoch→x-position (as you offset earlier)
x_map = {'Pre':0, 'Post':1}
offset = {'Pre':-0.1, 'Post':+0.1}

for _, row in posthoc.iterrows():
    ct = row['cell_type']
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    x = x_map[ct]+offset[ct]  # test is Pre vs Post, annotate at Pre position
    # y-height: just above max for that group
    ymax = anrzdf[(anrzdf.cell_type==ct)]['width_cm'].max()
    ax.text(x, ymax, stars, ha='center', va='bottom',fontsize=42)
    ax.text(x, ymax+5, f'{ct} near v far\np={p:.2g}', ha='center', va='bottom',fontsize=12,
            rotation=45)
plt.savefig(os.path.join(os.path.join(savedst, 'near_to_far_dark_time_field_width.svg')))

#%%
# transition from 3 to 1 only
plt.rc('font', size=20)
rzdf = bigdf[(bigdf.epoch.str.contains('epoch1_rz3') | bigdf.epoch.str.contains('epoch2_rz1'))]
rzdf2 = bigdf[(bigdf.epoch.str.contains('epoch2_rz3') | bigdf.epoch.str.contains('epoch3_rz1'))]
rzdf3 = bigdf[(bigdf.epoch.str.contains('epoch3_rz3') | bigdf.epoch.str.contains('epoch4_rz1'))]
rzdf4 = bigdf[(bigdf.epoch.str.contains('epoch4_rz3') | bigdf.epoch.str.contains('epoch5_rz1'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])
rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]
# only get super close rz1 rewlocs
rzdf['rewloc_cm'] = [float(xx[-5:]) for xx in rzdf.rewloc.values]
rzdf = rzdf[((rzdf.epoch=='rz3')&(rzdf.rewloc_cm.values>210))|((rzdf.epoch=='rz1')&(rzdf.rewloc_cm.values<120))]
fig, ax = plt.subplots(figsize=(4,5))
sns.stripplot(x='epoch',y='width_cm',data=rzdf,hue='cell_type',alpha=0.05,dodge=True)
sns.boxplot(x='epoch',y='width_cm',data=rzdf,fill=False,hue='cell_type')
ax.spines[['top','right']].set_visible(False)
# per animal
s=10
hue_order = ['Pre','Post']
fig, ax = plt.subplots(figsize=(4,5))
anrzdf = rzdf.groupby(['animal', 'epoch', 'cell_type']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="cell_type", ascending=False)
anrzdf = anrzdf.sort_values(by="epoch", ascending=True)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190') & (anrzdf.animal!='e139')
#              & (anrzdf.animal!='e216')]
# anrzdf=anrzdf[(anrzdf.animal!='z16')]
order = ['rz3', 'rz1']
sns.stripplot(x='epoch',y='width_cm',data=anrzdf,hue='cell_type',alpha=0.7,dodge=True,s=s,
    palette='Dark2',hue_order=hue_order,order=order)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='width_cm',hue='cell_type',data=anrzdf,
        fill=False,palette='Dark2',hue_order=hue_order,order=order,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
ind = ['Pre','Post']
for j,ct in enumerate(ind):
    for i in range(len(ans)):    
        df_ = anrzdf[(anrzdf.animal==ans[i]) & (anrzdf.cell_type==ct)]
        df_ = df_.sort_values(by="epoch", ascending=False)
        color=sns.color_palette('Dark2')[j]
        ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='width_cm',
        data=df_,
        errorbar=None, color=color, linewidth=2, alpha=0.3,ax=ax)

# 3) remove whatever legend was just created
ax.legend_.remove()
# 4) re-add only the stripplot legend, placing it outside
ax.legend(
    h_strip, l_strip,
    title='Cell Type',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.
)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Field width (cm)')
ax.set_xlabel('')
ax.set_xticklabels(['Far', 'Near'],rotation=45)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
palette=sns.color_palette('Dark2')
# anrzdf has columns: ['animal','epoch','cell_type','width_cm']
# make sure epoch & cell_type are categorical
anrzdf['epoch'] = anrzdf['epoch'].astype('category')
anrzdf['cell_type'] = anrzdf['cell_type'].astype('category')
aov = AnovaRM(
    data=anrzdf,
    depvar='width_cm',
    subject='animal',
    within=['epoch','cell_type']
).fit()
print(aov)

results = []
for ct in ['Pre','Post']:
    sub = anrzdf[anrzdf['cell_type']==ct]
    near = sub[sub['epoch']=='rz3']['width_cm']
    far  = sub[sub['epoch']=='rz1']['width_cm']
    t, p = scipy.stats.ttest_rel(near[~np.isnan(near)], far[~np.isnan(far)])
    results.append({'cell_type':ct, 't_stat':t, 'p_uncorrected':p})

posthoc = pd.DataFrame(results)
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected']*len(posthoc), 1.0)
print(posthoc)
# mapping epoch→x-position (as you offset earlier)
x_map = {'Pre':0, 'Post':1}
offset = {'Pre':-0.1, 'Post':+0.1}

for _, row in posthoc.iterrows():
    ct = row['cell_type']
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    x = x_map[ct]+offset[ct]  # test is Pre vs Post, annotate at Pre position
    # y-height: just above max for that group
    ymax = anrzdf[(anrzdf.cell_type==ct)]['width_cm'].max()
    ax.text(x, ymax+1, stars, ha='center', va='bottom',fontsize=42)
    ax.text(x, ymax+4, f'{ct} far v near\np={p:.2g}', ha='center', va='bottom',fontsize=12,
            rotation=45)
plt.savefig(os.path.join(os.path.join(savedst, 'far_to_near_dark_time_field_width.svg')))

#%%
# get corresponding licking behavior 
lickbigdf = pd.concat(lick_dfs)
lickbigdf['lick_cm_diff'] = lickbigdf['last_lick_loc_cm']-lickbigdf['first_lick_loc_cm']
lickbigdf['lick_time_diff'] = lickbigdf['last_lick_time']-lickbigdf['first_lick_time']

# 1 to 3 transition
rzdf = lickbigdf[(lickbigdf.epoch.str.contains('epoch1_rz1') | lickbigdf.epoch.str.contains('epoch2_rz3'))]
rzdf2 = lickbigdf[(lickbigdf.epoch.str.contains('epoch2_rz1') | lickbigdf.epoch.str.contains('epoch3_rz3'))]
rzdf3 = lickbigdf[(lickbigdf.epoch.str.contains('epoch3_rz1') | lickbigdf.epoch.str.contains('epoch4_rz3'))]
rzdf4 = lickbigdf[(lickbigdf.epoch.str.contains('epoch4_rz1') | lickbigdf.epoch.str.contains('epoch5_rz3'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])

rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]

# per animal
s=10
fig, axes_all = plt.subplots(nrows=2,ncols = 4, figsize=(11,10))
axes=axes_all[0]
color='mediumvioletred'
# only get super close rz1 rewlocs
anrzdf = rzdf.groupby(['animal', 'epoch']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="epoch", ascending=True)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190')]
ax=axes[0]
sns.stripplot(x='epoch',y='lick_cm_diff',data=anrzdf,alpha=0.7,dodge=True,s=s,ax=ax,color=color)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_cm_diff',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=True)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_cm_diff',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('$\Delta$ Distance (cm) (first-last lick)')  
ax.set_xticklabels(['Near','Far'])
ax.set_xlabel('')

ax=axes[1]
sns.stripplot(x='epoch',y='lick_time_diff',data=anrzdf,alpha=0.7,color=color,dodge=True,ax=ax,s=s)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_time_diff',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=True)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_time_diff',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('$\Delta$ Time (s) (first-last lick)')
ax.set_xticklabels(['Near','Far'])
ax.set_xlabel('')

ax=axes[2]
sns.stripplot(x='epoch',y='lick_rate_hz',data=anrzdf,color=color,alpha=0.7,dodge=True,ax=ax,s=s)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_rate_hz',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=True)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_rate_hz',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('Lick rate (Hz)')  
ax.set_xticklabels(['Near','Far'])
ax.set_xlabel('')

ax=axes[3]
sns.stripplot(x='epoch',y='avg_velocity_cm_s',data=anrzdf,
    ax=ax,color=color,alpha=0.7,dodge=True,s=s)
sns.boxplot(x='epoch',y='avg_velocity_cm_s',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)
ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=True)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='avg_velocity_cm_s',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('Velocity (cm/s)')  
ax.set_xticklabels(['Near','Far'])
ax.set_xlabel('')

from statsmodels.stats.multitest import multipletests
# Filter for common animals only
df_epoch1 = anrzdf[(anrzdf.epoch == 'rz1')].sort_values('animal')
df_epoch2 = anrzdf[(anrzdf.epoch == 'rz3')].sort_values('animal')
# Sanity check: matched animals
assert all(df_epoch1.animal.values == df_epoch2.animal.values)
# Paired t-tests
lick_cm_diff_p = scipy.stats.wilcoxon(df_epoch1['lick_cm_diff'], df_epoch2['lick_cm_diff']).pvalue
lick_rate_hz_p = scipy.stats.wilcoxon(df_epoch1['lick_rate_hz'], df_epoch2['lick_rate_hz']).pvalue
lick_time_diff_p = scipy.stats.wilcoxon(df_epoch1['lick_time_diff'], df_epoch2['lick_time_diff']).pvalue
avg_velocity_cm_s_p = scipy.stats.wilcoxon(df_epoch1['avg_velocity_cm_s'], df_epoch2['avg_velocity_cm_s']).pvalue
# Bonferroni correction
pvals = [lick_cm_diff_p, lick_time_diff_p,lick_rate_hz_p, avg_velocity_cm_s_p]
reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
# Define position for annotations per epoch
epochs = sorted(anrzdf['epoch'].unique())
x_positions = np.arange(len(epochs))  # x positions for each epoch (0, 1, 2)
plt.tight_layout()
# Function to convert p-value to asterisk significance
def get_asterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Add annotations for each subplot
metrics = ['lick_cm_diff', 'lick_time_diff','lick_rate_hz', 'avg_velocity_cm_s']
for i, metric in enumerate(metrics):
    ax = axes[i]
    # Get max y value for the current metric to place annotation above the data
    ymax = anrzdf[metric].max()
    y_text = ymax + 0.01 * abs(ymax)  # Offset a bit above max value
    y_p  = ymax - 0.005 * abs(ymax)
    pval = pvals_corr[i]
    star = get_asterisks(pval)
    ax.annotate(f'{star}',
                xy=(.5, y_text),  # position over epoch 1 (middle box)
                xycoords='data',
                ha='center', va='bottom',
                fontsize=46, fontweight='bold')
    ax.annotate(f'{pval:.2g}',
                xy=(.5, y_p),  # position over epoch 1 (middle box)
                xycoords='data',
                ha='center', va='bottom',
                fontsize=11)
    
# 3 to 1 transition
rzdf = lickbigdf[(lickbigdf.epoch.str.contains('epoch1_rz3') | lickbigdf.epoch.str.contains('epoch2_rz1'))]
rzdf2 = lickbigdf[(lickbigdf.epoch.str.contains('epoch2_rz3') | lickbigdf.epoch.str.contains('epoch3_rz1'))]
rzdf3 = lickbigdf[(lickbigdf.epoch.str.contains('epoch3_rz3') | lickbigdf.epoch.str.contains('epoch4_rz1'))]
rzdf4 = lickbigdf[(lickbigdf.epoch.str.contains('epoch4_rz3') | lickbigdf.epoch.str.contains('epoch5_rz1'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])

rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]

# per animal
s=10
axes = axes_all[1]
color='mediumvioletred'
# only get super close rz1 rewlocs
anrzdf = rzdf.groupby(['animal', 'epoch']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="epoch", ascending=False)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190')]
ax=axes[0]
sns.stripplot(x='epoch',y='lick_cm_diff',data=anrzdf,alpha=0.7,dodge=True,s=s,ax=ax,color=color)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_cm_diff',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=False)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_cm_diff',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('$\Delta$ Distance (cm) (first-last lick)')  
ax.set_xticklabels(['Far','Near'])

ax=axes[1]
sns.stripplot(x='epoch',y='lick_time_diff',data=anrzdf,alpha=0.7,color=color,dodge=True,ax=ax,s=s)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_time_diff',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=False)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_time_diff',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('$\Delta$ Time (s) (first-last lick)')
ax.set_xticklabels(['Far','Near'])

ax=axes[2]
sns.stripplot(x='epoch',y='lick_rate_hz',data=anrzdf,color=color,alpha=0.7,dodge=True,ax=ax,s=s)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='lick_rate_hz',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)
ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=False)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='lick_rate_hz',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('Lick rate (Hz)')  
ax.set_xticklabels(['Far','Near'])

ax=axes[3]
sns.stripplot(x='epoch',y='avg_velocity_cm_s',data=anrzdf,
    ax=ax,color=color,alpha=0.7,dodge=True,s=s)
sns.boxplot(x='epoch',y='avg_velocity_cm_s',data=anrzdf,
        fill=False,ax=ax,color=color,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)
ans = anrzdf.animal.unique()
for i in range(len(ans)):    
    df_ = anrzdf[(anrzdf.animal==ans[i])]
    df_ = df_.sort_values(by="epoch", ascending=False)
    ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='avg_velocity_cm_s',
    data=df_,
    errorbar=None, color='dimgrey', linewidth=2, alpha=0.3,ax=ax)
ax.set_ylabel('Velocity (cm/s)')  
ax.set_xticklabels(['Far','Near'])
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
# Filter for common animals only
df_epoch1 = anrzdf[(anrzdf.epoch == 'rz1')].sort_values('animal')
df_epoch2 = anrzdf[(anrzdf.epoch == 'rz3')].sort_values('animal')
# Sanity check: matched animals
assert all(df_epoch1.animal.values == df_epoch2.animal.values)
# Paired t-tests
lick_cm_diff_p = scipy.stats.wilcoxon(df_epoch1['lick_cm_diff'], df_epoch2['lick_cm_diff']).pvalue
lick_rate_hz_p = scipy.stats.wilcoxon(df_epoch1['lick_rate_hz'], df_epoch2['lick_rate_hz']).pvalue
lick_time_diff_p = scipy.stats.wilcoxon(df_epoch1['lick_time_diff'], df_epoch2['lick_time_diff']).pvalue
avg_velocity_cm_s_p = scipy.stats.wilcoxon(df_epoch1['avg_velocity_cm_s'], df_epoch2['avg_velocity_cm_s']).pvalue
# Bonferroni correction
pvals = [lick_cm_diff_p, lick_time_diff_p,lick_rate_hz_p, avg_velocity_cm_s_p]
reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
# Define position for annotations per epoch
epochs = sorted(anrzdf['epoch'].unique())
x_positions = np.arange(len(epochs))  # x positions for each epoch (0, 1, 2)
plt.tight_layout()
# Function to convert p-value to asterisk significance
def get_asterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Add annotations for each subplot
metrics = ['lick_cm_diff', 'lick_time_diff','lick_rate_hz', 'avg_velocity_cm_s']
for i, metric in enumerate(metrics):
    ax = axes[i]
    # Get max y value for the current metric to place annotation above the data
    ymax = anrzdf[metric].max()
    y_text = ymax + 0.01 * abs(ymax)  # Offset a bit above max value
    y_p  = ymax - 0.005 * abs(ymax)
    pval = pvals_corr[i]
    star = get_asterisks(pval)
    ax.annotate(f'{star}',
                xy=(.5, y_text),  # position over epoch 1 (middle box)
                xycoords='data',
                ha='center', va='bottom',
                fontsize=46, fontweight='bold')
    ax.annotate(f'{pval:.2g}',
                xy=(.5, y_p),  # position over epoch 1 (middle box)
                xycoords='data',
                ha='center', va='bottom',
                fontsize=11)
plt.savefig(os.path.join(os.path.join(savedst, 'transition_licks_velocity.svg')))

#%%
ii=152 # near to far
ii=132 # far to near
# behavior
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
        'stat', 'licks'])
VR = fall['VR'][0][0][()]
scalingf = VR['scalingFACTOR'][0][0]
try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
except:
        rewsize = 10
ypos = fall['ybinned'][0]/scalingf
track_length=180/scalingf    
forwardvel = fall['forwardvel'][0]    
changeRewLoc = np.hstack(fall['changeRewLoc'])
trialnum=fall['trialnum'][0]
rewards = fall['rewards'][0]
lick=fall['licks'][0]
lick[ypos<3]=0
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
rz = get_rewzones(rewlocs,1/scalingf)       

eps = np.where(changeRewLoc)[0]
rew = (rewards==1).astype(int)
mask = np.array([True if xx>10 and xx<28 else False for xx in trialnum])
mask = np.zeros_like(trialnum).astype(bool)
# mask[eps[0]+3000:eps[1]+12000]=True # near to far
mask[eps[0]+7000:eps[1]+18000]=True # far to near
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(20,5))
ax.plot(ypos[mask],zorder=1)
ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k',
        zorder=2)
ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan',
    zorder=2)
# ax.add_patch(
# patches.Rectangle(
#     xy=(0,newrewloc-10),  # point of origin.
#     width=len(ypos[mask]), height=20, linewidth=1, # width is s
#     color='slategray', alpha=0.3))
ax.add_patch(
patches.Rectangle(
    xy=(0,(changeRewLoc[eps][0]/scalingf)-10),  # point of origin.
    width=len(ypos[mask]), height=20, linewidth=1, # width is s
    color='slategray', alpha=0.3))

ax.set_ylim([0,270])
ax.spines[['top','right']].set_visible(False)
ax.set_title(f'{day}')
# plt.savefig(os.path.join(savedst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')
