
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
# cm_window = [10,20,30,40,50,
# 60,70,80] # cm
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
# bigdf = bigdf[bigdf['75_quantile']>0]
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
# scatterplot of lick distance vs. field width
rzdf = bigdf[(bigdf.epoch.str.contains('epoch1_rz1') | bigdf.epoch.str.contains('epoch2_rz3'))]
rzdf2 = bigdf[(bigdf.epoch.str.contains('epoch2_rz1') | bigdf.epoch.str.contains('epoch3_rz3'))]
rzdf3 = bigdf[(bigdf.epoch.str.contains('epoch3_rz1') | bigdf.epoch.str.contains('epoch4_rz3'))]
rzdf4 = bigdf[(bigdf.epoch.str.contains('epoch4_rz1') | bigdf.epoch.str.contains('epoch5_rz3'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])
rzdf = rzdf.groupby(['animal','day','epoch','cell_type']).median(numeric_only=True)
rzdf=rzdf.reset_index()
# only pre
rzdf=rzdf[rzdf.cell_type=='Pre']
rzdf=rzdf.drop(columns=['cell_type'])
rzdf['day']=rzdf['day'].astype(int)
rzdf=rzdf.reset_index()
# rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]
lrzdf = lickbigdf[(lickbigdf.epoch.str.contains('epoch1_rz1') | lickbigdf.epoch.str.contains('epoch2_rz3'))]
rzdf2 = lickbigdf[(lickbigdf.epoch.str.contains('epoch2_rz1') | lickbigdf.epoch.str.contains('epoch3_rz3'))]
rzdf3 = lickbigdf[(lickbigdf.epoch.str.contains('epoch3_rz1') | lickbigdf.epoch.str.contains('epoch4_rz3'))]
rzdf4 = lickbigdf[(lickbigdf.epoch.str.contains('epoch4_rz1') | lickbigdf.epoch.str.contains('epoch5_rz3'))]
lrzdf = pd.concat([lrzdf,rzdf2,rzdf3,rzdf4])
lrzdf = lrzdf.groupby(['animal','day','epoch']).median(numeric_only=True)
lrzdf=lrzdf.reset_index()
lrzdf['day']=lrzdf['day'].astype(int)
alldf = pd.merge(lrzdf, rzdf, on=['animal', 'day', 'epoch'], how='inner')
alldf['lick_dist'] = np.array(alldf['last_lick_loc_cm']-alldf['first_lick_loc_cm']).astype(float)
alldf['width_cm'] = alldf['width_cm'].astype(float)
alldf = alldf.dropna(subset=['width_cm'])
a=0.5
sns.regplot(x='lick_dist', y='width_cm', data=alldf, scatter=True, line_kws={"color": "dodgerblue"},color='k')
r, p = scipy.stats.pearsonr(alldf['lick_dist'], alldf['width_cm'])
plt.text(0.05, 0.95, f'r = {r:.2g}\np = {p:.3g}', transform=plt.gca().transAxes,
         fontsize=17, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

plt.xlabel('Lick Distance (cm)')
plt.ylabel('Field Width (cm)')
plt.title('Correlation Between Lick Distance and Field Width')
plt.tight_layout()

#%%
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def get_transition_df(lickbigdf, rz_pairs):
    rzdf = pd.concat([
        lickbigdf[lickbigdf['epoch'].str.contains(pair[0]) | lickbigdf['epoch'].str.contains(pair[1])]
        for pair in rz_pairs
    ])
    rzdf = rzdf.reset_index(drop=True)
    rzdf['epoch'] = [e[-3:] for e in rzdf['epoch']]
    return rzdf

def plot_transition_panel(rzdf, axes, label_order, color='mediumvioletred'):
    # Mean per animal x epoch
    anrzdf = rzdf.groupby(['animal', 'epoch']).mean(numeric_only=True).reset_index()
    anrzdf = anrzdf.sort_values(by="epoch", ascending=(label_order == ['Near', 'Far']))

    metrics = [
        ('lick_cm_diff', '$\Delta$ Distance (cm) (first-last lick)'),
        ('lick_time_diff', '$\Delta$ Time (s) (first-last lick)'),
        ('lick_rate_hz', 'Lick rate (Hz)'),
        ('avg_velocity_cm_s', 'Velocity (cm/s)'),
    ]

    pvals = []

    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        sns.stripplot(x='epoch', y=metric, data=anrzdf, alpha=0.7, dodge=True, s=10, ax=ax, color=color)
        sns.boxplot(x='epoch', y=metric, data=anrzdf, fill=False, ax=ax, color=color, showfliers=False)
        ax.spines[['top','right']].set_visible(False)

        for animal in anrzdf['animal'].unique():
            df_ = anrzdf[anrzdf.animal == animal].sort_values('epoch', ascending=(label_order == ['Near', 'Far']))
            x_vals = np.arange(len(df_)) + (i * 0.1) - 0.1
            sns.lineplot(x=x_vals, y=df_[metric].values, ax=ax, color='dimgrey', linewidth=2, alpha=0.3)

        ax.set_ylabel(ylabel)
        ax.set_xticklabels(label_order)
        ax.set_xlabel('')

        # Stats
        df1 = anrzdf[anrzdf['epoch'] == 'rz1'].sort_values('animal')
        df2 = anrzdf[anrzdf['epoch'] == 'rz3'].sort_values('animal')
        assert all(df1['animal'].values == df2['animal'].values)
        p = stats.wilcoxon(df1[metric], df2[metric]).pvalue
        pvals.append(p)

    # Bonferroni correction
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    def get_asterisks(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    for i, ax in enumerate(axes):
        ymax = anrzdf[metrics[i][0]].max()
        y_text = ymax + 0.01 * abs(ymax)
        y_p = ymax - 0.005 * abs(ymax)
        ax.annotate(get_asterisks(pvals_corr[i]), xy=(.5, y_text), ha='center', fontsize=46, fontweight='bold')
        ax.annotate(f'{pvals_corr[i]:.2g}', xy=(.5, y_p), ha='center', fontsize=11)

    return anrzdf

# === USAGE ===

# Preprocessing
lickbigdf = pd.concat(lick_dfs)
lickbigdf['lick_cm_diff'] = lickbigdf['last_lick_loc_cm'] - lickbigdf['first_lick_loc_cm']
lickbigdf['lick_time_diff'] = lickbigdf['last_lick_time'] - lickbigdf['first_lick_time']

# Plot setup
fig, axes_all = plt.subplots(nrows=2, ncols=4, figsize=(11, 10))
plt.tight_layout()

# Plot 1→3
rz_pairs_13 = [('epoch1_rz1', 'epoch2_rz3'), ('epoch2_rz1', 'epoch3_rz3'),
               ('epoch3_rz1', 'epoch4_rz3'), ('epoch4_rz1', 'epoch5_rz3')]
rzdf13 = get_transition_df(lickbigdf, rz_pairs_13)
plot_transition_panel(rzdf13, axes_all[0], label_order=['Near', 'Far'])

# Plot 3→1
rz_pairs_31 = [('epoch1_rz3', 'epoch2_rz1'), ('epoch2_rz3', 'epoch3_rz1'),
               ('epoch3_rz3', 'epoch4_rz1'), ('epoch4_rz3', 'epoch5_rz1')]
rzdf31 = get_transition_df(lickbigdf, rz_pairs_31)
plot_transition_panel(rzdf31, axes_all[1], label_order=['Far', 'Near'],color='mediumslateblue')
fig.suptitle('Last 8 correct trials')
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
