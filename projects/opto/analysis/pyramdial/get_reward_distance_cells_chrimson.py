"""get reward distance cells between opto and non opto conditions
feb 2025
vip chrimson excitation
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10 
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_chrimson_rewardcells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_vipexcitation.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
cm_window = 20
# iterate through all animals
for ii in range(len(conddf)):
    day = int(conddf.days.values[ii])
    animal = conddf.animals.values[ii]
    if True:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        radian_alignment, goal_cell_iind, goal_cell_prop, goal_cell_null, dist_to_rew,\
                num_epochs, pvals, rates_all, total_cells, epoch_perm = get_rew_cells_opto(params_pth, pdf, \
                radian_alignment_saved, animal, day, ii, conddf, goal_cell_iind, goal_cell_prop, \
                goal_cell_null, dist_to_rew, num_epochs, pvals, rates_all, total_cells, \
                        epoch_perm, radian_alignment)
pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.index.isin(inds))]
df['goal_cell_prop'] = goal_cell_prop
df['goal_cell_prop']=df['goal_cell_prop']*100
df['opto'] = df.optoep.values>1
df['opto'] = ['stim' if xx==True else 'no_stim' for xx in df.opto.values]
df['condition'] = ['vip' if xx=='vip_ex' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = goal_cell_null
df['goal_cell_prop_shuffle']=df['goal_cell_prop_shuffle']*100
df=df[df.p_value<0.2]
# remove 0 goal cell prop
df = df[df.goal_cell_prop>0]
# df=df[df.days>5]
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df, x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df['p_value'].values<0.05)/len(df)
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')

#%%
##########################  per animal paired comparison ##########################
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
df = df[(df.animals!='e200')]
# exclude outliere?
# df = df[(df.days!=7) & (df.days!=13) & (df.days!=16)]
df_plt = df
color = 'darkgoldenrod'
# top 75%?
df_an = df_plt.groupby(['animals','condition','opto']).median(numeric_only=True)
order = ['no_stim', 'stim']
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_plt,hue_order=order,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        s=9, dodge=True,alpha=.5)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_an,hue_order=order,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        s=s, dodge=True)
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,hue_order=order,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        fill=False,ax=ax, errorbar='se')
sns.barplot(x='condition', y='goal_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
# animal lines
df_an = df_an.reset_index()
ans = ['z15', 'z17', 'z14']#df_an.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=[-.2,0.2], y='goal_cell_prop', 
    data=df_an[df_an.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['VIP\nExcitation', 'Control'],rotation=45)
ax.set_ylabel('Reward cell %')
df_plt = df_plt.reset_index()
rewprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='stim')), 'goal_cell_prop']
shufprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='no_stim')), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# per animal stats
rewprop = df_an.loc[((df_an.condition=='vip')&(df_an.opto=='stim')), 'goal_cell_prop']
shufprop = df_an.loc[((df_an.condition=='vip')&(df_an.opto=='no_stim')), 'goal_cell_prop']
t,pval = scipy.stats.ttest_rel(rewprop, shufprop)

# statistical annotation    
fs=46
ii=0
y=50
pshift=10
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'nonopto vs. opto chrimson\np={pval:.3g}',fontsize=12,rotation=45)

ii=1
# control vs. chrimson
rewprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='stim')), 'goal_cell_prop']
shufprop = df_plt.loc[((df_plt.condition=='ctrl')&(df_plt.opto=='stim')), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# statistical annotation    
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'ctrl vs. chrimson\np={pval:.3g}',fontsize=12,rotation=45)
ax.set_title('n=animals')
plt.savefig(os.path.join(savedst, 'reward_cell_prop_ctrlvopto_chrimson.svg'),bbox_inches='tight')
#%%
# subtract by led off sessions
# ----------------------------------------
# Plotting Stim - No Stim per Animal
# ----------------------------------------
fig2, ax2 = plt.subplots(figsize=(2, 5))
df_diff = (
    df_an[df_an.opto == 'stim']
    .set_index(['animals', 'condition'])[['goal_cell_prop']]
    .rename(columns={'goal_cell_prop': 'stim'})
)
df_diff['no_stim'] = df_an[df_an.opto == 'no_stim'].set_index(['animals', 'condition'])['goal_cell_prop']
df_diff['delta'] = df_diff['stim'] - df_diff['no_stim']
df_diff = df_diff.reset_index()

# Plot
sns.stripplot(data=df_diff, x='condition', y='delta', ax=ax2, 
             palette={'ctrl': "slategray", 'vip': color}, size=s, dodge=True)
sns.barplot(data=df_diff, x='condition', y='delta', ax=ax2, 
             palette={'ctrl': "slategray", 'vip': color}, fill=False)

# Aesthetics
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel('Δ Reward cell % (LEDon-LEDoff)')
ax2.set_xticklabels(['Control', 'VIP'], rotation=45)
ax2.set_title('Per-animal difference\n\n')
ax2.spines[['top', 'right']].set_visible(False)
# control vs. chrimson
rewprop = df_diff.loc[((df_diff.condition=='vip')), 'delta']
shufprop = df_diff.loc[((df_diff.condition=='ctrl')), 'delta']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# statistical annotation    
y = 12
ii=0
if pval < 0.001:
        ax2.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax2.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax2.text(ii, y, "*", ha='center', fontsize=fs)
ax2.text(ii, y, f'{pval:.2g}', ha='center', fontsize=12)

plt.savefig(os.path.join(savedst, 'reward_cell_prop_difference_chrimson.svg'), bbox_inches='tight')
#%%