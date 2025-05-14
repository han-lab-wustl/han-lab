"""get reward distance cells between opto and non opto conditions
oct 2024
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
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper\vip_r21'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
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
#%%
plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
# inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
# df = df[((df.animals!='e217')) & (df.index.isin(inds))]
df['goal_cell_prop'] = goal_cell_prop
df['goal_cell_prop']=df['goal_cell_prop']*100
df['opto'] = df.optoep.values>1
df['opto'] = ['stim' if xx==True else 'no_stim' for xx in df.opto.values]
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = goal_cell_null
df['goal_cell_prop_shuffle']=df['goal_cell_prop_shuffle']*100
# df=df[df.p_value<0.05]
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
fig,ax = plt.subplots(figsize=(4,5))
# av across mice
df = df[(df.animals!='e200')]
# only get -1 pre opto values
df=df[(df.optoep==-1) | (df.optoep>1)]
df_plt = df
color = 'red'
# top 75%?
df_an = df_plt.groupby(['animals','condition','opto']).quantile(0.25,numeric_only=True)
order = ['no_stim', 'stim']
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_plt,hue_order=order,
        palette={'no_stim': "slategray", 'stim': color},
        s=9, dodge=True,alpha=.5)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_an,hue_order=order,
        palette={'no_stim': "slategray", 'stim': color},
        s=s, dodge=True)
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,hue_order=order,
        palette={'no_stim': "slategray", 'stim': color},
        fill=False,ax=ax, errorbar='se')
sns.barplot(x='condition', y='goal_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
# animal lines
df_an = df_an.reset_index()
ans = ['e216', 'e218', 'e217']#df_an.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=[-.2,0.2], y='goal_cell_prop', 
    data=df_an[df_an.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['VIP\nInhibition', 'Control'],rotation=45)
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
plt.savefig(os.path.join(savedst, 'reward_cell_prop_ctrlvopto_inhib.svg'),bbox_inches='tight')
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
df_diff = df_diff[(df_diff.animals!='e190')]
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
t,pval = scipy.stats.ttest_ind(rewprop, shufprop)
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

plt.savefig(os.path.join(savedst, 'reward_cell_prop_difference_inhib.svg'), bbox_inches='tight')

#%%
############################## OLD ##############################
############################## OLD ##############################
############################## OLD ##############################


plt.rc('font', size=18)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[df.optoep>1]
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = goal_cell_null
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df, x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df['p_value'].values<0.05)/len(df)
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(2,5))
# av across mice
df = df[(df.animals!='e200')&(df.animals!='e189')]
df_plt = df
df_plt = df_plt.groupby(['animals','condition']).mean(numeric_only=True)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='condition',data=df_plt,
        palette={'ctrl': "slategray", 'vip': "red"},
        s=s)
sns.barplot(x='condition', y='goal_cell_prop',
        data=df_plt,
        palette={'ctrl': "slategray", 'vip': "red"},
        fill=False,ax=ax, color='k', errorbar='se')
sns.barplot(x='condition', y='goal_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'],rotation=45)
ax.set_ylabel('Reward-centric cell proportion\n(LEDoff vs. LEDon)')
rewprop = df_plt.loc[(df_plt.index.get_level_values('condition')=='vip'), 'goal_cell_prop']
shufprop = df_plt.loc[(df_plt.index.get_level_values('condition')=='ctrl'), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# statistical annotation    
fs=46
ii=0.5; y=.37; pshift=.05
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

plt.savefig(os.path.join(savedst, 'reward_cell_prop_ctrlvopto.svg'),bbox_inches='tight')

#%%    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perms['condition'] = ['vip' if (xx=='e218') or (xx=='e216') else 'ctrl' for xx in df_perms['animals'].values]
df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = df_permsav.index.get_level_values("epoch_comparison").unique()
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
    shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'condition','num_epochs']).mean(numeric_only=True)
#%%
# quantify reward cells vip vs. opto
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals','condition','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(5,5))
# av across mice
cmap = ['slategray','crimson']
sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='condition',
        data=df_plt2,palette=cmap,
        s=12, dodge=True, alpha=.8)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='condition',
        data=df_plt2,palette=cmap,
        fill=False,ax=ax, errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward cell proportion')
eps = [2,3,4]
y = 0.16
pshift = 0.02
fs=36
for con in ['vip', 'ctrl']:
    for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[((df_plt2.index.get_level_values('num_epochs')==ep) & \
            (df_plt2.index.get_level_values('condition')==con)), 'goal_cell_prop']
        shufprop = df_plt2.loc[((df_plt2.index.get_level_values('num_epochs')==ep)\
            & (df_plt2.index.get_level_values('condition')==con)), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{con}, {ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'{con},p={pval:.3g}',fontsize=10)
    y+=.02

# plt.savefig(os.path.join(savedst, 'reward_cell_prop_per_an.png'), 
#         bbox_inches='tight')
#%%

df['recorded_neurons_per_session'] = total_cells
df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df_plt_,
        s=150, ax=ax)
sns.regplot(x='recorded_neurons_per_session', y='goal_cell_prop',
        data=df_plt_,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
        df_plt_['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Av. # of neurons per session')
ax.set_ylabel('Reward cell proportion')

plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
        bbox_inches='tight')

#%%
df['success_rate'] = rates_all

an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    sns.scatterplot(x='success_rate', y='goal_cell_prop',
            data=df[(df.animals==an)&(df.opto==False)&(df.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()

# #%%
# # #examples
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greateer than 2
bin_size=3 # cm
# get abs dist tuning 
tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)

# # #plot example tuning curve
plt.rc('font', size=30)  
fig,axes = plt.subplots(1,3,figsize=(20,20), sharex = True)
for ep in range(3):
        axes[ep].imshow(tcs_correct_abs[ep,com_goal[0]][np.argsort(coms_correct_abs[0,com_goal[0]])[:60],:]**.3)
        axes[ep].set_title(f'Epoch {ep+1}')
        axes[ep].axvline((rewlocs[ep]-rewsize/2)/bin_size, color='w', linestyle='--', linewidth=4)
        axes[ep].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
        axes[ep].set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
axes[0].set_ylabel('Reward-distance cells')
axes[2].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'abs_dist_tuning_curves_3_ep.svg'), bbox_inches='tight')

# fig,axes = plt.subplots(1,4,figsize=(15,20), sharey=True, sharex = True)
# axes[0].imshow(tcs_correct[0,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[0].set_title('Epoch 1')
# im = axes[1].imshow(tcs_correct[1,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[1].set_title('Epoch 2')
# im = axes[2].imshow(tcs_correct[2,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[2].set_title('Epoch 3')
# im = axes[3].imshow(tcs_correct[3,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[3].set_title('Epoch 4')
# ax = axes[1]
# ax.set_xticks(np.arange(0,bins+1,10))
# ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
# ax.axvline((bins/2), color='w', linestyle='--')
# axes[0].axvline((bins/2), color='w', linestyle='--')
# axes[2].axvline((bins/2), color='w', linestyle='--')
# axes[3].axvline((bins/2), color='w', linestyle='--')
# axes[0].set_ylabel('Reward distance cells')
# axes[3].set_xlabel('Reward-relative distance (rad)')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'tuning_curves_4_ep.png'), bbox_inches='tight')
# for gc in goal_cells:
#%%
gc = 51
plt.rc('font', size=24)  
fig2,ax2 = plt.subplots(figsize=(5,5))

for ep in range(3):        
        ax2.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
ax2.set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
ax2.set_xlabel('Absolute position (cm)')
ax2.set_ylabel('$\Delta$ F/F')
        
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_tuning_per_ep.svg'), bbox_inches='tight')

fig2,ax2 = plt.subplots(figsize=(5,5))
for ep in range(3):        
        ax2.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(bins/2, color="k", linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,bins+1,30))
ax2.set_xticklabels(np.round(np.arange(-np.pi, 
        np.pi+np.pi/1.5, np.pi/1.5),1))
ax2.set_xlabel('Radian position ($\Theta$)')
ax2.set_ylabel('$\Delta$ F/F')
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_aligned_tuning_per_ep.svg'), bbox_inches='tight')