
"""
zahra
july 2024
quantify reward-relative cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_by_trialtype, intersect_arrays
from rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'true_pc.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\tuning_curves_pcs_nopto.p"
# with open(saveddataset, "rb") as fp: #unpickle
#         radian_alignment_saved = pickle.load(fp)
# initialize var
radian_alignment_saved = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
# cm_window = [10,20,30,40,50,60,70,80] # cm
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if ((animal!='e217') & (animal!='e200')) & (conddf.optoep.values[ii]==-1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
        pcs = np.vstack(np.array(fall['putative_pcs'][0]))
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))        
        lasttr=8 # last trials
        bins=90
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1 epoch
        pc_bool = np.sum(pcs,axis=0)>=len(eps)-2        
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        bin_size=3 # cm
        # get abs dist tuning 
        tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
            Fc3,trialnum,rewards,forwardvel,
            rewsize,bin_size)
        # get cells that maintain their coms across at least 2 epochs
        place_window = 15 # cm converted to rad                
        perm = list(combinations(range(len(coms_correct_abs)), 2))     
        com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
        # get cells across all epochs that meet crit
        pcs = np.unique(np.concatenate(compc))
        pcs_all = intersect_arrays(*compc)
        # get per comparison
        pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
        pc_ind.append(pcs_all);pc_p=len(pcs_all)/len(coms_correct_abs[0])
        epoch_perm.append(perm)
        pc_prop.append([pcs_p_per_comparison,pc_p])
        num_epochs.append(len(coms_correct_abs))

        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'steelblue']
        coms_all.append(coms_correct_abs)
        for gc in pcs:
            fig, ax = plt.subplots()
            for ep in range(len(coms_correct_abs)):
                ax.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
                ax.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,10))
                ax.set_xticklabels(np.arange(0,track_length+bin_size*10,bin_size*10).astype(int))
                ax.set_xlabel('Absolute position (cm)')
                ax.set_ylabel('Fc3')
                ax.spines[['top','right']].set_visible(False)
            ax.legend()
        #     plt.savefig(os.path.join(savedst, 'true_place_cell.png'), bbox_inches='tight', dpi=500)
            plt.close('all')
            pdf.savefig(fig)
        
pdf.close()
#%%

plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep==-1) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['place_cell_prop'] = [xx[1] for xx in pc_prop]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['place_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
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
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)

df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(5,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,
        s=8)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)

eps = [2,3,4]
y = 0.3
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii, y+.05, f'p={pval:.3g}')

plt.savefig(os.path.join(savedst, 'nearrew_cell_prop_per_an.svg'), 
        bbox_inches='tight')
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
plt.savefig(os.path.join(savedst, 'rec_cell_nearrew_prop_per_an.svg'), 
        bbox_inches='tight')