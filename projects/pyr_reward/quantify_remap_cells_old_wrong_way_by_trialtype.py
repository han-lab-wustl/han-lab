"""
calculate proportion of goal cells using old way
eleonora's method of using absolute distance
added ways to take mean across null distribution
matched to radian alignment distance method for getting goal cells
zahra
aug 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.backends.backend_pdf
from itertools import combinations
import matplotlib as mpl
from placecell import intersect_arrays
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.opto.analysis.pyramdial.placecell import consecutive_stretch_time

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data'
savepth = os.path.join(savedst, '20cm_window_goal_cells_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []

for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]==-1):
        day = conddf.days.values[ii]
        pln=0
        if animal=='e145': pln=2
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
        #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_early_trials',
            'tuning_curves_late_trials', 'coms', 'coms_early_trials', 'trialnum', 'rewards',
            'ybinned', 'forwardvel', 'timedFF', 'VR'])        
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum = fall['trialnum'][0]; rewards = fall['rewards'][0]
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf; timedFF = fall['timedFF'][0]
        forwardvel = fall['forwardvel'][0]
        eptest = conddf.optoep.values[ii]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))    
        # exclude last ep if too little trials
        lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
        if lastrials<8:
            eps = eps[:-1]            
        bin_size = 3    
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]    
        goal_window = 20 # cm
        coms = np.array([np.hstack(xx) for xx in coms])
        # relative to reward
        coms_rewrel = np.array([com-rewlocs[ii] for ii, com in enumerate(coms)])                 
        perm = list(combinations(range(len(coms)), 2))     
        epoch_perm.append(perm)
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in coms_rewrel]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        # get per comparison
        goal_cells_p_per_comparison = [len(xx)/len(coms[0]) for xx in com_goal]

        dist_to_rew.append(np.array([com-rewlocs[ii] for ii, com in enumerate(coms)]))        
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms[0])
        goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
        num_epochs.append(len(coms))
        colors = ['navy', 'red', 'green', 'k','darkorange']
        for gc in goal_cells:
            fig, ax = plt.subplots()
            for ep in range(len(coms)):
                ax.plot(tcs_late[ep][gc,:], label=f'epoch {ep}', color=colors[ep])
                ax.axvline(rewlocs[ep]/bin_size, color=colors[ep])
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iterations
        num_iterations = 5000
        shuffled_dist = np.zeros((num_iterations))
        goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms[ii].shape[0])) for ii in range(1, len(coms))]
            [random.shuffle(shuf) for shuf in shufs]
            com_shufs = np.zeros_like(coms)
            com_shufs[0,:] = coms[0]
            com_shufs[1:1+len(shufs),:] = [coms[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-rewlocs_shuf[ii] for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms[0]) for xx in com_goal]
            goal_cells_shuf = intersect_arrays(*com_goal); shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
        # save median of goal cell shuffle
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
        goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        total_cells.append(len(coms[0]))
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')

pdf.close()
# %%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep==-1)]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
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
#%%
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
y = 0.45
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
