"""
calculate proportion of goal cells
zahra
june 2024
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
savepth = os.path.join(savedst, 'goal_cells_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%

goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates = []
total_cells = []
time_per_rew_bout_mean = [] # s time the animal spends in rew loc

for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    if not animal=='e217':
        day = conddf.days.values[ii]
        plane=0 #TODO: make modular  
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{plane}_Fall.mat"
        # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
        #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_early_trials',
            'tuning_curves_late_trials', 'coms', 'coms_early_trials', 'trialnum', 'rewards',
            'ybinned', 'forwardvel', 'timedFF', 'VR'])        
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum = fall['trialnum'][0]; rewards = fall['rewards'][0]
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
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
        # quantify performance
        rate = []; time_per_rew_bout_ep = []
        for ep in range(len(eps)-1):
            rewloc = rewlocs[ep]
            eprng = range(eps[ep], eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            time = timedFF[eprng]
            # only for correct trials
            correct = np.array([xx in str_trials for xx in trialnum[eprng]])
            time = time[correct]
            ypos = ybinned[eprng][correct]
            # speed = forwardvel[eprng][correct] # correct for animals that lick outside rewloc
            # time_in_rew = time[(ypos>(rewloc-rewsize)) & (ypos<(rewloc+rewsize))]
            # time_in_rew_ =consecutive_stretch_time(time_in_rew)
            # time_per_rew_bout = [xx[-1]-xx[0] for xx in time_in_rew_]
            # time_per_rew_bout_ep.append(np.nanmean(time_per_rew_bout))
            rate.append(success/total_trials)
        time_per_rew_bout_mean.append(np.nanmean(time_per_rew_bout_ep))
        rates.append(np.nanmean(np.array(rate)))
        bin_size = 3    
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]    
        window = 20 # cm
        goal_window = 10 # cm
        coms = np.array([np.hstack(xx) for xx in coms])
        # relative to reward
        coms_rewrel = np.array([com-rewlocs[ii] for ii, com in enumerate(coms)])                 
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        dist_to_rew.append(np.array([com-rewlocs[ii] for ii, com in enumerate(coms)]))        
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms[0])
        goal_cell_prop.append(goal_cell_p)
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
        num_iterations = 1000
        shuffled_dist = np.zeros((num_iterations))
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
            goal_cells_shuf = intersect_arrays(*com_goal)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        total_cells.append(len(coms[0]))
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')

pdf.close()
# %%
# goal cells across epochs
df = conddf[conddf.animals!='e217']
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df[df.opto==False],
        s=8)
ax.spines[['top','right']].set_visible(False)
#%%
# split into pre and post reward cells
pre_rew = [[cellind for cellind in range(xx[:,goal_cell_iind[kk]].shape[1]) if np.nanmedian(xx[:,goal_cell_iind[kk]][:,cellind])<0] for kk,xx in enumerate(dist_to_rew)]
post_rew = [[cellind for cellind in range(xx[:,goal_cell_iind[kk]].shape[1]) if np.nanmedian(xx[:,goal_cell_iind[kk]][:,cellind])>0] for kk,xx in enumerate(dist_to_rew)]
pre_rew_prop = [len(xx)/total_cells[ii] for ii,xx in enumerate(pre_rew)]
post_rew_prop = [len(xx)/total_cells[ii] for ii,xx in enumerate(post_rew)]

df['pre_rew_prop'] = pre_rew_prop
df['post_rew_prop'] = post_rew_prop
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='pre_rew_prop',
        hue='animals',data=df[(df.opto==False)&(df.p_value<0.05)],
        s=8)
ax.spines[['top','right']].set_visible(False)
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='post_rew_prop',
        hue='animals',data=df[(df.opto==False)&(df.p_value<0.05)],
        s=8)
ax.spines[['top','right']].set_visible(False)
#%%
df['success_rate'] = rates

an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    sns.scatterplot(x='success_rate', y='post_rew_prop',
            data=df[(df.animals==an)&(df.opto==False)&(df.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()
#%%
df['recorded_neurons_per_session'] = total_cells
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df[(df.opto==False)&(df.p_value<0.05)],
        s=150, ax=ax)
ax.spines[['top','right']].set_visible(False)
#%%
# histogram of dist to rew
df['com_distance_to_reward'] = [np.concatenate(xx) for xx in dist_to_rew]
an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    dists = df.loc[df.animals==an, 'com_distance_to_reward']
    for dist in dists:
        ax.hist(dist,bins=100)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()
ax.set_ylabel('# Cells')
ax.set_xlabel('Distance to reward \n\
    (COM-rewardloc.)')

#%%
# only reward cells
df['com_distance_to_reward_goal_cells'] = [np.concatenate(xx[:,goal_cell_iind[ii]]) for ii,xx in enumerate(dist_to_rew)]
an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    dists = df.loc[df.animals==an, 'com_distance_to_reward_goal_cells']
    for dist in dists:
        ax.hist(dist)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
ax.set_ylabel('# Cells')
ax.set_xlabel('Distance to reward \n\
    (COM-rewardloc.)')
fig.suptitle('Reward cells only')
fig.tight_layout()

#%%
# histograms of p-values
# shuffled reward loc
fig,ax = plt.subplots()
ax.hist(pvals,bins=40)
ax.axvline(0.05,color='k',linewidth=2,linestyle='--')
ax.set_ylabel('Sessions')
ax.set_xlabel('P-value')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Reward cell proportion compared\nto shuffled cell indicies')
# %%

dfagg = df#.groupby(['animals', 'opto', 'condition']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize=(8,6))
ax = sns.barplot(x='animals', y='p_value',
        hue='opto',data=dfagg, 
        fill=False)
ax = sns.stripplot(x='animals', y='p_value',
        hue='opto',data=dfagg, 
        s=10)
ax.spines[['top','right']].set_visible(False)

# ax.axhline(0.05,color='k',linewidth=2,linestyle='--')
# ax.get_legend().set_visible(False)

#%% 
# compare reward cell proportion vs. number of sessions recorded
an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]    
    df_an = df[df.animals==an]
    sessions = list(df_an.reset_index().index)
    df_an['sessions'] = sessions
    sns.scatterplot(x='sessions', y='post_rew_prop',
            data=df_an[(df_an.opto==False)&(df_an.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
ax.set_ylabel('% Reward cells of pyramidal cells')
ax.set_xlabel('sessions')
fig.suptitle('Sessions vs. proportion of reward cells')
fig.tight_layout()
