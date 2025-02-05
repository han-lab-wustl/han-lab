
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
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto_paramsweep.p"
# import tuning curve
tc = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(tc, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
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
cm_windows = [10,20,30,40,50,60,70,80,100,120,150,200] # cm
#%%
# iterate through all animals
for cm_window in cm_windows:
    for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]==-1):
            if animal=='e145': pln=2 
            else: pln=0
            params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
            print(params_pth)
            fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
                'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                'stat'])
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
            eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
            tcs_early = []; tcs_late = []        
            ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
            lasttr=8 # last trials
            bins=90
            rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
            track_length_rad = track_length*(2*np.pi/track_length)
            bin_size=track_length_rad/bins
            success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
            rates_all.append(success/total_trials)
            if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
                tcs_correct, coms_correct, tcs_fail, coms_fail, \
                com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
            else:# remake tuning curves relative to reward        
                # takes time
                fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
                Fc3 = fall_fc3['Fc3']
                dFF = fall_fc3['dFF']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
                # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                # skew_mask = skew_filter>2
                Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
                tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                    rewards,forwardvel,rewsize,bin_size)          
            goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
            # change to relative value 
            coms_rewrel = np.array([com-np.pi for com in coms_correct])
            perm = list(combinations(range(len(coms_correct)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            dist_to_rew.append(coms_rewrel)
            # get goal cells across all epochs        
            goal_cells = intersect_arrays(*com_goal)
            # get per comparison
            goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
            goal_cell_iind.append(goal_cells);goal_cell_p=len(goal_cells)/len(coms_correct[0])
            epoch_perm.append(perm)
            goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p]);num_epochs.append(len(coms_correct))
            # get shuffled iterations
            num_iterations = 5000; shuffled_dist = np.zeros((num_iterations))
            # max of 5 epochs = 10 perms
            goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
            for i in range(num_iterations):
                # shuffle locations
                rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
                shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
                [random.shuffle(shuf) for shuf in shufs]
                # first com is as ep 1, others are shuffled cell identities
                com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
                com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
                # OR shuffle cell identities
                # relative to reward
                coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
                perm = list(combinations(range(len(coms_correct)), 2))     
                com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
                # get goal cells across all epochs
                com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
                goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
                goal_cells_shuf = intersect_arrays(*com_goal); shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
                goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
                goal_cell_shuf_ps.append(goal_cell_shuf_p)
                goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
            # save median of goal cell shuffle
            goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
            goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
            goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
            p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
            pvals.append(p_value);             
            print(f'{animal}, day {day}, window {cm_window}\n significant goal cells proportion p-value: {p_value}')
            total_cells.append(len(coms_correct[0]))
            radian_alignment[f'{animal}_{day:03d}_index{ii:03d}_cm_window{cm_window:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]


# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
windows = [int(xx[-3:]) for xx in radian_alignment.keys()]
inds = [int(xx[-16:-13]) for xx in radian_alignment.keys()]
df = pd.DataFrame()
orgdf = conddf[((conddf.animals!='e217')) & (conddf.optoep==-1) & (conddf.index.isin(inds))]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]
df['window'] = windows
df['animals'] = np.concatenate([orgdf.animals.values]*len(cm_windows))

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df, x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.p_value.values<0.05)/len(df)
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs','window']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='window',data=df_plt,
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
for cm_window in cm_windows:
    for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_plt.loc[((df_plt.index.get_level_values('num_epochs')==ep) & 
                            (df_plt.index.get_level_values('window')==cm_window)), 'goal_cell_prop']
        shufprop = df_plt.loc[((df_plt.index.get_level_values('num_epochs')==ep) & 
                            (df_plt.index.get_level_values('window')==cm_window)), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ranksums(rewprop, shufprop)
        print(f'{ep} epochs, window {cm_window}cm, pval: {pval}')
#%%    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perm_window = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.window.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perms['window'] = np.concatenate(df_perm_window)
df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison','window']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
        hue='window',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs', 'window']).mean(numeric_only=True)

cols = 4 # ~ sqrt of number of windows
fig,axes = plt.subplots(nrows=cols,ncols=cols,figsize=(15,15),sharex=True)
r=0;c=0
for ii,win in enumerate(cm_windows):
    df_plt2 = pd.concat([df_permsav2,df_plt])
    df_plt2 = df_plt2[df_plt2.index.get_level_values('window')==win]
    # df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
    df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
    # number of epochs vs. reward cell prop incl combinations    
    # av across mice
    ax = axes[r,c]
    sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
            data=df_plt2,s=8,ax=ax)
    sns.barplot(x='num_epochs', y='goal_cell_prop',
            data=df_plt2,
            fill=False,ax=ax, color='k', errorbar='se')
    ax = sns.lineplot(data=df_plt2, # correct shift
            x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
            label='shuffle',ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.legend().set_visible(False)
    ax.set_title(f'window = {win:02d}')
    eps = [2,3,4]
    y = 0.3+ii*.07
    pshift = 0.1+ii*.01
    fs=36
    for ii,ep in enumerate(eps):
            rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
            shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
            t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
            print(f'{ep} epochs, pval: {pval}')
            # statistical annotation        
            if pval < 0.001:
                    ax.text(ii, y, "***", ha='center', fontsize=fs)
            elif pval < 0.01:
                    ax.text(ii, y, "**", ha='center', fontsize=fs)
            elif pval < 0.05:
                    ax.text(ii, y, "*", ha='center', fontsize=fs)
            ax.text(ii, y+pshift, f'p={pval:.2g}',fontsize=12)    
    # for row / column
    if r==(cols-1):
        r=0; c+=1
    else:
        r+=1
fig.tight_layout()
#%%
# plot p value as a function of window
fig,ax = plt.subplots()
sns.stripplot(x='window', y='p_value',hue='num_epochs',data=df_plt2[df_plt2.index.get_level_values('num_epochs')<5],s=7,ax=ax,
            palette='colorblind')
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Window for distance (cm)')
ax.axhline(0.05, color='slategray',linestyle='--')
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
