
#%%
"""
zahra
jan 2025
isolate locomotion cells
"""

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
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
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,extract_data_nearrew,perireward_binned_activity
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'near_rew_acc.pdf')
#%%
goal_window_cm=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# radian_alignment_saved = {} # overwrite
goal_cell_iinds = []
goal_cell_props = []
goal_cell_nulls = []
num_epochs = []
pvals = []
rates_all = []
total_cells_all = []
epoch_perm = []
radian_alignment = {}
lasttr=8 #  last trials
bins=90
num_iterations=1000
saveto = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'

# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii] 
    animal = conddf.animals.values[ii]
    if animal!='e217' and conddf.optoep.values[ii]<2:
        pln=0
        if animal=='e145' or animal=='e139': pln=2
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat'])
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
                rewards=rewards[:-1]        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
        rate = success/total_trials
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        dFF=dFF[:,skew>2]
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av,pdist = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # 9/19/24
            # find correct trials within each epoch!!!!
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)          
        fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
        ops = fall_stat['ops']
        stat = fall_stat['stat']
        meanimg=np.squeeze(ops)[()]['meanImg']
        s2p_iind = np.arange(stat.shape[1])
        s2p_iind_filter = s2p_iind[((fall['iscell'][:,0]).astype(bool))]
        s2p_iind_filter = s2p_iind_filter[skew>2]
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        # only get cells near reward        
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])                
        # tuning curves that are close to each other across epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0))] for com in com_goal if len(com)>0]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        if len(goal_cells)>0:
            s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
        else: 
            s2p_iind_goal_cells=[]
        ### get acc correlation
        velocity = fall['forwardvel'][0]
        veldf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(veldf.rolling(5).mean().values)
        acc=velocity[1:]/np.diff(fall['timedFF'][0])
        acc_ = pd.DataFrame(acc).interpolate(method ='linear',limit_direction ='backward').values
        acc_ = pd.DataFrame(acc_).interpolate(method ='linear',limit_direction ='forward').values[:,0]

        racc = [scipy.stats.pearsonr(acc_,dFF[1:,gc])[1] for gc in goal_cells]
        # threshold of .2 correlation
        thres=0.1
        goal_cells = [gc for ii,gc in enumerate(goal_cells) if racc[ii]>thres]
        # get per comparison
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        goal_cell_iind=goal_cells
        goal_cell_p=len(goal_cells)/len(coms_correct[0])
        goal_cell_prop=[goal_cells_p_per_comparison,goal_cell_p]
        num_epoch=len(coms_correct)
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        if len(goal_cells)>0:
            rows = int(np.ceil(np.sqrt(len(goal_cells))))
            cols = int(np.ceil(len(goal_cells) / rows))
            fig, axes = plt.subplots(rows, cols, sharex=True)
            if len(goal_cells) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(goal_cells):            
                for ep in range(len(coms_correct)):
                    ax = axes[i]
                    ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    ax.axvline((bins/2), color='k')
                    ax.set_title(f'cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,bins+1,20))
            ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
            ax.set_xlabel('Radian position (centered start rew loc)')
            ax.set_ylabel('Fc3')
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # get shuffled iterations
        shuffled_dist = np.zeros((num_iterations))
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
            # get goal cells across all epochs
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            # (com near goal)
            com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)>0))] for com in com_goal if len(com)>0]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
            if len(com_goal)>0:
                goal_cells_shuf = intersect_arrays(*com_goal)
            else:
                goal_cells_shuf=[]
            ### get acc correlation
            velocity = fall['forwardvel'][0]
            veldf = pd.DataFrame({'velocity': velocity})
            velocity = np.hstack(veldf.rolling(5).mean().values)
            acc=velocity[1:]/np.diff(fall['timedFF'][0])
            # interpoalte for nans
            acc_ = pd.DataFrame(acc).interpolate(method ='linear',limit_direction ='backward').values
            acc_ = pd.DataFrame(acc_).interpolate(method ='linear',limit_direction ='forward').values[:,0]
            racc=[]
            for gc in goal_cells_shuf:
                dff = pd.DataFrame(dFF[1:,gc]).interpolate(method ='linear',limit_direction ='backward').values
                dff = pd.DataFrame(dff).interpolate(method ='linear',limit_direction ='forward').values[:,0]
                r = scipy.stats.pearsonr(acc_,dff)[1]
                racc.append(r)
            # threshold of .2 correlation
            goal_cells_shuf = [gc for ii,gc in enumerate(goal_cells_shuf) if racc[ii]>thres]
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
        # save median of goal cell shuffle
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
        goal_cell_null=[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
        total_cells=len(coms_correct[0])
        # save
        rates_all.append(rate); pvals.append(p_value); total_cells_all.append(total_cells)
        epoch_perm.append(perm); goal_cell_iinds.append(goal_cell_iind); 
        goal_cell_props.append(goal_cell_prop); num_epochs.append(num_epoch)
        goal_cell_nulls.append(goal_cell_null)
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av,goal_cells]
pdf.close()

# save pickle of dcts
with open(saveto, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp)
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df=df[df.index.isin(inds)]
df = df[((df.animals!='e217') | (df.animals!='e145') | (df.animals!='e139')) & (df.optoep<2)]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_props]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_nulls]
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False ,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(3,5))
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
# epcomp= [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_props]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_nulls]
# df_perms['epoch_comparison']=
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
# df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

df_perms = df_perms[df_perms.animals!='e189']
# skipped fro now because it wasn't working
# df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

# fig,ax = plt.subplots(figsize=(7,5))
# sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
#         hue='animals',data=df_permsav,
#         s=8,ax=ax)
# sns.barplot(x='epoch_comparison', y='goal_cell_prop',
#         data=df_permsav,
#         fill=False,ax=ax, color='k', errorbar='se')
# ax = sns.lineplot(data=df_permsav, # correct shift
#         x='epoch_comparison', y='goal_cell_prop_shuffle',
#         color='grey', label='shuffle')

# ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))
# #%%
# eps = df_permsav.index.get_level_values("epoch_comparison").unique()
# for ep in eps:
#     # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
#     rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
#     shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
#     t,pval = scipy.stats.ranksums(rewprop, shufprop)
#     print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)

df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,
        s=10,alpha=0.7)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None)

ax.spines[['top','right']].set_visible(False)
ax.legend()
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Post reward cell proportion')
eps = [2,3,4]
y = 0.18
pshift=.03
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop[~np.isnan(shufprop.values)], shufprop.values[~np.isnan(shufprop.values)])
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)
ax.set_title('Post-reward cells',pad=90)
plt.savefig(os.path.join(savedst, 'postrew_cell_prop_per_an.svg'), 
        bbox_inches='tight')
df_plt2=df_plt2.reset_index()

#%%
# find tau/decay
from scipy.optimize import curve_fit
# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)
tau_all = []
for an in df_plt2.animals.unique():
        try:
                # Initial guesses for the optimization
                initial_guess = [4, 2]  # Amplitude guess and tau guess
                y = df_plt2[df_plt2.animals==an]
                t=np.array([2,3,4])
                # Fit the model to the data using curve_fit
                params, params_covariance = curve_fit(exponential_decay, t, y.goal_cell_prop.values, p0=initial_guess)
                # Extract the fitted parameters
                A_fit, tau_fit = params
                tau_all.append(tau_fit)
                # Generate the fitted curve using the optimized parameters
                y_fit = exponential_decay(t, A_fit, tau_fit)
        except:
                print(an)


#%%
# as a function of session/day
df_plt = df.groupby(['animals','session_num','num_epochs']).mean(numeric_only=True)
df_permsav2 = df_perms.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[(df_plt2.index.get_level_values('num_epochs')==2) & (df_plt2.index.get_level_values('animals')!='e200')]
df_plt2 = df_plt2.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(7,5))
# av across mice
sns.stripplot(x='session_num', y='goal_cell_prop',hue='animals',
        data=df_plt2,s=10,alpha=0.7)
sns.barplot(x='session_num', y='goal_cell_prop',color='darkslateblue',
        data=df_plt2,fill=False,ax=ax, errorbar='se')
ax.set_xlim([-.5,9.5])
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('# of sessions')
ax.set_ylabel('Reward-distance cell proportion')
df_reset = df_plt2.reset_index()
sns.regplot(x='session_num', y='goal_cell_prop',
        data=df_reset, scatter=False, color='k')
r, p = scipy.stats.pearsonr(df_reset['session_num'], 
        df_reset['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.3f}, p={:.3g}'.format(r, p),
        transform=ax.transAxes)
ax.set_title('2 epoch combinations')

#%%
# per session
df_plt2 = pd.concat([df_perms,df])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[(df_plt2.num_epochs<5) & (df_plt2.num_epochs>1)]
# df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(6,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='animals',s=10,alpha=0.4,
        data=df_plt2,dodge=True)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='animals',
        data=df_plt2,
        fill=False,ax=ax, errorbar='se')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.1))
ax.set_ylabel('Post reward cell proportion')
ax.set_title('Post-reward cells')
plt.savefig(os.path.join(savedst, 'postrew_cell_prop_per_session.svg'), 
        bbox_inches='tight')
#%%
df['success_rate'] = rates_all
dffil = df[df.goal_cell_prop>0]
dffil=dffil[dffil.success_rate>.6]
# all animals
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='success_rate', y='goal_cell_prop',hue='animals',
        data=dffil,
        s=100, ax=ax)
sns.regplot(x='success_rate', y='goal_cell_prop',
        data=dffil,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(dffil['success_rate'], 
        dffil['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Success rate')
ax.set_ylabel('Post-reward cell proportion')
plt.savefig(os.path.join(savedst, 'postrew_v_correctrate.svg'), 
        bbox_inches='tight')
#%%

an_nms = dffil[dffil.animals!='e189'].animals.unique()
num_plots = len(an_nms)
rows = int(np.ceil(np.sqrt(num_plots)))
cols = int(np.ceil(num_plots / rows))

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
axes = axes.flatten()  # Flatten the axes array for easier plotting

for i, an in enumerate(an_nms):
        ax = axes[i]
        sns.scatterplot(x='success_rate', y='goal_cell_prop', data=dffil[(dffil.animals == an)], s=100, ax=ax)
        sns.regplot(x='success_rate', y='goal_cell_prop', data=dffil[(dffil.animals == an)], ax=ax, scatter=False, color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(an)
        try:
                r, p = scipy.stats.pearsonr(dffil[(dffil.animals == an)]['success_rate'], 
                dffil[(dffil.animals == an)]['goal_cell_prop'])
                ax.text(.2, .5, 'r={:.2f}, p={:.2g}'.format(r, p),
                        transform=ax.transAxes)
        except Exception as e:
                print(e)

# Hide any remaining unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

#%%
