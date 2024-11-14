
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
from rewardcell import get_radian_position, acc_corr_cells
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\retreat_2024'
savepth = os.path.join(savedst, 'acc_corr_correcttr_skewfilt.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\acc_corr_cell_bytrialtype_nopto.p"
tcsave = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(tcsave, "rb") as fp: #unpickle
        tcsave = pickle.load(fp)
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
cm_window = 10
# cm_window = [10,20,30,40,50,60,70,80] # cm
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'timedFF'])
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
        if f'{animal}_{day:03d}_index{ii:03d}' in tcsave.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,\
            goal_cell_shuf_ps_av = tcsave[f'{animal}_{day:03d}_index{ii:03d}']            
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
        com_near_rew = 20*(2*np.pi/track_length)
        # find coms near rew
        post_rew_coms = [np.where((xx>0) & (xx<com_near_rew))[0] for xx in coms_rewrel]        
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*post_rew_coms)
        # get per epoch NOT PER COMPARISON
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in post_rew_coms]
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms_correct[0])
        goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
        assert len(coms_correct)==len(goal_cells_p_per_comparison)
        num_epochs.append(len(coms_correct))
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod']
        for gc in goal_cells:
            fig, ax = plt.subplots()
            for ep in range(len(coms_correct)):
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins/2), color='k')
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                ax.set_xticks(np.arange(0,bins+1,10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),3))
                ax.set_xlabel('Radian position (centered at start of rew loc)')
                ax.set_ylabel('Fc3')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iterationsollllllpoik
        num_iterations = 5000; shuffled_dist = np.zeros((num_iterations))
        # max of 5 epochs = 10 perms
        goal_cell_shuf_ps_per_comp = np.ones((num_iterations,6))*np.nan
        goal_cell_shuf_ps = []
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
            # find coms near rew
            com_goal_shuf = [np.where((xx>0) & (xx<com_near_rew))[0] for xx in coms_rewrel]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_shuf]
            goal_cells_shuf = intersect_arrays(*com_goal_shuf)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
        # save median of goal cell shuffle
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
        goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value); 
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
        total_cells.append(len(coms_correct[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                        post_rew_coms, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]

pdf.close()

# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<1)]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt[df_plt.num_epochs<5]
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize=(3,5))
sns.stripplot(x='num_epochs', y='goal_cell_prop',data=df_plt,
        s=10, color = 'k',ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', ax=ax)
ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))

ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Post-reward cell proportion')
eps = [2,3,4]
y = 0.1
pshift = 0.012
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10)
        
plt.savefig(os.path.join(savedst, 'acc_post_rew_cells_sig.svg'), bbox_inches='tight')
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

# plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
#         bbox_inches='tight')

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
