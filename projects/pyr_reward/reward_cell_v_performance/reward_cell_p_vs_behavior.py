
"""
zahra
get tuning curves with dark time
reward cell p vs. behavior
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,\
    make_tuning_curves_by_trialtype_w_darktime,get_radian_position_first_lick_after_rew_w_dt
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials, get_lick_selectivity
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

#%%
####################################### RUN CODE #######################################
# initialize var
radian_alignment_saved = {} # overwrite
p_goal_cells=[]
p_goal_cells_dt = []
goal_cells_iind=[]
pvals = []
bins = 90
datadct = {}
goal_cell_null= []
perms = []
cm_window = 20 # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):        
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
                'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                'stat', 'licks'])
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
        licks=fall['licks'][0]
        time=fall['timedFF'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            licks=licks[:-1]
            time=time[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        lasttr = 8
        rates = []; ls_all = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
            # lick selectivity
            mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum[eprng]])
            ls = get_lick_selectivity(ybinned[eprng][mask], trialnum[eprng][mask], licks[eprng][mask], rewlocs[ep], rewsize,
                    fails_only = False)
            ls_all.append(np.nanmean(ls))

        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates

        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # tc w/ dark time
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct)
        # get null goal cells
        goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist=goal_cell_shuffle(coms_correct, goal_window,\
                    perm,num_iterations = 1000)
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
        # behavioral metrics
        performance_per_comparison = [np.nanmean([rates[p[0]],rates[p[1]]]) for p in perm]
        lick_sel_per_comparison = [np.nanmean([ls_all[p[0]],ls_all[p[1]]]) for p in perm]
        vel_per_comparison = [np.nanmean([forwardvel[eps[p[0]:p[0]+1]],forwardvel[eps[p[1]:p[1]+1]]]) for p in perm]
        null_per_comparison = np.nanmean(goal_cell_shuf_ps_per_comp[:,:len(perm)],axis=(0))
        # save perm        
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct,goal_cells_p_per_comparison,performance_per_comparison,lick_sel_per_comparison, vel_per_comparison, null_per_comparison, perm, rz_perm]

pdf.close()
####################################### RUN CODE #######################################
#%%
plt.rc('font', size=20)          # controls default text sizes

reward_cell_p_per_comp = [v[2] for k,v in datadct.items()]
successrate_p_per_comp = [v[3] for k,v in datadct.items()]
ls_p_per_comp = [v[4] for k,v in datadct.items()]
vel_per_comp = [v[5] for k,v in datadct.items()]
null_reward_cell_p_per_comp = [v[6] for k,v in datadct.items()]

animal_df = [k.split('_')[0] for k,v in datadct.items()]
df=pd.DataFrame()
df['reward_cell_p'] = np.concatenate(reward_cell_p_per_comp)
df['null_reward_cell_p'] =  np.concatenate(null_reward_cell_p_per_comp)
df['success_rate'] = np.concatenate(successrate_p_per_comp)

df['reward_cell_p'] = df['reward_cell_p']*100
df['success_rate'] =df['success_rate']*100
df['lick_selectivity'] = np.concatenate(ls_p_per_comp)
df['velocity'] = np.concatenate(vel_per_comp)
df['animal'] = np.concatenate([[an]*len(reward_cell_p_per_comp[kk]) for kk,an in enumerate(animal_df)])
df=df[(df.animal!='e189')&(df.animal!='e190')]
a=0.4
# Assume df is already defined with 'success_rate', 'reward_cell_p', 'animal'

a = 0.4
animals = df['animal'].unique()
n_animals = len(animals)

# Determine square-ish grid size
n_cols = int(np.ceil(np.sqrt(n_animals)))
n_rows = int(np.ceil(n_animals / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3* n_rows), sharex=True, sharey=True)
axes = axes.flatten()

for idx, animal in enumerate(animals):
    ax = axes[idx]
    subdf = df[df['animal'] == animal]
    
    sns.regplot(
        x='success_rate',
        y='reward_cell_p',
        data=subdf,
        scatter_kws={'color':'k','alpha': a,'s':70},
        line_kws={'color': 'cornflowerblue'},
        ax=ax
    )    
    r, p = scipy.stats.pearsonr(subdf['success_rate'], subdf['reward_cell_p'])
    ax.set_title(f'{animal}\nr={r:.2f}, p={p:.3g}',fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines[['top','right']].set_visible(False)
    
ax.set_xlabel('% Correct trials')
ax.set_ylabel('% Reward cell')

# Hide any unused subplots
for ax in axes[n_animals:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'reward_cell_v_successrate.svg'), 
        bbox_inches='tight')
#%%
# Determine square-ish grid size
n_cols = int(np.ceil(np.sqrt(n_animals)))
n_rows = int(np.ceil(n_animals / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3* n_rows), sharex=True, sharey=True)
axes = axes.flatten()

for idx, animal in enumerate(animals):
    ax = axes[idx]
    subdf = df[df['animal'] == animal]
    
    sns.regplot(
        x='lick_selectivity',
        y='reward_cell_p',
        data=subdf,
        scatter_kws={'color':'k','alpha': a,'s':70},
        line_kws={'color': 'cornflowerblue'},
        ax=ax
    )    
    r, p = scipy.stats.pearsonr(subdf['lick_selectivity'], subdf['reward_cell_p'])
    ax.set_title(f'{animal}\nr={r:.2f}, p={p:.3g}',fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines[['top','right']].set_visible(False)
    
ax.set_xlabel('Lick selectivity\n(last 8 trials)')
ax.set_ylabel('% Reward cell')

# Hide any unused subplots
for ax in axes[n_animals:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'reward_cell_v_lick_selectivity.svg'), 
        bbox_inches='tight')
#%%
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3* n_rows), sharex=True, sharey=True)
axes = axes.flatten()

for idx, animal in enumerate(animals):
    ax = axes[idx]
    subdf = df[df['animal'] == animal]
    
    sns.regplot(
        x='velocity',
        y='reward_cell_p',
        data=subdf,
        scatter_kws={'color':'k','alpha': a,'s':70},
        line_kws={'color': 'cornflowerblue'},
        ax=ax
    )    
    r, p = scipy.stats.pearsonr(subdf['velocity'], subdf['reward_cell_p'])
    ax.set_title(f'{animal}\nr={r:.2f}, p={p:.3g}',fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines[['top','right']].set_visible(False)
    
ax.set_xlabel('Velocity (cm/s)')
ax.set_ylabel('% Reward cell')

# Hide any unused subplots
for ax in axes[n_animals:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'reward_cell_v_velocity.svg'), 
        bbox_inches='tight')
#%%
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

test_results = []
xvars = ['success_rate', 'lick_selectivity', 'velocity']
labels = ['% Correct trials', 'Lick selectivity', 'Velocity']

for xvar, label in zip(xvars, labels):
    real_r, null_r = [], []

    for animal in animals:
        subdf = df[df['animal'] == animal]

        # real r
        r_real, _ = scipy.stats.pearsonr(subdf[xvar], subdf['reward_cell_p'])

        # null r
        r_null, _ = scipy.stats.pearsonr(subdf[xvar], subdf['null_reward_cell_p'])

        real_r.append(r_real)
        null_r.append(r_null)

    # Paired t-test
    tval, pval = scipy.stats.wilcoxon(real_r, null_r)
    test_results.append((label, pval, real_r, null_r))

# Bonferroni correction
labels, raw_pvals, real_r_all, null_r_all = zip(*test_results)
_, pvals_corr, _, _ = multipletests(raw_pvals, method='bonferroni')

pl = ['cornflowerblue', 'grey']
s=12
a=0.7
# Plotting
fig, axes = plt.subplots(ncols=3,figsize=(9,5),sharex=True)
for i, label in enumerate(labels):
    real_r = real_r_all[i]
    null_r = null_r_all[i]
    pval = pvals_corr[i]
    if label=='% Correct trials':
        real_r=[r*100 for r in real_r]
        null_r=[n*100 for n in null_r]
    plot_df = pd.DataFrame({
        'animal': animals.tolist() * 2,
        'r_value': real_r + null_r,
        'type': ['Real'] * len(real_r) + ['Shuffle'] * len(null_r)
    })

    ax=axes[i]
    sns.barplot(data=plot_df, x='type', y='r_value',hue='type', errorbar='se', palette=pl, fill=False,ax=ax)
    sns.stripplot(data=plot_df, x='type', y='r_value', hue='type',palette=pl, alpha=a, jitter=True,s=s,ax=ax)
    # Add connecting lines
    plot_df_l = pd.DataFrame({
    'animal': animals.tolist(),
    'real': real_r,
    'null': null_r
})
    for _, row in plot_df_l.iterrows():
        ax.plot([0,1], [row['real'], row['null']], color='gray', alpha=0.5, zorder=1,linewidth=1.5)

    # Significance star
    if pval < 0.001:
        star = '***'
    elif pval < 0.01:
        star = '**'
    elif pval < 0.05:
        star = '*'
    else:
        star = 'n.s.'

    ymax = plot_df['r_value'].max()
    ax.text(0.5, ymax - 0.02, f'{pval:.2g}', ha='center', va='bottom', fontsize=12)
    ax.set_title(f'{label}')
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('')
    sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(savedst, f'real_vs_null_r.svg'), bbox_inches='tight')
plt.show()
