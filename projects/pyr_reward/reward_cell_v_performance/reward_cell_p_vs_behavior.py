
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
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle, \
    intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, get_lick_selectivity,smooth_lick_rate
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
bins = 90
datadct = {}
cm_window = 20 # cm
num_iterations=1000
#%%
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
                'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
        fr = 31.25
        if animal=='e190' or animal=='z9':
            fr=fr/2
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
        rates = []; ls_all = []; lr_all = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
            # lick rate and selecitivty
            mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum[eprng]])
            ls = get_lick_selectivity(ybinned[eprng][mask], trialnum[eprng][mask], licks[eprng][mask], rewlocs[ep], rewsize,
                    fails_only = False)
            lr = smooth_lick_rate(licks[eprng][mask], 1/fr, sigma_sec=0.7)
            ls_all.append(np.nanmean(ls))
            lr_all.append(np.nanmean(lr))
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
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
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size) 
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm_real = list(combinations(range(len(coms_correct)), 2)) 
        rz_perm_real = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm_real]   
        # account for cells that move to the end/front
        # Define a small window around pi (e.g., epsilon)
        epsilon = .7 # 20 cm
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm_real):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm_real[jj][0]]-coms_rewrel[perm_real[jj][1]]) for jj in range(len(perm_real))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        #only get perms with non zero cells  
        # get both pre and post rew cells at the same time
        cell_types = ['pre', 'post']
        goal_cell_shuf_ps_per_comp_cll=[]
        goal_cells_p_per_comparison_cll=[]
        for cell_type in cell_types:
            if cell_type=='post':
                com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                    xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
            elif cell_type=='pre':
                com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                    xx], axis=0)<=0))] if len(com)>0 else [] for com in com_goal]
            perm_real=[p for ii,p in enumerate(perm_real) if len(com_goal_farrew[ii])>0]
            rz_perm_real=[p for ii,p in enumerate(rz_perm_real) if len(com_goal_farrew[ii])>0]
            com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
            # get goal cells across all epochs        
            if len(com_goal_farrew)>0:
                goal_cells = intersect_arrays(*com_goal_farrew); 
            else:
                goal_cells=[]    
            goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew]
            goal_cells_p_per_comparison_cll.append(goal_cells_p_per_comparison)
            # max of 5 epochs = 10 perms
            goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
            for i in range(num_iterations):
                shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
                [random.shuffle(shuf) for shuf in shufs]
                # first com is as ep 1, others are shuffled cell identities
                com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
                com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
                coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
                # OR shuffle cell identities relative to reward
                perm = list(combinations(range(len(com_shufs)), 2)) 
                rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
                # Define a small window around pi (e.g., epsilon)
                epsilon = .7 # 20 cm
                # Find COMs near pi and shift to -pi
                com_loop_w_in_window = []
                for pi,p in enumerate(perm):
                    for cll in range(coms_rewrel.shape[1]):
                        com1_rel = coms_rewrel[p[0],cll]
                        com2_rel = coms_rewrel[p[1],cll]
                        # print(com1_rel,com2_rel,com_diff)
                        if ((abs(com1_rel - np.pi) < epsilon) and 
                        (abs(com2_rel + np.pi) < epsilon)):
                                com_loop_w_in_window.append(cll)
                # get abs value instead
                coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
                com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
                com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
                #only get perms with non zero cells  
                # get both pre and post rew cells at the same time
                if cell_type=='post':
                    com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                        xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
                elif cell_type=='pre':
                    com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                        xx], axis=0)<=0))] if len(com)>0 else [] for com in com_goal]
                perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
                rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
                com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
                # get goal cells across all epochs        
                if len(com_goal_farrew)>0:
                    goal_cells = intersect_arrays(*com_goal_farrew); 
                else:
                    goal_cells=[]    
                goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew]
                goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
            goal_cell_shuf_ps_per_comp_cll.append(goal_cell_shuf_ps_per_comp)
        try:
            # behavioral metrics
            performance_per_comparison = [np.nanmean([rates[p[0]],rates[p[1]]]) for p in perm_real]
            lick_sel_per_comparison = [np.nanmean([ls_all[p[0]],ls_all[p[1]]]) for p in perm_real]
            lick_r_per_comparison = [np.nanmean([lr_all[p[0]],lr_all[p[1]]]) for p in perm_real]

            vel_per_comparison = [np.nanmean([forwardvel[eps[p[0]:p[0]+1]],forwardvel[eps[p[1]:p[1]+1]]]) for p in perm_real]
            null_per_comparison = [np.nanmean(gc[:,:len(perm_real)],axis=(0)) for gc in goal_cell_shuf_ps_per_comp_cll]
            reward_p_per_comparison = [[np.nanmean([gc[p[0]],gc[p[1]]]) if len(perm_real)>1 else gc[0] for p in perm_real] for gc in goal_cells_p_per_comparison_cll]
            # save perm        
            # cells = pre and post
            datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct,reward_p_per_comparison,performance_per_comparison,lick_sel_per_comparison, vel_per_comparison, null_per_comparison,lick_r_per_comparison, perm, rz_perm]
        except:
            pass

####################################### RUN CODE #######################################
#%%
plt.rc('font', size=20)          # controls default text sizes

pre_reward_cell_p_per_comp = [v[2][0] for k,v in datadct.items()]
post_reward_cell_p_per_comp = [v[2][1] for k,v in datadct.items()]
successrate_p_per_comp = [v[3] for k,v in datadct.items()]
ls_p_per_comp = [v[4] for k,v in datadct.items()]
vel_per_comp = [v[5] for k,v in datadct.items()]
pre_null_reward_cell_p_per_comp = [v[6][0] for k,v in datadct.items()]
post_null_reward_cell_p_per_comp = [v[6][1] for k,v in datadct.items()]
lick_r_per_comp=[v[7] for k,v in datadct.items()]
animal_df = [k.split('_')[0] for k,v in datadct.items()]
#%%
cll_df = [pre_reward_cell_p_per_comp,post_reward_cell_p_per_comp]
cll_null = [pre_null_reward_cell_p_per_comp,post_null_reward_cell_p_per_comp]
lbl=['pre','post']
dfs=[]
for cll in range(len(cll_df)):
    df=pd.DataFrame()
    df['reward_cell_p'] = np.concatenate(cll_df[cll])
    df['null_reward_cell_p'] =  np.concatenate(cll_null[cll])
    df['success_rate'] = np.concatenate(successrate_p_per_comp)
    df['reward_cell_p'] = df['reward_cell_p']*100
    df['success_rate'] =df['success_rate']*100
    df['lick_rate'] = np.concatenate(lick_r_per_comp)
    df['velocity'] = np.concatenate(vel_per_comp)
    df['animal'] = np.concatenate([[an]*len(cll_df[cll][kk]) for kk,an in enumerate(animal_df)])
    df['cell_type'] = [lbl[cll]]*len(df)
    dfs.append(df)
df=pd.concat(dfs)
#%%
# df=df[(df.animal!='e189')&(df.animal!='e190')]
# Assume df is already defined with 'success_rate', 'reward_cell_p', 'animal'
a = 0.4
animals = df['animal'].unique()
n_animals = len(animals)

# Determine square-ish grid size
n_cols = int(np.ceil(np.sqrt(n_animals)))
n_rows = int(np.ceil(n_animals / n_cols))

behs=['success_rate','lick_rate','velocity']
behs_lbl=['% Correct trials','Lick rate','Velocity']
for b,beh in enumerate(behs):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3* n_rows),sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette('Dark2')
    for idx, animal in enumerate(animals):
        ax = axes[idx]
        subdf = df[df['animal'] == animal]
        rs=[];ps=[]
        for i, ct in enumerate(cell_types):
            sub = subdf[subdf['cell_type'] == ct]
            if len(sub) > 1:  # Ensure enough data
                sns.regplot(
                    x=beh,
                    y='reward_cell_p',
                    data=sub,
                    scatter_kws={'color': palette[i], 'alpha': 0.6, 's': 50},
                    line_kws={'color': palette[i]},
                    ax=ax,
                    label=ct
                )
            r, p = scipy.stats.pearsonr(sub[beh], sub['reward_cell_p'])
            rs.append(r),ps.append(p)
        ax.set_title(f'{animal}\npre r={rs[0]:.2f}, p={ps[0]:.3g}\npost r={rs[1]:.2f}, p={ps[1]:.3g}',fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel(behs_lbl[b])
    ax.set_ylabel('% Reward cell')
    # Hide any unused subplots
    for ax in axes[n_animals:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(savedst, f'reward_cell_v_{beh}.svg'), 
            bbox_inches='tight')
#%%
# compare to null
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

test_results = []
xvars = ['success_rate', 'lick_rate', 'velocity']
lbls = ['% Correct trials', 'Lick rate', 'Velocity']
real_r_beh, null_r_beh = [], []

cell_types=['pre','post']
for xvar, label in zip(xvars, lbls):
    real_r_all, null_r_all = [], []
    for i, ct in enumerate(cell_types):
        real_r, null_r = [], []
        for animal in animals:
            sub = df[(df['animal'] == animal)&(df['cell_type'] == ct)]
            # real r
            r_real, _ = scipy.stats.pearsonr(sub[xvar], sub['reward_cell_p'])
            # null r
            r_null, _ = scipy.stats.pearsonr(sub[xvar], sub['null_reward_cell_p'])
            real_r.append(r_real)
            null_r.append(r_null)
        # Paired t-test
        tval, pval = scipy.stats.wilcoxon(real_r, null_r)
        test_results.append((label, ct, pval, real_r, null_r))
        real_r_all.append(real_r)
        null_r_all.append(null_r) # per cell
    real_r_beh.append(real_r_all)
    null_r_beh.append(null_r_all) # per cell

# Bonferroni correction
labels, cll_types, raw_pvals, real_r_all, null_r_all = zip(*test_results)
_, pvals_corr, _, _ = multipletests(raw_pvals, method='fdr_bh')
#%%
s=12
a=0.7
# Build custom palette
custom_palette = {
    'Real_pre': palette[0],
    'Real_post': palette[1],
    'Shuffle_pre': 'gray',
    'Shuffle_post': 'gray'
}

s = 10
a = 0.7
fig, axes = plt.subplots(ncols=3, figsize=(14, 6))

for i, label in enumerate(lbls):
    ax = axes[i]
    bar_data = []

    for j, ct in enumerate(cell_types):  # pre, post
        real_r = real_r_beh[i][j]
        null_r = null_r_beh[i][j]
        if label == '% Correct trials':
            real_r = [r * 100 for r in real_r]
            null_r = [r * 100 for r in null_r]

        bar_data.append(pd.DataFrame({
            'animal': animals,
            'r_value': real_r,
            'type': 'Real',
            'cell_type': ct,
            'group': f'Real_{ct}'
        }))
        bar_data.append(pd.DataFrame({
            'animal': animals,
            'r_value': null_r,
            'type': 'Shuffle',
            'cell_type': ct,
            'group': f'Shuffle_{ct}'
        }))

    plot_df = pd.concat(bar_data)

    # Bar plot
    sns.barplot(data=plot_df, x='cell_type', y='r_value', hue='group',
                palette=custom_palette, errorbar='se', fill=False, ax=ax,legend=False)

    # Strip plot
    sns.stripplot(data=plot_df, x='cell_type', y='r_value', hue='group',
                  palette=custom_palette, alpha=a, jitter=True, dodge=True, s=s, ax=ax)

    # Draw lines between Real and Shuffle per animal per cell_type
    for ll, ct in enumerate(cell_types):
        df_real = plot_df[(plot_df['cell_type'] == ct) & (plot_df['type'] == 'Real')]
        df_null = plot_df[(plot_df['cell_type'] == ct) & (plot_df['type'] == 'Shuffle')]
        for animal in animals:
            real_val = df_real[df_real['animal'] == animal]['r_value'].values
            null_val = df_null[df_null['animal'] == animal]['r_value'].values
            if len(real_val) and len(null_val):
                ax.plot([ll -.3+ll*0.4, ll-.1+(0.4*ll)], [real_val[0], null_val[0]], color='gray', alpha=0.5, zorder=1)

        # Add p-value
        pval_idx = i * 2 + ll
        pval = pvals_corr[pval_idx]
        y_max = max(df_real['r_value'].max(), df_null['r_value'].max())
        ax.text(ll, y_max + 0.02 * abs(y_max), f'p={pval:.2g}', ha='center', va='bottom', fontsize=11)

    ax.set_title(label)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('Cell Type')
    ax.set_xticklabels(['Pre', 'Post'])
    ax.legend_.remove()
    sns.despine()
# Show legend only once (after last plot)
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(handles, labels_, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, title='Correlation Type')

plt.tight_layout()
plt.savefig(os.path.join(savedst, f'pre_post_real_vs_null_r.svg'), bbox_inches='tight')
plt.show()
