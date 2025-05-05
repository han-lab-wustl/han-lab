
"""
zahra
get tuning curves with dark time
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
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'dark_time_tuning.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)


def get_radian_position_first_lick_after_rew_w_dt(i, eps, ybinned, licks, reward, rewsize,rewlocs,
                    trialnum):
    """
    Computes radian position aligned to the first lick after reward.
    Parameters:
    - i = epoch
    - eps: List of trial start indices.
    - ybinned: 1D array of position values.
    - licks: 1D binary array (same length as ybinned) indicating lick events.
    - reward: 1D binary array (same length as ybinned) indicating reward delivery.
    - track_length: Total length of the circular track.

    Returns:
    - rad: 1D array of radian positions aligned to the first lick after reward.
    """
    rad = []  # Store radian coordinates
    # for i in range(len(eps) - 1):
    # Extract data for the current trial
    y_trial = ybinned#[eps[i]:eps[i+1]]
    licks_trial = licks#[eps[i]:eps[i+1]]
    reward_trial = reward#[eps[i]:eps[i+1]]
    trialnum_trial = trialnum#[eps[i]:eps[i+1]]
    unique_trials = np.unique(trialnum)  # Get unique trial numbers [eps[i]:eps[i+1]]
    for tr,trial in enumerate(unique_trials):
        # Extract data for the current trial
        trial_mask = trialnum_trial == trial  # Boolean mask for the current trial
        y = y_trial[trial_mask]
        licks_trial_ = licks_trial[trial_mask]
        reward_trial_ = reward_trial[trial_mask]
        # Find the reward location in this trial
        reward_indices = np.where(reward_trial_ > 0)[0]  # Indices where reward occurs
        if len(reward_indices) == 0:
            try:
                # 1.5 bc doesn't work otherwise?
                y_rew = np.where((y<(rewlocs[tr][i]+rewsize*.5)) & (y>(rewlocs[i][tr]-rewsize*.5)))[0][0]
                reward_idx=y_rew
            except Exception as e: # if trial is empty??
                reward_idx=int(len(y)/2) # put in random middle place of trials
        else:
            reward_idx = reward_indices[0]  # First occurrence of reward
        # Find the first lick after the reward
        lick_indices_after_reward = np.where((licks_trial_ > 0) & (np.arange(len(licks_trial_)) > reward_idx))[0]
        if len(lick_indices_after_reward) > 0:
            first_lick_idx = lick_indices_after_reward[0]  # First lick after reward
        else:
            # if animal did not lick after reward/no reward was given
            first_lick_idx=reward_idx
        # Convert positions to radians relative to the first lick
        first_lick_pos = y[first_lick_idx]
        track_length = np.max(y) # custom max for each trial w dark time
        rad.append((((((y - first_lick_pos) * 2 * np.pi) / track_length) + np.pi) % (2 * np.pi)) - np.pi)

    if len(rad) > 0:
        rad = np.concatenate(rad)
        return rad
    else:
        return np.array([])  # Return empty array if no valid trials

def make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,bin_size,
            lasttr=8,bins=90,track_length=270,
            velocity_filter=False):    
    rates = []; tcs_fail = []; tcs_correct = []; coms_correct = []; coms_fail = []        
    rewlocs_w_dt = []; ybinned_dt = []
    failed_trialnm = []; 
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        rewloc = rewlocs[ep]
        ypos_ep = ybinned[eprng]
        vel_ep = forwardvel[eprng]
        trial_ep = trialnum[eprng]        
        reward_ep = rewards[eprng]
        lick_ep = licks[eprng]
        # Get dark time frames and estimate distance
        ypos_dt = []
        rewloc_per_trial = []
        rewloc_bool = []
        # get the dark time and add it to the beginning of the trial
        for trial in np.unique(trial_ep):
            trial_mask = trial_ep==trial
            # constant set to dt
            ypos_num = ypos_ep[trial_mask][5]
            ypos_trial = ypos_ep[trial_mask]
            # remove random end of track value            
            ypos_trial[:5] = ypos_num
            dark_mask = ypos_trial == ypos_num
            dark_vel = vel_ep[trial_mask][dark_mask]
            dark_frames = np.sum(dark_mask)
            dark_dt = time[eprng][trial_mask][dark_mask] 
            dark_distance = np.nanmean(dark_vel) * dark_dt
            dark_distance = dark_distance-dark_distance[0]            
            # find start of rew loc index
            rewloc = (ypos_trial >= rewlocs[ep]-rewsize/2-3) & (ypos_trial <= rewlocs[ep]+rewsize/2+3)
            rewloc = consecutive_stretch(np.where(rewloc)[0])[0]
            rewloc = min(rewloc)
            from scipy.ndimage import gaussian_filter1d
            dt_ind = np.where(ypos_trial==ypos_num)[0]
            ypos_trial_new = ypos_trial.copy()
            ypos_trial_new[ypos_trial_new==ypos_num] = dark_distance
            ypos_trial_new[ypos_trial>ypos_num] = ypos_trial_new[ypos_trial>ypos_num]+dark_distance[-1]
            ypos_dt.append(ypos_trial_new)
            # get new rewloc 
            rewloc_per_trial.append(ypos_trial_new[rewloc])
            rl_bool = np.zeros_like(ypos_trial_new)
            rl_bool[rewloc]=1
            rewloc_bool.append(rl_bool)
        
        # nan pad position
        ypos_w_dt = np.concatenate(ypos_dt)
        ybinned_dt.append(ypos_w_dt)
        # realign to reward????        
        rewloc_bool = np.concatenate(rewloc_bool)
        # test
        # plt.plot(ypos_w_dt)
        # plt.plot(rewloc_bool*400)
        relpos = get_radian_position_first_lick_after_rew_w_dt(ep, eps, ypos_w_dt, lick_ep, 
                reward_ep, rewsize, rewloc_per_trial,
                trial_ep)
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        # in between failed trials only!!!!! 4/2025
        if len(strials)>0:
            failed_inbtw = np.array([int(xx)-strials[0] for xx in ftrials])
            failed_inbtw=np.array(ftrials)[failed_inbtw>0]
        else: # for cases where an epoch was started but not enough trials
            failed_inbtw=np.array(ftrials)
        failed_trialnm.append(failed_inbtw)
        # trials going into tuning curve
        print(f'Failed trials in failed tuning curve\n{failed_inbtw}\n')

        F_all = Fc3[eprng,:]            
        # simpler metric to get moving time
        if velocity_filter==True:
            moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        else:
            moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F_all = F_all[moving_middle,:]
        relpos_all = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            # last 8 correct trials
            if len(strials)>0:
                mask = [True if xx in strials[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_correct.append(tc)
                coms_correct.append(com)
            # failed trials                        
            # UPDATE 4/16/25
            # only take last 8 failed trials?
            if len(failed_inbtw)>0:
                mask = [True if xx in failed_inbtw[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                # print(f'Fluorescence array size:\n{F.shape}\n')
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_fail.append(tc)
                coms_fail.append(com)
        rewlocs_w_dt.append(rewloc_per_trial)
    tcs_correct = np.array(tcs_correct); coms_correct = np.array(coms_correct)  
    tcs_fail = np.array(tcs_fail); coms_fail = np.array(coms_fail)  
    
    return tcs_correct, coms_correct, tcs_fail, coms_fail, rewlocs_w_dt, ybinned_dt

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
p_goal_cells=[]
p_goal_cells_dt = []
bins = 90
goal_window_cm=20
# cm_window = [10,20,30,40,50,60,70,80] # cm
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
        rates = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
        rate=np.nanmean(np.array(rates))
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
        tcs_correct_dt, coms_correct_dt, tcs_fail, coms_fail, rewloc_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,bin_size,bins=120)
        # normal tc
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          

        #test
        # bins=120
        # fig,axes = plt.subplots(ncols=len(tcs_correct))
        # for ep in range(len(tcs_correct)):
        #     ax=axes[ep]
        #     min_rewloc = np.quantile(np.array(rewloc_dt[ep]),.1)
        #     max_rewloc = np.quantile(np.array(rewloc_dt[ep]),.9)
        #     av_ypos = np.nanmax(ybinned_dt[ep])
        #     binsize_dt = av_ypos/bins
        #     min_rewloc_tc = min_rewloc/binsize_dt
        #     max_rewloc_tc = max_rewloc/binsize_dt
        #     ax.imshow(tcs_correct[ep][np.argsort(coms_correct[0])]**.5)
        #     ax.axvline(bins/2, color='w',linestyle='--')
        # # ax.axvline(max_rewloc_tc, color='w')
        # fig,ax = plt.subplots()
        # ax.plot(ybinned_dt[ep])
        
        ################################# NOT TESTED BELOW ######################################
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew = get_goal_cells(goal_window, coms_correct, type = 'all')
        goal_cells_dt, com_goal_postrew_dt = get_goal_cells(goal_window, coms_correct_dt, type = 'all')
        #only get perms with non zero cells
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
        rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
        
        p_goal_cells.append(len(goal_cells)/len(coms_correct[0]))
        p_goal_cells_dt.append(len(goal_cells_dt)/len(coms_correct_dt[0]))
        print(f'Goal cells w/o dt: {goal_cells}\n\
            Goal cells w/ dt: {goal_cells_dt}')

# Assembly activity is a time series that measures how active a particular 
# neuronal ensemble (identified via PCA) is at each time point. It reflects 
# coordinated activity, not just individual spikes.
pdf.close()
#%%
# look through all the assemblies
df = conddf.copy()
df = df[(df.animals!='e217') & (df.optoep.values<2)]
df['p_cells_in_assemblies'] = p_rewcells_in_assemblies
df['p_cells_in_assemblies'] = df['p_cells_in_assemblies'] *100
ax = sns.histplot(x = 'p_cells_in_assemblies',data=df)

#%%
from projects.pyr_reward.rewardcell import cosine_sim_ignore_nan
from matplotlib import colors

# look through all the assemblies
df = conddf.copy()
df = df[(df.animals!='e217') & (df.optoep.values<2)]
an_plt = 'z9' # 1 eg animal
an_day = 19
cs_all = []; num_epochs = []
plt.close('all')
plot = False
for ii,ass in enumerate(assembly_cells_all_an):
    # if df.iloc[ii].animals==an_plt and df.iloc[ii].days==an_day:
        print(f'{df.iloc[ii].animals}, {df.iloc[ii].days}')
        ass_all = list(ass.values()) # all assemblies
        cs_per_ep = []; ne = []
        for jj,asm in enumerate(ass_all):
            perm = list(combinations(range(len(asm)), 2)) 
            # consecutive ep only
            perm = [p for p in perm if p[0]-p[1]==-1]
            cs = [cosine_sim_ignore_nan(asm[p[0]], asm[p[1]]) for p in perm]
            cs_per_ep.append(cs)
            if plot:
                fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                gamma=.5
                for kk,tcs in enumerate(asm):
                    ax = axes[kk]
                    vmin = np.min(tcs)
                    vmax = np.max(tcs)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                    im=ax.imshow(tcs[np.argsort(com_per_cell)]**gamma,aspect='auto',norm=norm)
                    ax.set_title(f'Epoch {kk+1}')
                    ax.axvline(bins/2, color='w', linestyle='--')
                ax.set_xticks(np.arange(0,bins,30))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2))
                fig.suptitle(f'Pre-reward ensemble \n {df.iloc[ii].animals}, {df.iloc[ii].days} \n\
                    Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}')
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')
                if jj==0:
                    plt.savefig(os.path.join(savedst,f'{an_plt}_{an_day}_prerew_ensemble_eg.svg'),bbox_inches='tight')
        num_epochs.append(len(asm))
        cs_all.append(cs_per_ep)
            # plt.figure()
            # plt.plot(tcs[np.argsort(com_per_cell)].T)
# %%# %%
# add 2 ep combinaitions as 2 ep
df2 = pd.DataFrame()
df2['cosine_sim_across_ep'] = np.hstack([np.concatenate(xx) if len(xx)>0 else np.nan for xx in cs_all])
df2['animals'] = np.concatenate([[df.iloc[ii].animals]*len(np.concatenate(xx)) if len(xx)>0 else [df.iloc[ii].animals] for ii,xx in enumerate(cs_all)])
df2['num_epochs'] =[2]*len(df2)

df['num_epochs'] = num_epochs
# df['cosine_sim_across_ep'] = [np.quantile(xx,.75) if len(xx)>0 else np.nan for xx in cs_all]
df['cosine_sim_across_ep'] = [np.mean(xx) if len(xx)>0 else np.nan for xx in cs_all]
df['cosine_sim_across_ep'] = [np.nanmin(xx) if len(xx)>0 else np.nan for xx in cs_all]
df = pd.concat([df,df2])
df = df.dropna(subset=['cosine_sim_across_ep', 'num_epochs'])
dfan = df.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
dfan = dfan.reset_index()
dfan = dfan[dfan.num_epochs<5]
df_clean = dfan
# temp
df_clean = df_clean[(df_clean.animals!='e139') & (df_clean.animals!='e200') & (df_clean.animals!='e190')]
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


# Pairwise comparisons (Bonferroni)
unique_groups = sorted(df_clean['num_epochs'].unique())
group_data = {group: df_clean[df_clean['num_epochs'] == group]['cosine_sim_across_ep'] for group in unique_groups}

comparisons = list(combinations(unique_groups, 2))
raw_pvals = []
for g1, g2 in comparisons:
    _, pval = scipy.stats.wilcoxon(group_data[g1], group_data[g2])
    raw_pvals.append(pval)

# Bonferroni correction
reject, corrected_pvals, _, _ = multipletests(raw_pvals, method='bonferroni')
# Plot
s=10
plt.figure(figsize=(3,5))
ax = sns.barplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, errorbar='se',
            fill=False, color='k')
sns.stripplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, color='k', jitter=True,
            s=s,alpha=0.7)
# Annotate
fs = 30
pshift = 0.05
max_y = df_clean['cosine_sim_across_ep'].max()

for i, ((g1, g2), pval, rej) in enumerate(zip(comparisons, corrected_pvals, reject)):
    x1, x2 = int(g1)-2, int(g2)-2
    y = max_y + 0.05 * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y+0.01, y+0.01, y], lw=1.5, c='k')

    if pval < 0.001:
        star = '***'
    elif pval < 0.01:
        star = '**'
    elif pval < 0.05:
        star = '*'
    else:
        star = 'ns'

    ax.text((x1 + x2) / 2, y + 0.015, star, ha='center', fontsize=fs)
    ax.text((x1 + x2) / 2, y + 0.015 + pshift, f'p={pval:.2g}', ha='center', rotation=45, fontsize=12)

ax.set_title('Pre-reward ensembles', pad=50)
plt.tight_layout()
plt.show()
#%%
plt.rc('font', size=20)
# compare to post rew
df_post = pd.read_csv(r'Z:\condition_df\postrew_ensemble.csv')
df_nonrew = pd.read_csv(r'Z:\condition_df\place_ensemble.csv')
df_shuffle = pd.read_csv(r'Z:\condition_df\shuffle_ensemble.csv')
df_post['cell_type'] = ['Post-reward']*len(df_post)
df_nonrew['cell_type'] = ['Place']*len(df_nonrew)
df_shuffle['cell_type'] = ['Shuffle']*len(df_nonrew)
df_pre = df_clean
df_pre['cell_type'] = ['Pre-reward']*len(df_pre)
# palette = seaborn Dark2
s=10
df_all = pd.concat([df_pre, df_post, df_nonrew])
order = ['Place', 'Pre-reward', 'Post-reward']
plt.figure(figsize=(6,4))
ax = sns.barplot(x='num_epochs', y='cosine_sim_across_ep', hue='cell_type',data=df_all, errorbar='se',
            fill=False, palette = 'Dark2')
sns.barplot(x='num_epochs', y='cosine_sim_across_ep',data=df_shuffle, errorbar='se',
            color='dimgrey',alpha=0.3,
            label='shuffle', err_kws={'color': 'grey'},ax=ax)
ax = sns.stripplot(x='num_epochs', y='cosine_sim_across_ep', hue='cell_type',data=df_all, dodge=True,
            s=s,alpha=0.7,palette = 'Dark2')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0, 0.5), title='Cell Type')

# make lines
df_all = df_all.reset_index()
ax.spines[['top','right']].set_visible(False)

# Plot individual lines per animal with x-axis offset
offset = {'Pre-reward': -0.2, 'Post-reward': 0.2}
for animal in df_all.animals.unique():
    for cell_type in ['Pre-reward', 'Post-reward']:
        df_sub = df_all[(df_all.animals == animal) & (df_all.cell_type == cell_type)]
        if df_sub.empty:
            continue
        x_vals = df_sub.num_epochs + offset[cell_type]
        ax.plot(x_vals-2, df_sub.cosine_sim_across_ep, color=sns.color_palette('Dark2')[['Pre-reward', 'Post-reward'].index(cell_type)],
                alpha=0.3, linewidth=2)
# Get unique epochs
epochs = sorted(df_all.num_epochs.unique())
ymax = .6
y_offsets = [ymax + (i * 0.03) for i in range(len(epochs))]

fs = 40  # font size for stars
pshift = 0.08  # p-value label offset

# non rew vs. pre reward
# for i, epoch in enumerate(epochs):
#     data_epoch = df_all[df_all.num_epochs == epoch]
#     pre_vals = data_epoch[data_epoch.cell_type == 'Pre-reward']['cosine_sim_across_ep'].dropna()
#     post_vals = data_epoch[data_epoch.cell_type == 'Place']['cosine_sim_across_ep'].dropna()

#     # t-test
#     stat, pval = scipy.stats.ranksums(pre_vals, post_vals)
#     # Plot annotation
#     x = i
#     y = y_offsets[i]
#     # Show p-value (optional)
#     ax.text(x, y + pshift, f'place vs. pre p={pval:.2g}', ha='center', fontsize=12, rotation=45)

for i, epoch in enumerate(epochs):
    data_epoch = df_all[df_all.num_epochs == epoch]
    pre_vals = data_epoch[data_epoch.cell_type == 'Pre-reward']['cosine_sim_across_ep'].dropna()
    post_vals = data_epoch[data_epoch.cell_type == 'Post-reward']['cosine_sim_across_ep'].dropna()
    # t-test
    stat, pval = scipy.stats.ranksums(pre_vals, post_vals)

    # Plot annotation
    x = i
    y = y_offsets[i]
    if pval < 0.001:
        ax.text(x, y, "***", ha='center', fontsize=fs)
    elif pval < 0.01:
        ax.text(x, y, "**", ha='center', fontsize=fs)
    elif pval < 0.05:
        ax.text(x, y, "*", ha='center', fontsize=fs)

    # Show p-value (optional)
    ax.text(x, y + pshift, f'post v pre\np={pval:.2g}', ha='center', fontsize=12, rotation=45)
    
    # pre-reward comp
for i, ((g1, g2), pval, rej) in enumerate(zip(comparisons, corrected_pvals, reject)):
    x1, x2 = int(g1)-2, int(g2)-2
    y = max_y + 0.05 * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y+0.01, y+0.01, y], lw=1.5, c='k')

    if pval < 0.001:
        star = '***'
    elif pval < 0.01:
        star = '**'
    elif pval < 0.05:
        star = '*'
    else: star=''

    ax.text((x1 + x2) / 2, y + 0.015, star, ha='center', fontsize=fs)
    ax.text((x1 + x2) / 2, y + 0.015 + pshift, f'p={pval:.2g}', ha='center', rotation=45, fontsize=12)

ax.set_ylabel('Mean ensemble cosine similarity')
ax.set_xlabel('# of reward loc. switches')

plt.savefig(os.path.join(savedst, 'ensemble_cosine_sim_pre_v_post.svg'))
#%%
# histogram of cell % in assemblies
fig, axes = plt.subplots(ncols=2,figsize=(10,5))
ax=axes[0]
sns.histplot(
    x='p_cells_in_assemblies',
    hue='cell_type',
    data=df_all,
    bins=5,
    palette='Dark2',
    multiple='dodge',  # This avoids overlapping
ax=ax)
ax.set_xlabel('Dedicated cell % in ensemble')
ax.set_ylabel('Sessions')
ax.spines[['top','right']].set_visible(False)
ax=axes[1]
sns.boxplot(
    x='cell_type',
    y='p_cells_in_assemblies',
    hue='cell_type',
    data=df_all,    
    palette='Dark2',    
ax=ax)
ax.set_ylabel('Dedicated cell % in ensemble')
# ax.set_ylabel('Sessions')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'pcells_in_ensembles.svg'))
