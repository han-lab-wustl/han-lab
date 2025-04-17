#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from behavior import get_success_failure_trials,\
get_performance, get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior_chrimson_onlyz14.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing

bin_size = 3
# com shift analysis
dcts = []
for dd,day in enumerate(conddf.days.values):
    dct = {}
    animal = conddf.animals.values[dd]
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['VR'])
    VR = fall['VR'][0][0][()]
    eps = np.where(np.hstack(VR['changeRewLoc']>0))[0]
    eps = np.append(eps, len(np.hstack(VR['changeRewLoc'])))
    scalingf = VR['scalingFACTOR'][0][0]
    track_length = 180/scalingf
    nbins = track_length/bin_size
    ybinned = np.hstack(VR['ypos']/scalingf)
    rewlocs = np.ceil(np.hstack(VR['changeRewLoc'])[np.hstack(VR['changeRewLoc']>0)]/scalingf).astype(int)
    rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
    trialnum = np.hstack(VR['trialNum'])
    rewards = np.hstack(VR['reward'])
    forwardvel = np.hstack(VR['ROE']); time =np.hstack(VR['time'])
    forwardvel=-0.013*forwardvel[1:]/np.diff(time) # make same size
    forwardvel = np.append(forwardvel, np.interp(len(forwardvel)+1, np.arange(len(forwardvel)),forwardvel))
    licks = np.hstack(VR['lickVoltage'])
    licks = licks<=-0.065 # remake boolean
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)   
        if len(eps)<4: eptest = 2 # if no 3 epochs    
    rates_opto, rates_prev, lick_prob_opto, \
    lick_prob_prev, trials_bwn_success_opto, \
    trials_bwn_success_prev, vel_opto, vel_prev, lick_selectivity_per_trial_opto,\
    lick_selectivity_per_trial_prev, lick_rate_opto, lick_rate_prev, com_opto, com_prev = get_performance(eptest, eps, trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize)
    rewzones = get_rewzones(rewlocs, 1.5)
    
    dct['velocity'] = [vel_prev, vel_opto]
    dct['lick_selectivity'] = [lick_selectivity_per_trial_prev, lick_selectivity_per_trial_opto]
    dct['com'] = [com_prev, com_opto]
    dct['rates'] = [rates_prev, rates_opto]    
    dct['lick_p'] = [lick_prob_prev, lick_prob_opto]
    dct['rewlocs'] = [rewlocs[eptest-2], rewlocs[eptest-1]]
    dct['rewzones'] = [rewzones[eptest-2], rewzones[eptest-1]]
    dct['trials_before_success'] = [trials_bwn_success_prev, trials_bwn_success_opto]
    dcts.append(dct)
#%%
# plot performance 
s = 12 # pontsize
dcts_opto = np.array(dcts)
df=pd.DataFrame()
df['rates'] = np.concatenate([dct['rates'] for dct in dcts])
# com opto
df['com'] = np.concatenate([dct['com'] for dct in dcts])
df['lick_selectivity']=np.concatenate([[np.nanmean(dct['lick_selectivity'][0]),np.nanmean(dct['lick_selectivity'][1])] for dct in dcts])
df['opto'] = np.repeat(conddf.optoep.values>1,2)
df['condition'] = np.repeat(['vip' if xx=='vip_ex' else 'ctrl' for xx in conddf.in_type.values],2)
df['epoch']= np.concatenate([['previous_epoch', 'stim_epoch']*len(dcts)])
df['animals'] = np.repeat(conddf.animals.values,2)
df['optoep'] = np.repeat(conddf.optoep.values,2)
# plot rates vip vs. ctl led off and on
# df = df[(df.animals!='e189')&(df.animals!='z9')]
# df=df[(df.optoep.values>1)]
bigdf_plot = df.groupby(['animals', 'epoch', 'opto', 'condition']).mean(numeric_only=True)
bigdf_plot = df # do not sum by animals
fig,ax = plt.subplots(figsize=(3,5))
sns.barplot(x="epoch", y="rates",hue='opto', data=bigdf_plot,
    # palette={'ctrl': "slategray", 'vip': "darkgoldenrod"},                
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="epoch", y="rates",hue='opto', data=bigdf_plot,
            # palette={'ctrl': 'slategray','vip': "darkgoldenrod"},                
            s=s,ax=ax,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Performance')
ax.set_xticks([0,1], labels=['LEDoff', 'LEDon'])
ax.set_xlabel('')
x1 = df.loc[((df.condition=='vip')&(df.epoch=='previous_epoch')), 'rates'].values
x2 = df.loc[((df.condition=='vip')&(df.epoch=='stim_epoch')), 'rates'].values
# x2 = df.loc[((df.condition=='ctrl')&(df.opto==True)), 'rates'].values
t,pval = scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# statistical annotation    
fs=46
ii=1.5; y=.8; pshift=.07
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)
#%%
from statsmodels.stats import power as smp
# Step 1: Calculate the means and standard deviations
# hypothetical b/c no behavior diff yet
# suppose rz of 5cm
group1=x1; group2=x2
mean1 = .9#np.mean(group1)
mean2 = .7#np.mean(group2)
std1 = .15#np.std(group1, ddof=1)
std2 = .16#np.std(group2, ddof=1)
# Step 2: Calculate pooled standard deviation
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
# Step 3: Calculate Cohen's d
cohens_d = (mean1 - mean2) / pooled_std
# Step 4: Perform Power Analysis using the calculated Cohen's d
alpha = 0.05  # Significance level
power = 0.8   # Desired power
analysis = smp.TTestIndPower()
sample_size = analysis.solve_power(effect_size=cohens_d, alpha=alpha, power=power, alternative='two-sided')
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Required sample size per group: {sample_size:.2f}")
# plt.savefig(os.path.join(savedst, 'behavior.svg'),  bbox_inches='tight')
#%%
# plot lick selectivity and lick com
s=14
# bigdf_plot = df.groupby(['animals', 'condition', 'opto']).median(numeric_only=True)
fig,ax = plt.subplots(figsize=(2,5))
bigdf_plot = bigdf_plot[bigdf_plot.epoch=='stim_epoch'] # led on only
bigdf_plot = bigdf_plot.sort_values(by='condition')
sns.barplot(x="condition", y="lick_selectivity",hue='condition', data=bigdf_plot,
    palette={'ctrl': "slategray", 'vip': "darkgoldenrod"},                
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="lick_selectivity",hue='condition', data=bigdf_plot,
            palette={'ctrl': 'slategray','vip': "darkgoldenrod"},                
            s=s,ax=ax,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Lick Selectivity, last 5 trials (LEDon)')
ax.set_xticks([0,1], labels=['Control', 'VIP\nExcitation'])
ax.set_xlabel('')
# bigdf_plot=bigdf_plot.reset_index()
x1 = bigdf_plot.loc[((bigdf_plot.condition=='vip')&(bigdf_plot.opto==True)), 'lick_selectivity'].values
x2 = bigdf_plot.loc[((bigdf_plot.condition=='ctrl')&(bigdf_plot.opto==True)), 'lick_selectivity'].values
t,pval = scipy.stats.ranksums(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# statistical annotation    
fs=46
ii=0.5; y=1; pshift=.2
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)
# plt.savefig(os.path.join(savedst, 'lick_selectivity.svg'),  bbox_inches='tight')
# bigdf_plot = df.groupby(['animals', 'condition', 'opto']).median(numeric_only=True)
#%%
# lick com by rewzone
df = conddf
df['rates_diff'] = [np.diff(dct['rates'])[0] for dct in dcts]
df['velocity_diff'] = [np.diff(dct['velocity'])[0] for dct in dcts]
df['velocity'] = [dct['velocity'][0] for dct in dcts]
# com opto
df['com'] = [dct['com'][1] for dct in dcts]
df['lick_selectivity']=[np.nanmean(dct['lick_selectivity'][1]) for dct in dcts]
df['opto'] = conddf.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in conddf.in_type.values]
df['rewzone_transition'] = [f'reward_zone {(tuple([int(xx) for xx in dct["rewzones"]]))}' for dct in dcts]
df=df[(df.optoep.values>1)]
df=df[df.animals!='e190']
bigdf_plot = df.groupby(['animals', 'condition', 'opto', 'rewzone_transition']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(6,6))
sns.barplot(x="condition", y="com",hue='rewzone_transition', data=bigdf_plot,              
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="com",hue='rewzone_transition', data=bigdf_plot,                
            s=s,ax=ax,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Center of Mass Licks-Reward Loc. (cm)')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xlabel('')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside
pvals = []
for rz in df.rewzone_transition.unique():
    try: 
        x1 = bigdf_plot.loc[((bigdf_plot.index.get_level_values('condition')=='vip')&(bigdf_plot.index.get_level_values('opto')==True)&(bigdf_plot.index.get_level_values('rewzone_transition')==rz)), 'com'].values
        x2 = bigdf_plot.loc[((bigdf_plot.index.get_level_values('condition')=='ctrl')&(bigdf_plot.index.get_level_values('opto')==True)&(bigdf_plot.index.get_level_values('rewzone_transition')==rz)), 'com'].values
        t,pval = scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
        pvals.append([rz,pval])
    except Exception as e:
        print(e)
plt.savefig(os.path.join(savedst, 'com.svg'),  bbox_inches='tight')

# only 3 to 1
bigdf_plot = bigdf_plot.reset_index()
bigdf_plot=bigdf_plot[bigdf_plot.animals!='e190']

df2plt = bigdf_plot[bigdf_plot.rewzone_transition=='reward_zone (3, 1)']
fig,ax = plt.subplots(figsize=(2,6))
sns.barplot(x="condition", y="com",hue='condition', data=df2plt, 
            palette={'ctrl': "slategray", 'vip': "red"},                     
            errorbar='se', fill=False,ax=ax)
sns.stripplot(x="condition", y="com",hue='condition', data=df2plt,                
              palette={'ctrl': "slategray", 'vip': "red"},        
            s=s,ax=ax,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_ylabel(f'Center of Mass Licks-Reward Loc. (cm) \n(Far to Near)')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'])
ax.set_xlabel('')
# fig.tight_layout()
x1 = bigdf_plot.loc[((bigdf_plot.condition=='vip')&(bigdf_plot.opto==True)&(bigdf_plot.rewzone_transition=='reward_zone (3, 1)')), 'com'].values
x2 = bigdf_plot.loc[((bigdf_plot.condition=='ctrl')&(bigdf_plot.opto==True)&(bigdf_plot.rewzone_transition=='reward_zone (3, 1)')), 'com'].values
t,pval = scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])
# statistical annotation    
fs=46
ii=0.5; y=15; pshift=.2
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

plt.savefig(os.path.join(savedst, 'far2near_com.svg'),  bbox_inches='tight')

#%%
# velocity
plt.figure()
bigdf_plot = df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)
plt.figure(figsize=(3.5,6))
ax = sns.barplot(x="opto", y="velocity",hue='condition', data=bigdf_plot,
    palette={'ctrl': "slategray", 'vip': "red"},                
            errorbar='se', fill=False)
sns.stripplot(x="opto", y="velocity",hue='condition', data=bigdf_plot,
            palette={'ctrl': 'slategray','vip': "red"},                
            s=10)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)

# x1 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type == 'vip_ledon'), 'velocity_diff'].values
# x2 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type == 'vip_ledoff'), 'velocity_diff'].values
# x3 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type=='ctrl_ledon'), 'velocity_diff'].values
# x4 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type=='ctrl_ledoff'), 'velocity_diff'].values
# scipy.stats.f_oneway(x1[~np.isnan(x1)], x2, x3, x4[~np.isnan(x4)])
# p_values= sp.posthoc_ttest([x1,x2,x3,x4])#,p_adjust='holm-sidak')
# print(p_values)

#%%
# plot rates by rewzone

bigdf_plot = bigdf.groupby(['animal', 'rewzones_transition', 'vip_ctrl_type']).mean(numeric_only=True)

# plot by rew zone transition
bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_transition')
plt.figure()
ax = sns.barplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"},
                errorbar='se', fill=False)
sns.stripplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(savedst, 'rewzone_transition.svg'), bbox_inches='tight')

# controls
bigdf_plot = bigdf[~bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_opto')
plt.figure()
ax = sns.barplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"},
                errorbar='se', fill=False)
sns.stripplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# plot by rew zone transition
bigdf_plot = bigdf[~bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_transition')
plt.figure()
ax = sns.barplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"},
                errorbar='se', fill=False)
sns.stripplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


#%%
# tests rewz 1 vs. 3
bigdf_test = bigdf
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_3'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_3'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2)

# tests 3-1 transition
bigdf_test = bigdf
x1 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledoff') & (bigdf_test.rewzones_transition=='rz_3-1'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_3-1'), 'rates_diff'].values
scipy.stats.ranksums(x1[~np.isnan(x1)], x2)

# tests 1-3 transition
bigdf_test = bigdf
x1 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledoff') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
scipy.stats.ranksums(x1[~np.isnan(x1)], x2)

# tests 3-1 transition in controls
bigdf_test = bigdf
x1 = bigdf_test.loc[~bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl') & bigdf.in_type.str.contains('ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
x2 = bigdf_test.loc[~bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl') & bigdf.in_type.str.contains('ledoff') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2[~np.isnan(x2)])

#%%
# plot lick tuning 
# first, just take average
dcts_opto = np.array(dcts)
dfs = []
for ii,dct in enumerate(dcts_opto):
    lick_p_prev, lick_p_opto = dct['lick_p']
    pre_prev, rew_prev, post_prev = lick_p_prev
    pre_o, rew_o, post_o = lick_p_opto
    dists_prev = [pre_prev.values, rew_prev.values, \
        post_prev.values]
    dists_o = [pre_o.values, rew_o.values, \
        post_o.values]
    df = pd.DataFrame()
    lick_all_cond_prev = [np.percentile(pre_prev.values,75), np.percentile(rew_prev.values,75), \
        np.percentile(post_prev.values,75)]
    lick_all_cond_opto = [np.percentile(pre_o.values,75), np.percentile(rew_o.values,75), \
        np.percentile(post_o.values,75)]    
    df['lick_prob_prev'] = np.array(lick_all_cond_prev)
    df['lick_prob_opto'] = np.array(lick_all_cond_opto)
    df['lick_probability_difference_condition'] = df['lick_prob_opto']-df['lick_prob_prev']
    df['lick_probability_difference'] = np.nanmean(np.array(lick_all_cond_opto))-np.nanmean(np.array(lick_all_cond_prev))
    # df['lick_dist'] = [scipy.spatial.distance.jensenshannon(xx,dists_o[ii]) for ii,xx in enumerate(dists_prev)]
    df['lick_condition'] = ['pre_reward', 'reward', 'post_reward']
    df['rewzones_transition'] = [f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}_pre',
                                f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}_rew',
                                f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}_post']
    df['rewzones_opto'] = [f'rz_{dct["rewzones"][1].astype(int)}_pre',
                        f'rz_{dct["rewzones"][1].astype(int)}_rew',
                        f'rz_{dct["rewzones"][1].astype(int)}_post']
    df['animal'] = [conddf.animals.values[ii]]*3   
    if conddf.optoep.values[ii]>1:    
        df['in_type'] = [f'{conddf.in_type.values[ii]}_ledon']*3
        df['led'] = [True]*3
        df['vip_ctrl_type'] = df['in_type']
        if not conddf.in_type.values[ii]=="vip":
            df['vip_ctrl_type'] = 'ctrl_ledon'
    else: 
        df['in_type'] = [f'{conddf.in_type.values[ii]}_ledoff']*3
        df['led'] = [False]*3
        df['vip_ctrl_type'] = df['in_type']
        if not conddf.in_type.values[ii]=="vip":
            df['vip_ctrl_type'] = 'ctrl_ledoff'
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   
#%%
bigdf_plot = bigdf[(bigdf.in_type.str.contains('vip'))]
bigdf_plot = bigdf_plot.groupby(['animal', 'lick_condition', 'in_type']).mean(numeric_only=True)
bigdf_plot.sort_values('lick_condition')

plt.figure(figsize=(4,6))
ax = sns.barplot(x="lick_condition", y="lick_probability_difference_condition",hue='in_type', data=bigdf_plot,
    palette={'vip_ledoff': "slategray", 'vip_ledon': "red"},
    errorbar='se', fill=False)
ax = sns.stripplot(x="lick_condition", y="lick_probability_difference_condition",hue='in_type', data=bigdf_plot,
    palette={'vip_ledoff': "slategray", 'vip_ledon': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.get_legend().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig(os.path.join(savedst, 'lick_vip.svg'), bbox_inches='tight')

bigdf_plot = bigdf
bigdf_plot = bigdf_plot.groupby(['animal', 'vip_ctrl_type']).mean(numeric_only=True)
plt.figure(figsize=(3,6))
ax = sns.barplot(x="vip_ctrl_type", y="lick_probability_difference",hue='vip_ctrl_type', data=bigdf_plot,
    palette={'ctrl_ledoff': "gray", 'ctrl_ledon': "lightcoral", 'vip_ledoff': "slategray", 'vip_ledon': "red"},
    errorbar='se', fill=False)
ax = sns.stripplot(x="vip_ctrl_type", y="lick_probability_difference",hue='vip_ctrl_type', data=bigdf_plot,
    palette={'ctrl_ledoff': "gray", 'ctrl_ledon': "lightcoral", 'vip_ledoff': "slategray", 'vip_ledon': "red"},
    s=9)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.savefig(os.path.join(savedst, 'lick_vip_mean.svg'), bbox_inches='tight')

#%%
# per session
bigdftest = bigdf[(bigdf.in_type.str.contains('vip'))]
scipy.stats.ranksums(bigdftest.loc[(bigdftest.in_type != 'vip_ledon'), 'lick_probability_difference'].values,
    bigdftest.loc[(bigdftest.in_type == 'vip_ledon'), 'lick_probability_difference'].values)

# per mouse
x1 = bigdf_plot.loc[(bigdf_plot.index.get_level_values('vip_ctrl_type') == 'vip_ledon'), 'lick_probability_difference'].values
x2 = bigdf_plot.loc[(bigdf_plot.index.get_level_values('vip_ctrl_type') == 'vip_ledoff'), 'lick_probability_difference'].values
x3 = bigdf_plot.loc[(bigdf_plot.index.get_level_values('vip_ctrl_type')=='ctrl_ledon'), 'lick_probability_difference'].values
x4 = bigdf_plot.loc[(bigdf_plot.index.get_level_values('vip_ctrl_type')=='ctrl_ledoff'), 'lick_probability_difference'].values
scipy.stats.f_oneway(x1[~np.isnan(x1)], x2, x3, x4[~np.isnan(x4)])
p_values= sp.posthoc_ttest([x1,x2,x3,x4])#,p_adjust='holm-sidak')
print(p_values)

# vs. controls
bigdf_plot = bigdf[~(bigdf.in_type.str.contains('vip'))]
bigdf_plot = bigdf_plot.groupby(['animal', 'lick_condition', 'in_type']).mean(numeric_only=True)
plt.figure()
ax = sns.barplot(x="lick_condition", y="lick_diff_quantile",hue='led', data=bigdf_plot,
    palette={False: "lightgray", True: "lightcoral"},
    ci=68, fill=False)
ax = sns.stripplot(x="lick_condition", y="lick_diff_quantile",hue='led', data=bigdf_plot,
    palette={False: "lightgray", True: "lightcoral"})
# ax = sns.stripplot(x="lick_condition", y="lick_diff_quantile",hue='led', data=bigdf_plot,
#     palette={False: "lightgray", True: "lightcoral"})
ax.tick_params(axis='x', labelrotation=90)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().set_visible(False)
plt.savefig(os.path.join(savedst, 'lick_ctrl.svg'), bbox_inches='tight')

bigdftest = bigdf[~(bigdf.in_type.str.contains('vip'))]
scipy.stats.mannwhitneyu(bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.led==False), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.led==True), 'lick_diff_quantile'].values)

#%%
# get trials before success (first and average)

dcts_opto = np.array(dcts)
dfs = []
for ii,dct in enumerate(dcts_opto):
    # print(conddf.iloc[ii])
    ts = dct['trials_before_success']
    try:
        df = pd.DataFrame([ts[0][0]], columns = ['trials_before_first_success_prev'])
        df['trials_before_first_success_opto'] = ts[1][0]
        df['trials_before_success_median_prev'] = np.nanmean(ts[0])
        df['trials_before_success_median_opto'] = np.nanmean(ts[1])
        
        df['trials_before_first_success_ledoff-on'] = df['trials_before_first_success_opto'].astype(int)-df['trials_before_first_success_prev'].astype(int)
        df['trials_before_success_med_ledoff-on'] = df['trials_before_success_median_opto']-df['trials_before_success_median_prev']
        # df['lick_dist'] = [scipy.spatial.distance.jensenshannon(xx,dists_o[ii]) for ii,xx in enumerate(dists_prev)]
        df['rewzones_transition'] = f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}'
        df['rewzones_opto'] = f'rz_{dct["rewzones"][1].astype(int)}'
        df['animal'] = conddf.animals.values[ii]   
        if conddf.optoep.values[ii]>1 and conddf.in_type.values[ii]=='vip':    
            df['in_type'] = f'{conddf.in_type.values[ii]}'
            df['led'] = 'LED on'
        elif conddf.optoep.values[ii]<2 and conddf.in_type.values[ii]=='vip':    
            df['in_type'] = f'{conddf.in_type.values[ii]}'
            df['led'] = 'LED off'
        elif conddf.optoep.values[ii]<2 and conddf.in_type.values[ii]!='vip':    
            df['in_type'] = 'ctrl'
            df['led'] = 'LED off'
        else: 
            df['in_type'] = 'ctrl'
            df['led'] = 'LED on'
        dfs.append(df)
    except Exception as e:
        print(e)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   

# %%
# plot
bigdf_plot = bigdf#[(bigdf.in_type.str.contains('vip'))]
bigdf_plot = bigdf_plot.groupby(['animal', 'in_type', 'led']).mean(numeric_only=True)
# bigdf_plot.sort_values('lick_condition')
fig, axes = plt.subplots(figsize=(4,6))
ax = sns.barplot(x="led", y="trials_before_first_success_ledoff-on", hue='in_type', data=bigdf_plot,
    palette={'ctrl': 'slategray','vip': "red"}, 
    errorbar='se', fill=False)
ax = sns.stripplot(x="led", y="trials_before_first_success_ledoff-on", hue='in_type',data=bigdf_plot,
                palette={'ctrl': 'slategray','vip': "red"},s=s)
ax.tick_params(axis='x', labelrotation=90)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().set_visible(False)

# ax.set_ylim(-5,8)
vipledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledon", 
                        "trials_before_first_success_ledoff-on"].values
vipledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledoff",
                    "trials_before_first_success_ledoff-on"].values
ctrlledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledoff",
                    "trials_before_first_success_ledoff-on"].values
ctrlledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledon",
                    "trials_before_first_success_ledoff-on"].values
scipy.stats.f_oneway(vipledon, vipledoff, ctrlledoff, ctrlledon)
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([vipledon,vipledoff,ctrlledon,ctrlledoff])#,p_adjust='holm-sidak')
print(p_values)
plt.savefig(os.path.join(savedst, 'trails_before_success_start_diff.svg'), bbox_inches='tight', dpi=500)

fig, axes = plt.subplots(figsize=(4,6))
ax = sns.barplot(x="led", y="trials_before_success_med_ledoff-on", hue='in_type', data=bigdf_plot,
    palette={'ctrl': 'slategray','vip': "red"}, 
    errorbar='se', fill=False)
ax = sns.stripplot(x="led", y="trials_before_success_med_ledoff-on", hue='in_type',data=bigdf_plot,
                palette={'ctrl': 'slategray','vip': "red"},s=s)
ax.tick_params(axis='x', labelrotation=90)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_legend().set_visible(False)

# ax.set_ylim(-5,8)
vipledon = bigdf_plot.loc[((bigdf_plot.index.get_level_values('in_type') == "vip")&(bigdf_plot.index.get_level_values('led') == "LED on")), 
            "trials_before_success_med_ledoff-on"].values
vipledoff = bigdf_plot.loc[((bigdf_plot.index.get_level_values('in_type') == "vip")&(bigdf_plot.index.get_level_values('led') == "LED off")), "trials_before_success_med_ledoff-on"].values
ctrlledoff = bigdf_plot.loc[((bigdf_plot.index.get_level_values('in_type') == "ctrl")&(bigdf_plot.index.get_level_values('led') == "LED on")), "trials_before_success_med_ledoff-on"].values
ctrlledon = bigdf_plot.loc[((bigdf_plot.index.get_level_values('in_type') == "ctrl")&(bigdf_plot.index.get_level_values('led') == "LED off")), "trials_before_success_med_ledoff-on"].values
scipy.stats.f_oneway(vipledon, vipledoff, ctrlledoff, ctrlledon)
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([vipledon,vipledoff,ctrlledon,ctrlledoff])#,p_adjust='holm-sidak')
print(p_values)

plt.savefig(os.path.join(savedst, 'trails_before_success_med_start_diff.svg'), bbox_inches='tight', dpi=500)

#%%
# by rew zone
bigdf_plot = bigdf#[(bigdf.in_type.str.contains('vip'))]
# bigdf_plot = bigdf_plot.groupby(['rewzones_transition']).mean()
# bigdf_plot.sort_values('lick_condition')
plt.figure()
ax = sns.barplot(x="in_type", y="trials_before_success_med_ledoff-on", hue='rewzones_transition', data=bigdf_plot,
    errorbar=('ci', 68), fill=False)
ax = sns.stripplot(x="in_type", y="trials_before_success_med_ledoff-on", hue='rewzones_transition',data=bigdf_plot)
ax.tick_params(axis='x', labelrotation=90)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylim(0,1)

#%%
vipledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledon", "trials_before_success_med_ledoff-on"].values
vipledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledoff", "trials_before_success_med_ledoff-on"].values
ctrlledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledoff", "trials_before_success_med_ledoff-on"].values
ctrlledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledon", "trials_before_success_med_ledoff-on"].values
scipy.stats.f_oneway(vipledon, vipledoff, ctrlledoff, ctrlledon)
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([vipledon,vipledoff,ctrlledon,ctrlledoff])#,p_adjust='holm-sidak')
print(p_values)