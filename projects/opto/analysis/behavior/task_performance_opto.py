
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import get_success_failure_trials, get_performance, get_rewzones
#%%
# import condition df
conddf = pd.read_csv(r"Z:\conddf.csv", index_col=None)
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing
#%%
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
    eps = np.hstack(np.ravel([eps, len(np.hstack(VR['changeRewLoc']))]))
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
    licks = licks<=-0.075 # remake boolean
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)   
    if len(eps)<4: eptest = 2 # if no 3 epochs    
    rates_opto, rates_prev, lick_p_opto, lick_p_prev = get_performance(eptest, eps, 
    trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize)
    rewzones = get_rewzones(rewlocs, 1.5)

    dct['rates'] = [rates_prev, rates_opto]    
    dct['lick_p'] = [lick_p_prev, lick_p_opto]
    dct['rewlocs'] = [rewlocs[eptest-2], rewlocs[eptest-1]]
    dct['rewzones'] = [rewzones[eptest-2], rewzones[eptest-1]]
    dcts.append(dct)
#%%
# plot performance by rewzones
    # plot coms of enriched cells    
dcts_opto = np.array(dcts)

dfs=[]; dfs_diff = []
for ii,dct in enumerate(dcts_opto):
    rates_prev, rates_opto = dct['rates']   
    df = pd.DataFrame([rates_prev], columns = ['rates_prev'])
    df['rates_diff'] = rates_opto-rates_prev
    df['rewzones_transition'] = f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}'
    df['rewzones_opto'] = f'rz_{dct["rewzones"][1].astype(int)}'
    df['animal'] = conddf.animals.values[ii]    
    if conddf.optoep.values[ii]>1:    
        df['opto'] = True    
        df['in_type'] = f'{conddf.in_type.values[ii]}_ledon'
        df['vip_ctrl_type'] = df['in_type']
        if not conddf.in_type.values[ii]=="vip":
            df['vip_ctrl_type'] = 'ctrl_ledon'
    else:#elif optoep[ii]==0: 
        df['opto'] = False    
        df['in_type'] = f'{conddf.in_type.values[ii]}_ledoff'
        df['vip_ctrl_type'] = df['in_type']
        if not conddf.in_type.values[ii]=="vip":
            df['vip_ctrl_type'] = 'ctrl_ledoff'
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   
#%%
# plot rates vip vs. ctl led off and on
bigdf_plot = bigdf.groupby(['animal', 'vip_ctrl_type']).mean()
bigdf_plot['vip_ctrl_type'] = [bigdf_plot.index[xx][1] for xx in range(len(bigdf_plot.index))]
plt.figure()
ax = sns.barplot(x="vip_ctrl_type", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
sns.stripplot(x="vip_ctrl_type", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
x1 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type == 'vip_ledon'), 'rates_diff'].values
x2 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type == 'vip_ledoff'), 'rates_diff'].values
x3 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type=='ctrl_ledon'), 'rates_diff'].values
x4 = bigdf_plot.loc[(bigdf_plot.vip_ctrl_type=='ctrl_ledoff'), 'rates_diff'].values
scipy.stats.f_oneway(x1[~np.isnan(x1)], x2, x3, x4[~np.isnan(x4)])
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([x1,x2,x3,x4])
print(p_values)
#%%
# plot rates by rewzone
bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_opto')
plt.figure()
ax = sns.barplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
sns.stripplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# plot by rew zone transition
bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_transition')
plt.figure()
ax = sns.barplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
sns.stripplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# controls
bigdf_plot = bigdf[~bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_opto')
plt.figure()
ax = sns.barplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"})
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
                palette={False: "lightgray", True: "lightcoral"})
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
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_3-1'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_3-1'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2)

# tests 1-3 transition
bigdf_test = bigdf
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2)

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
    df['lick_prob_prev'] = [np.percentile(pre_prev.values,75), np.percentile(rew_prev.values,75), \
        np.percentile(post_prev.values,75)]
    df['lick_prob_opto'] = [np.percentile(pre_o.values,75), np.percentile(rew_o.values,75), \
        np.percentile(post_o.values,75)]
    df['lick_diff_quantile'] = df['lick_prob_opto']-df['lick_prob_prev']
    # df['lick_dist'] = [scipy.spatial.distance.jensenshannon(xx,dists_o[ii]) for ii,xx in enumerate(dists_prev)]
    df['lick_condition'] = ['pre', 'rew', 'post']
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
    else: 
        df['in_type'] = [f'{conddf.in_type.values[ii]}_ledoff']*3
        df['led'] = [False]*3
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   
#%%
bigdf_plot = bigdf[(bigdf.in_type.str.contains('vip'))]
plt.figure()
ax = sns.barplot(x="lick_condition", y="lick_diff_quantile",hue='in_type', data=bigdf_plot,
    palette={'vip_ledoff': "slategray", 'vip_ledon': "red"})
# ax = sns.stripplot(x="lick_condition", y="lick_diff_quantile",hue='in_type', data=bigdf_plot,
#     palette={'vip_ledoff': "slategray", 'vip_ledon': "red"})
ax.tick_params(axis='x', labelrotation=90)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

bigdftest = bigdf[(bigdf.in_type.str.contains('vip'))]
scipy.stats.mannwhitneyu(bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.in_type != 'vip_ledon'), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.in_type == 'vip_ledon'), 'lick_diff_quantile'].values)

# vs. controls
bigdf_plot = bigdf[~(bigdf.in_type.str.contains('vip'))]
plt.figure()
ax = sns.barplot(x="lick_condition", y="lick_diff_quantile",hue='led', data=bigdf_plot,
    palette={False: "lightgray", True: "lightcoral"})
# ax = sns.stripplot(x="lick_condition", y="lick_diff_quantile",hue='led', data=bigdf_plot,
#     palette={False: "lightgray", True: "lightcoral"})
ax.tick_params(axis='x', labelrotation=90)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

bigdftest = bigdf[~(bigdf.in_type.str.contains('vip'))]
scipy.stats.mannwhitneyu(bigdftest.loc[(bigdftest.lick_condition == 'post') & (bigdftest.led==False), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.lick_condition == 'post') & (bigdftest.led==True), 'lick_diff_quantile'].values)


# anova 
# ax.set_ylim(0,.12)
#%%