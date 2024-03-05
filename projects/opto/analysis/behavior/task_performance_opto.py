
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import get_success_failure_trials, get_performance, get_rewzones
#%%

days_cnt_an1 = 13; days_cnt_an2=14; days_cnt_an3=24; days_cnt_an4=11; days_cnt_an5=7; days_cnt_an6=9
days_cnt_an7=19; days_cnt_an8=10
animals = np.hstack([['e218']*(days_cnt_an1), ['e216']*(days_cnt_an2), \
                    ['e201']*(days_cnt_an3), ['e186']*(days_cnt_an4), ['e190']*(days_cnt_an5), ['e189']*(days_cnt_an6), \
                        ['e200']*days_cnt_an7, ['e217']*days_cnt_an8])
in_type = np.hstack([['vip']*(days_cnt_an1), ['vip']*(days_cnt_an2), \
                    ['sst']*(days_cnt_an3), ['pv']*(days_cnt_an4), ['ctrl']*(days_cnt_an5), ['ctrl']*(days_cnt_an6), \
                        ['sst']*days_cnt_an7,['vip']*days_cnt_an8])
days = np.array([20,21,22,23, 35, 38, 41, 44,45, 47,48,49,50,7,8,9,37, 41, 48, \
                50, 54,55,56,57,58,59,60,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,\
                2,3,4,5,31,32,33,34,36,37,40,33,34,35,40,41,42,45,35,36,37,38,39,40,41,42,44, \
                    65,66,67,68,69,70,72,73,74,76,81,82,83,84,85,86,87,88,89,2,3,4,5,6,7,8,9,10,11])#[20,21,22,23]#
optoep = np.array([-1, -1,  0,  0,  3,  2,  3,  2,0,3,  0,0,2,  0,  0,  0,  2,  3,  2,  3,
        3,0,0,2, 0,0,3,-1, -1, -1,  2,  3,  0,  2,  3,  0,  2,  3,  0,  2,  3,  0,
        2,  3,  0,  2,  3,  0,  2,  3,  3,  0,  0,  0,  0,  2,  3,  2,  3,
        2,  3,  2, -1, -1, -1,  3,  0,  1,  3, -1, -1, -1, -1,  2,  3,  2,
        0,  2,  2,  3,  0,  2,  3,  0,  3,  0,  2,  0,  2,  3,  0,  2,  3,
        0,  2,  3,  0, -1, -1, -1, -1, 2, 3, 2, 0, 3, 0])#[2,3,2,3]
conddf = pd.DataFrame()
conddf['animals'] = animals
conddf['in_type'] = in_type
conddf['days'] = days
conddf['optoep'] = optoep
# days = np.arange(2,21)
# optoep = [-1,-1,-1,-1,2,3,2,0,3,0,2,0,2, 0,0,0,0,0,2]
# corresponding to days analysing
#%%
bin_size = 3
# com shift analysis
dcts = []
for dd,day in enumerate(days):
    dct = {}
    animal = animals[dd]
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
    eptest = optoep[dd]
    if optoep[dd]<2: eptest = random.randint(2,3)   
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
    df['animal'] = animals[ii]    
    if optoep[ii]>1:    
        df['opto'] = True    
        df['in_type'] = f'{in_type[ii]}_ledon'
    else:#elif optoep[ii]==0: 
        df['opto'] = False    
        df['in_type'] = f'{in_type[ii]}_ledoff'
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   
bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_opto')
plt.figure()
ax = sns.barplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#%%
# test sig
bigdf_test = bigdf
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon'), 'rates_diff'].values
x1 = x1[~np.isnan(x1)]
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon'), 'rates_diff'].values
x3 = bigdf_test.loc[(bigdf_test.in_type=='sst_ledon'), 'rates_diff'].values
x3 = x3[~np.isnan(x3)]
x4 = bigdf_test.loc[(bigdf_test.in_type=='sst_ledoff'), 'rates_diff'].values
x4 = x4[~np.isnan(x4)]
x5 = bigdf_test.loc[(bigdf_test.in_type=='ctrl_ledon'), 'rates_diff'].values
x5 = x5[~np.isnan(x5)]
x6 = bigdf_test.loc[(bigdf_test.in_type=='ctrl_ledoff'), 'rates_diff'].values
x6 = x6[~np.isnan(x6)]
scipy.stats.kruskal(x1, x2, x3, x4, x5, x6)
import scikit_posthocs as sp
# using the posthoc_dunn() function
p_values= sp.posthoc_dunn([x1,x2,x3,x4,x5,x6], p_adjust = 'holm-sidak')
print(p_values)
# per rew loc
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_1'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_1'), 'rates_diff'].values
x3 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_2'), 'rates_diff'].values
x4 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_2'), 'rates_diff'].values
x5 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_3'), 'rates_diff'].values
x6 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_3'), 'rates_diff'].values
scipy.stats.kruskal(x1[~np.isnan(x1)], x2, x3, x4, x5, x6)
# using the posthoc_dunn() function
p_values= sp.posthoc_dunn([x1,x2,x3,x4,x5,x6], p_adjust = 'holm')
print(p_values)
# vs do a ttest or ranksums
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_2'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto=='rz_2'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2)
#%%
scipy.stats.kruskal(bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto == 'rz_1'), 'rates_diff'].values, \
        bigdf_test.loc[(bigdf_test.in_type=='vip_ledoff') & (bigdf_test.rewzones_opto == 'rz_1'), 'rates_diff'].values,
        bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto == 'rz_2'), 'rates_diff'].values, \
        bigdf_test.loc[(bigdf_test.in_type=='vip_ledoff') & (bigdf_test.rewzones_opto == 'rz_2'), 'rates_diff'].values,
        bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_opto == 'rz_3'), 'rates_diff'].values, \
        bigdf_test.loc[(bigdf_test.in_type=='vip_ledoff') & (bigdf_test.rewzones_opto == 'rz_3'), 'rates_diff'].values)

bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_transition')
plt.figure()
ax = sns.barplot(x="rewzones_transition", y="rates_diff",hue='opto', data=bigdf_plot)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
x2 = bigdf_test.loc[(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
scipy.stats.ttest_ind(x1[~np.isnan(x1)], x2)

#%%
# plot lick tuning 
# first, just take average
dfs = []
for ii,dct in enumerate(dcts_opto):
    lick_p_prev, lick_p_opto = dct['lick_p']
    pre_prev, rew_prev, post_prev = lick_p_prev
    pre_o, rew_o, post_o = lick_p_opto
    df = pd.DataFrame()
    df['lick_prob_prev'] = [np.trapz(pre_prev.values), np.trapz(rew_prev.values), \
        np.trapz(post_prev.values)]
    df['lick_prob_opto'] = [np.trapz(pre_o.values), np.trapz(rew_o.values), \
        np.trapz(post_o.values)]
    df['lick_auc'] = df['lick_prob_opto']-df['lick_prob_prev']
    df['lick_condition'] = ['pre', 'rew', 'post']
    df['rewzones_transition'] = [f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}']*3
    df['rewzones_opto'] = [f'rz_{dct["rewzones"][1].astype(int)}']*3
    df['animal'] = [animals[ii]]*3   
    if optoep[ii]>1:    
        df['in_type'] = [f'{in_type[ii]}_ledon']*3
        df['led'] = ['ledon']*3
    else: 
        df['led'] = ['ledoff']*3
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   
#%%
plt.figure()
bigdf_plot = bigdf[~(bigdf.in_type.str.contains('vip'))]
ax = sns.barplot(x="lick_condition", y="lick_auc",hue='in_type', data=bigdf_plot)
ax.tick_params(axis='x', labelrotation=90)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

scipy.stats.ttest_ind(bigdf_plot.loc[(bigdf_plot.lick_condition == 'pre') & (bigdf_plot.in_type == 'vip_ledoff'), 'lick_auc'].values,
    bigdf_plot.loc[(bigdf_plot.lick_condition == 'pre') & (bigdf_plot.in_type == 'vip_ledon'), 'lick_auc'].values)

scipy.stats.kruskal(bigdf_test.loc[bigdf_test.in_type=='vip_ledon', 'lick_auc'].values, \
        bigdf_test.loc[bigdf_test.in_type=='vip_ledoff', 'lick_auc'].values, 
        bigdf_test.loc[bigdf_test.in_type=='sst_ledoff', 'lick_auc'].values,
        bigdf_test.loc[bigdf_test.in_type=='sst_ledon', 'lick_auc'].values,
        bigdf_test.loc[bigdf_test.in_type=='pv_ledoff', 'lick_auc'].values,
        bigdf_test.loc[bigdf_test.in_type=='pv_ledon', 'lick_auc'].values, 
        bigdf_test.loc[bigdf_test.in_type=='ctrl_ledoff', 'lick_auc'].values,
        bigdf_test.loc[bigdf_test.in_type=='ctrl_ledon', 'lick_auc'].values)


# anova 
# ax.set_ylim(0,.12)
#%%