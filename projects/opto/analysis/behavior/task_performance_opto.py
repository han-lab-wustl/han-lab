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
plt.rc('font', size=16)          # controls default text sizes

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

from behavior import get_success_failure_trials, get_performance, get_rewzones

# import condition df
conddf = pd.read_csv(r"Z:\conddf_behavior.csv", index_col=None)
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
    licks = licks<=-0.065 # remake boolean
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)   
    if len(eps)<4: eptest = 2 # if no 3 epochs    
    rates_opto, rates_prev, lick_prob_opto, lick_prob_prev, trials_bwn_success_opto, trials_bwn_success_prev = get_performance(eptest, eps, 
    trialnum, rewards, licks, ybinned, rewlocs, forwardvel, rewsize)
    rewzones = get_rewzones(rewlocs, 1.5)

    dct['rates'] = [rates_prev, rates_opto]    
    dct['lick_p'] = [lick_prob_prev, lick_prob_opto]
    dct['rewlocs'] = [rewlocs[eptest-2], rewlocs[eptest-1]]
    dct['rewzones'] = [rewzones[eptest-2], rewzones[eptest-1]]
    dct['trials_before_success'] = [trials_bwn_success_prev, trials_bwn_success_opto]
    dcts.append(dct)
#%%
# plot performance by rewzones
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
                palette={False: "slategray", True: "red"},
                ci=68, fill=False)
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
p_values= sp.posthoc_ttest([x1,x2,x3,x4])#,p_adjust='holm-sidak')
print(p_values)
plt.tight_layout()
#%%
# plot rates by rewzone
bigdf_plot = bigdf[bigdf.in_type.str.contains('vip') & ~bigdf.in_type.str.contains('ctrl')]
bigdf_plot = bigdf_plot.sort_values('rewzones_opto')
plt.figure()
ax = sns.barplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "slategray", True: "red"},
                ci=68, fill=False)
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
                palette={False: "slategray", True: "red"},
                ci=68, fill=False)
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
                palette={False: "lightgray", True: "lightcoral"},
                ci=68, fill=False)
sns.stripplot(x="rewzones_opto", y="rates_diff",hue='opto', data=bigdf_plot,
                palette={False: "lightgray", True: "lightcoral"},
                ci=68, fill=False)
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
                ci=68, fill=False)
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
scipy.stats.ranksums(x1[~np.isnan(x1)], x2)

# tests 1-3 transition
bigdf_test = bigdf
x1 = bigdf_test.loc[~(bigdf_test.in_type=='vip_ledon') & (bigdf_test.rewzones_transition=='rz_1-3'), 'rates_diff'].values
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
bigdf_plot = bigdf_plot.groupby(['animal', 'lick_condition', 'in_type']).mean()
bigdf_plot.sort_values('lick_condition')
plt.figure()
ax = sns.barplot(x="lick_condition", y="lick_diff_quantile",hue='in_type', data=bigdf_plot,
    palette={'vip_ledoff': "slategray", 'vip_ledon': "red"},
    ci=68, fill=False)
ax = sns.stripplot(x="lick_condition", y="lick_diff_quantile",hue='in_type', data=bigdf_plot,
    palette={'vip_ledoff': "slategray", 'vip_ledon': "red"})
ax.tick_params(axis='x', labelrotation=90)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# per session
bigdftest = bigdf[(bigdf.in_type.str.contains('vip'))]
scipy.stats.ranksums(bigdftest.loc[(bigdftest.lick_condition == 'pre') & (bigdftest.in_type != 'vip_ledon'), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.lick_condition == 'pre') & (bigdftest.in_type == 'vip_ledon'), 'lick_diff_quantile'].values)
scipy.stats.ranksums(bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.in_type != 'vip_ledon'), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.lick_condition == 'rew') & (bigdftest.in_type == 'vip_ledon'), 'lick_diff_quantile'].values)

# per mouse
bigdftest=bigdf_plot
scipy.stats.ttest_rel(bigdftest.loc[(bigdftest.index.get_level_values('lick_condition') == 'pre') & (bigdftest.index.get_level_values('in_type') != 'vip_ledon'), 'lick_diff_quantile'].values,
    bigdftest.loc[(bigdftest.index.get_level_values('lick_condition') == 'pre') & (bigdftest.index.get_level_values('in_type') == 'vip_ledon'), 'lick_diff_quantile'].values)


# vs. controls
bigdf_plot = bigdf[~(bigdf.in_type.str.contains('vip'))]
bigdf_plot = bigdf_plot.groupby(['animal', 'lick_condition', 'in_type']).mean()
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
        df['trials_before_success_median_prev'] = np.median(ts[0])
        df['trials_before_success_median_opto'] = np.median(ts[1])
        
        df['trials_before_first_success_ledoff-on'] = df['trials_before_first_success_opto']-df['trials_before_first_success_prev']
        df['trials_before_success_med_ledoff-on'] = df['trials_before_success_median_opto']-df['trials_before_success_median_prev']
        # df['lick_dist'] = [scipy.spatial.distance.jensenshannon(xx,dists_o[ii]) for ii,xx in enumerate(dists_prev)]
        df['rewzones_transition'] = f'rz_{dct["rewzones"][0].astype(int)}-{dct["rewzones"][1].astype(int)}'
        df['rewzones_opto'] = f'rz_{dct["rewzones"][1].astype(int)}'
        df['animal'] = conddf.animals.values[ii]   
        if conddf.optoep.values[ii]>1 and conddf.in_type.values[ii]=='vip':    
            df['in_type'] = f'{conddf.in_type.values[ii]}_ledon'
            df['led'] = True
        elif conddf.optoep.values[ii]<2 and conddf.in_type.values[ii]=='vip':    
            df['in_type'] = f'{conddf.in_type.values[ii]}_ledoff'
        elif conddf.optoep.values[ii]<2 and conddf.in_type.values[ii]!='vip':    
            df['in_type'] = 'ctrl_ledoff'
            df['led'] = False
        else: 
            df['in_type'] = 'ctrl_ledon'
            df['led'] = False
        dfs.append(df)
    except Exception as e:
        print(e)
bigdf = pd.concat(dfs)
bigdf.reset_index(drop=True, inplace=True)   

# %%
# plot
bigdf_plot = bigdf#[(bigdf.in_type.str.contains('vip'))]
bigdf_plot = bigdf_plot.groupby(['animal', 'in_type']).mean()
# bigdf_plot.sort_values('lick_condition')
plt.figure()
ax = sns.barplot(x="in_type", y="trials_before_first_success_ledoff-on", hue='in_type', data=bigdf_plot,
    palette={'ctrl_ledon': "lightcoral", 'ctrl_ledoff': 'lightgray', 'vip_ledon': "red",
            'vip_ledoff': "slategray"}, 
    errorbar=('ci', 68), fill=False)
ax = sns.stripplot(x="in_type", y="trials_before_first_success_ledoff-on", hue='in_type',data=bigdf_plot,
                palette={'ctrl_ledon': "lightcoral", 'ctrl_ledoff': 'lightgray', 'vip_ledon': "red",
            'vip_ledoff': "slategray"})
ax.tick_params(axis='x', labelrotation=90)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_ylim(-5,8)
vipledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledon", "trials_before_success_med_ledoff-on"].values
vipledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "vip_ledoff", "trials_before_success_med_ledoff-on"].values
ctrlledoff = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledoff", "trials_before_success_med_ledoff-on"].values
ctrlledon = bigdf_plot.loc[bigdf_plot.index.get_level_values('in_type') == "ctrl_ledon", "trials_before_success_med_ledoff-on"].values
scipy.stats.f_oneway(vipledon, vipledoff, ctrlledoff, ctrlledon)
import scikit_posthocs as sp
p_values= sp.posthoc_ttest([vipledon,vipledoff,ctrlledon,ctrlledoff])#,p_adjust='holm-sidak')
print(p_values)

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