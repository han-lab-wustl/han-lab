"""look at inactivated cells tracked across days
KL divergence: distance between 2 distributions, given how likely you can
make the second distribution from the first
"""
    
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import chain
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_cell_tracking.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'
animals = ['e216','e217','e218','e200','e201','e186','e189','e190']
days_tracked_per_an = {'e216':np.concatenate([[32,33],range(35,64),[65]]),
                    'e217': np.concatenate([range(2,21),[26,27,29,31,32,34,39,43,47]]),
                    'e218':np.arange(20,51),
                    'e200':np.concatenate([range(62,71),[72,73,74,76],range(80,91)]),
                    'e201':np.arange(55,76),
                    'e186':np.arange(1,52),
                    'e189':np.concatenate([[7,8],range(10,16),range(17,22),range(24,43),[44,45,46]]),
                    'e190':np.concatenate([range(6,10),[11,13],range(15,20),[21,22,24,27,28,29,33,34,35],
                                range(40,44),[45]])
                    }
with open(r"Z:\dcts_com_opto_cell_track_wcomp.p", "rb") as fp: #unpickle
    dcts = pickle.load(fp)

tracked_inactive_cell_inds = {}
for dd,day in enumerate(conddf.days.values):
    dct = dcts[dd]
    animal = conddf.animals.values[dd]
    if animal in animals:
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = days_tracked_per_an[animal]
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
        track_length = 270
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
            'tuning_curves_late_trials', 'coms_early_trials', 'pyr_tc_s2p_cellind'])
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        eps = np.append(eps, len(changeRewLoc))    
        comp = dct['comp'] # eps to compare    
        bin_size = 3    
        tcs_early = []; tcs_late = []
        for ii,tc in enumerate(fall['tuning_curves_early_trials'][0]):
            tcs_early.append(np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_early_trials'][0][ii]])))
            tcs_late.append(np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_late_trials'][0][ii]])))
        tcs_early = np.array(tcs_early)
        tcs_late = np.array(tcs_late)
        coms = fall['coms'][0]
        pyr_tc_s2p_cellind = fall['pyr_tc_s2p_cellind'][0] # s2p indices of cells in tuning curve
        # not_active_cell_ind = pyr_tc_s2p_cellind[~dct['inactive']] # s2p index of inactive cell that day
        # not_inactive_cell_ind = pyr_tc_s2p_cellind[~dct['active']]
        # inactive_cell_ind = np.intersect1d(not_active_cell_ind,not_inactive_cell_ind)
        inactive_cell_ind = pyr_tc_s2p_cellind[dct['inactive']]
        # tracked cell ind of inactive cell this day
        try:
            tracked_inactive_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in inactive_cell_ind]
            tracked_inactive_cell_inds[f'{animal}_{day:03d}'] = tracked_inactive_cell_ind
        except Exception as e:
            print(e)
#%%
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence

# separate opto vs. normal days
plt.rc('font', size=14)
maxbin = 5; ymax = 36
animals = ['e216','e218']
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(6,6), sharey=True)
js_s_opto = []
other_dist_opto = []
for i,an in enumerate(animals):
    opto = []
    ctrl = []
    for k,v in tracked_inactive_cell_inds.items():
        if str(k[:-4]) == an:
            if conddf.loc[(conddf.animals==str(k[:-4])) & (conddf.days==int(k[-3:])), 
            'optoep'].values[0]>1:
                opto.append(v)
            elif conddf.loc[(conddf.animals==str(k[:-4])) & (conddf.days==int(k[-3:])), 
            'optoep'].values[0]<2:
                ctrl.append(v)
    opto_count = Counter(chain(*opto))
    ctrl_count = Counter(chain(*ctrl))
    other_dist_opto.append(jensen_shannon(opto_count,ctrl_count))
    opto_count = np.array(list(opto_count.values()))
    opto_count = opto_count[opto_count>1]
    ctrl_count = np.array(list(ctrl_count.values()))
    ctrl_count = ctrl_count[ctrl_count>1]
    bins = np.linspace(1,maxbin,maxbin)
    if i==0:
        axes[0].hist(opto_count, bins=bins, alpha=0.5, color='red', linewidth=5, 
                label = 'LED on')            
        axes[1].hist(ctrl_count, bins=bins, alpha=0.5, color='slategray', 
                linewidth=5, label = 'LED off')
    else:
        axes[0].hist(opto_count, bins=bins, alpha=0.5, color='red', linewidth=5)            
        axes[1].hist(ctrl_count, bins=bins, alpha=0.5, color='slategray', 
                linewidth=5)
    
    js = KL(np.histogram(np.array(opto_count),bins=bins)[0],
                                            np.histogram(np.array(ctrl_count),bins=bins)[0])
    js_s_opto.append(js)
    
axes[0].legend();axes[1].legend()
axes[1].set_xlabel('# Days is inactive')
axes[0].set_ylabel('# of Cells')
axes[0].set_xticks(np.arange(1,maxbin+1))
axes[1].set_xticks(np.arange(1,maxbin+1))
axes[0].set_yticks(np.arange(0,ymax,5))
axes[1].set_title(f'VIP Inhibition (n=2) \n KL Divergence: {np.nanmedian(js_s_opto):.1f}')
axes[0].spines[['top','right']].set_visible(False)
axes[1].spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(savedst,'kl_div_vip.jpg'),dpi=500,bbox_inches='tight')

# ctrl mice
# separate opto vs. normal days
animals = ['e201','e200','e186','e189','e190']
maxbin=5; ymax = 21
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(6,6), sharey=True)
js_s_ctrl = []
other_dist_ctrl=[]
for i,an in enumerate(animals):
    opto = []
    ctrl = []
    for k,v in tracked_inactive_cell_inds.items():
        if str(k[:-4]) == an:
            if conddf.loc[(conddf.animals==str(k[:-4])) & (conddf.days==int(k[-3:])), 
            'optoep'].values[0]>1:
                opto.append(v)
            elif conddf.loc[(conddf.animals==str(k[:-4])) & (conddf.days==int(k[-3:])), 
            'optoep'].values[0]<2:
                ctrl.append(v)
    opto_count = Counter(chain(*opto))
    ctrl_count = Counter(chain(*ctrl))
    other_dist_ctrl.append(jensen_shannon(opto_count,ctrl_count))
    opto_count = np.array(list(opto_count.values()))
    opto_count = opto_count[opto_count>1]
    ctrl_count = np.array(list(ctrl_count.values()))
    ctrl_count = ctrl_count[ctrl_count>1]
    bins = np.linspace(1,maxbin,maxbin)
    if i==0:
        axes[0].hist(opto_count, bins=bins, alpha=0.5, color='coral', linewidth=5, 
                label = 'LED on')            
        axes[1].hist(ctrl_count, bins=bins, alpha=0.5, color='lightgray', 
                linewidth=5, label = 'LED off')
    else:
        axes[0].hist(opto_count, bins=bins, alpha=0.5, color='coral', linewidth=5)            
        axes[1].hist(ctrl_count, bins=bins, alpha=0.5, color='gray', 
                linewidth=5)
    
    js = KL(np.histogram(np.array(opto_count),bins=bins)[0],
                                            np.histogram(np.array(ctrl_count),bins=bins)[0])
    js_s_ctrl.append(js)
    
axes[0].legend();axes[1].legend()
axes[1].set_xlabel('# Days is inactive')
axes[0].set_ylabel('# of Cells')
axes[1].set_xticks(np.arange(1,maxbin+1))
axes[0].set_xticks(np.arange(1,maxbin+1))
axes[0].set_yticks(np.arange(0,ymax,5))
axes[1].set_title(f'Control (n=5) \n KL Divergence: {np.nanmedian(js_s_ctrl):.1f}')
# ax.set_xticks(np.arange(1,8))
axes[0].spines[['top','right']].set_visible(False)
axes[1].spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(savedst,'kl_div_ctrl.jpg'),dpi=500,bbox_inches='tight')

scipy.stats.ttest_ind(js_s_ctrl,js_s_opto)