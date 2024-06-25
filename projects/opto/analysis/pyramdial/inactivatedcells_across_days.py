"""look at inactivated cells tracked across days
KL divergence: distance between 2 distributions, given how likely you can
make the second distribution from the first
make tuning curve relative to reward
-1 to 0 and 0 to 1
bin by 0.02 (% of track before or after rew, etc.)
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
from placecell import get_moving_time, calc_COM_EH, make_tuning_curves_relative_to_reward
from projects.opto.behavior.behavior import get_success_failure_trials
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
tracked_inactive_activity = {}
for dd,day in enumerate(conddf.days.values):
    dct = dcts[dd]
    animal = conddf.animals.values[dd]
    if animal in animals:
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = days_tracked_per_an[animal]
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'Fc3', 
            'coms_early_trials', 'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel',
            'trialnum', 'rewards', 'iscell', 'bordercells'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        comp = dct['comp'];bin_size = 3;tcs_early = []; tcs_late = []        
        Fc3 = fall['Fc3']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr = 5 # last 5 trials
        bins=100
        # remake tuning curves relative to reward        
        ypos_rel, tcs_late, coms = make_tuning_curves_relative_to_reward(eps,rewlocs,ybinned,track_length,Fc3,trialnum,
            rewards,forwardvel,rewsize,lasttr=lasttr,bins=bins)
        tcs_late = np.array(tcs_late); coms = np.array(coms)
        ypos_rel = np.concatenate(ypos_rel)
        pyr_tc_s2p_cellind = fall['pyr_tc_s2p_cellind'][0]
        # suite2p indices
        inactive_cell_ind = pyr_tc_s2p_cellind[dct['inactive']]
        comp = dct['comp']
        # tracked cell ind of inactive cell this day
        try:
            tracked_inactive_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in inactive_cell_ind]
            tracked_inactive_cell_inds[f'{animal}_{day:03d}'] = tracked_inactive_cell_ind   
            tracked_cells_that_are_inactive_pyr_id = tracked_lut[day].values[tracked_inactive_cell_ind]
            inactive_cells_that_are_tracked_iind = [np.where(pyr_tc_s2p_cellind==xx)[0][0] for xx in tracked_cells_that_are_inactive_pyr_id]
            # includes s2p indices of inactive cells so you can find them in the tracked lut
            tracked_inactive_activity[f'{animal}_{day:03d}'] = [[tcs_late[c][inactive_cells_that_are_tracked_iind] for c in comp],\
                    rewlocs[comp],coms[comp][:,[inactive_cells_that_are_tracked_iind]]]
            # tracked_lut[day].values[tracked_inactive_cell_ind] == dct['inactive'][inactive_cells_that_are_tracked_iind]
        except Exception as e:
            print(e)
#%%
# tracked cell activity
tc_tracked_per_cond = {}
for k,v in tracked_inactive_cell_inds.items():
    if k in tracked_inactive_activity.keys():
        tcs = tracked_inactive_activity[k][0]
        rewlocs = tracked_inactive_activity[k][1]
        coms = tracked_inactive_activity[k][2]
        animal = k[:-4]
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = days_tracked_per_an[animal]
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
        tc_tracked = np.ones((2, len(tracked_lut), 100))*np.nan
        tracked_cell_id = v    
        tc_tracked[:,tracked_cell_id,:] = tcs
        tc_tracked_per_cond[k] = tc_tracked
#%%    
# compile per animal tcs
annm = 'e216'
an = np.array([v for k,v in tc_tracked_per_cond.items() if k[:-4]==annm])
# remove cells that are nan every tracked day
mask = (np.sum(np.sum(np.isnan(an[:,0,:,:]),axis=2),axis=0)<800) # not nan in all positions across all days
an = an[:,:,mask,:]
shp = int(np.ceil(np.sqrt(an.shape[2])))
for dy in range(an.shape[0]):    
    fig, axes = plt.subplots(ncols=shp,
                            nrows=shp,sharex=True,
                            figsize=(40,40))
    plt.rc('font', size=20)
    rr=0;cc=0
    for ii in range(an.shape[2]):
        ax=axes[rr,cc]
        ax.plot(an[dy,0,ii,:], color='slategray')
        ax.plot(an[dy,1,ii,:], color='r')
        ax.set_title(f'cell {ii}')
        ax.spines[['top','right']].set_visible(False)
        ax.axvline(x = int(bins/2)+1, color='k', linestyle='--')
        rr+=1
        if rr>np.ceil(np.sqrt(an.shape[2]))-1:cc+=1;rr=0        
    fig.suptitle(f'Day {dy}')
    # fig.suptitle(f'all days')
    fig.tight_layout()

#%%
# tracked cell frequency
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
# plt.savefig(os.path.join(savedst,'kl_div_vip.jpg'),dpi=500,bbox_inches='tight')

# ctrl mice
# separate opto vs. normal days
animals = ['e201','e200','e186','e189','e190']
maxbin=5; ymax = 21
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(6,6), sharey=True)
js_s_ctrl = []
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
# plt.savefig(os.path.join(savedst,'kl_div_ctrl.jpg'),dpi=500,bbox_inches='tight')

scipy.stats.ttest_ind(js_s_ctrl,js_s_opto)