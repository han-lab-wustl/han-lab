    """look at inactivated cells tracked across days
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
with open("Z:\dcts_com_opto_cell_track_wcomp.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)

tracked_inactive_cell_inds = {}
for dd,day in enumerate(conddf.days.values):
    dct = dcts[dd]
    animal = conddf.animals.values[dd]
    if animal =='e218':
        tracked_lut = scipy.io.loadmat(r"Y:\analysis\celltrack\e218_daily_tracking\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = np.arange(20,51)
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
    elif animal == 'e216':
        tracked_lut = scipy.io.loadmat(r"Y:\analysis\celltrack\e216_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = np.concatenate([[32,33],range(35,64),[65]])
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
    if animal == 'e218' or animal == 'e216':
        track_length = 270
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
            'tuning_curves_late_trials', 'coms_early_trials', 'pyr_tc_s2p_cellind'])
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
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
        inactive_cell_ind = pyr_tc_s2p_cellind[dct['inactive']] # s2p index of inactive cell that day
        # tracked cell ind of inactive cell this day
        try:
            tracked_inactive_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in inactive_cell_ind]
            tracked_inactive_cell_inds[f'{animal}_{day:03d}'] = tracked_inactive_cell_ind
        except Exception as e:
            print(e)
#%%
# separate opto vs. normal days
animals = ['e218', 'e216']
fig, ax = plt.subplots(figsize=(4,6))
js_s = []
for i,an in enumerate(animals):
    opto = []
    ctrl = []
    for k,v in tracked_inactive_cell_inds.items():
        if str(k[:-4]) == an:
            if conddf.loc[(conddf.animals==str(k[:-4])) & (conddf.days==int(k[-3:])), 'optoep'].values[0]>1:
                opto.append(v)
            else:
                ctrl.append(v)
    opto_count = Counter(chain(*opto))
    ctrl_count = Counter(chain(*ctrl))
    if i==0:
        ax.hist(list(opto_count.values()),bins=4, alpha=0.3, color='red', linewidth=5, 
                label = 'LED on')            
        ax.hist(list(ctrl_count.values()),bins=4, alpha=0.3, color='slategray', 
                linewidth=5, label = 'LED off')
    else:
        ax.hist(list(opto_count.values()),bins=4, alpha=0.3, color='red', linewidth=5)            
        ax.hist(list(ctrl_count.values()),bins=4, alpha=0.3, color='slategray', 
                linewidth=5)
    
    js = scipy.spatial.distance.jensenshannon(np.histogram(np.array(list(opto_count.values())),bins=8)[0],
                                            np.histogram(np.array(list(ctrl_count.values())),bins=8)[0])
    js_s.append(js)
    

ax.legend()
ax.set_xlabel('# Days cell is inactive')
ax.set_ylabel('# of Cells')
ax.set_title(f'Jensen-Shannon Divergence: {np.nanmean(js_s):01f}')
# ax.set_xticks(np.arange(1,8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

