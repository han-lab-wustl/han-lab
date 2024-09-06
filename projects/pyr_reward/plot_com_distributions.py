"""get coms to make histograms of place vs. reward-distance coms
"""

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype, get_success_failure_trials, consecutive_stretch, intersect_arrays
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\retreat_2024'
savepth = os.path.join(savedst, 'looping_cells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

coms = {}
#%%
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]==-1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
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
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]        # set vars
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))
        # exclude last ep if too little trials
        lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
        if lastrials<8:
            eps = eps[:-1]
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        bin_size=3 # cm
        # get abs dist tuning 
        tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
            Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,velocity_filter=True)
        coms_correct_abs_rel = np.array([xx-(rewlocs[ep]-rewsize/2) for ep,xx in enumerate(coms_correct_abs)])
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']                   
            coms_correct = coms_correct-np.pi
            # get goal cells across all epochs        
            goal_cells = intersect_arrays(*com_goal)

            coms[f'{animal}_{day:03d}_index{ii:03d}'] = [coms_correct_abs_rel, coms_correct, goal_cells]
#%%
# plot average coms in all cells / vs. goal cells
coms_goal_abs = []; coms_all_abs = []
for k,com in coms.items():
    coms_correct_abs_rel, coms_correct, goal_cells = com
    coms_goal_abs.append(np.ravel(coms_correct_abs_rel[:,goal_cells]))
    coms_all_abs.append(np.ravel(coms_correct_abs_rel[:,[xx for xx in range(coms_correct_abs_rel.shape[1]) if xx not in goal_cells]]))
#%%    
fig, axes = plt.subplots(1,2,figsize = (12,7), sharex=True)
axes[0].hist(np.concatenate(coms_all_abs), color='slategray')
axes[1].hist(np.concatenate(coms_goal_abs))
axes[1].set_xlabel('Center-of-mass rel. dist. rew.(cm)')
axes[1].set_ylabel('# cells')
axes[0].spines[['top','right']].set_visible(False)
axes[1].spines[['top','right']].set_visible(False)
axes[0].axvline(0, color='k', linestyle='--', linewidth = 3)
axes[1].axvline(0, color='k', linestyle='--', linewidth = 3)

axes[0].set_title('Other pyramidal cells')
axes[1].set_title('Reward-distance cells')
plt.savefig(os.path.join(savedst, 'hist_of_coms.png'), bbox_inches='tight')
plt.savefig(os.path.join(savedst, 'hist_of_coms.svg'))