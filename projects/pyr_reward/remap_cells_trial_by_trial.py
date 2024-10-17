
"""
zahra
oct 2024
goal cells per trial
"""
#%%

# Import necessary libraries
import numpy as np,h5py,scipy.io,matplotlib.pyplot as plt,sys,matplotlib
import matplotlib as mpl,pandas as pd,os,pickle,seaborn as sns
from itertools import combinations
from collections import Counter
import matplotlib.backends.backend_pdf as pdf_backend
from scipy import stats
plt.close('all')
# Configure matplotlib to use the Arial font and s et tick sizes
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
# Add custom paths (adjust according to your directory structure)
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
# Import custom functions
from placecell import (
    make_tuning_curves_radians_by_trialtype, 
    intersect_arrays, 
    make_tuning_curves_radians_trial_by_trial
)
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'trial_by_trial.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto_20240919.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)

cm_window=10 #cm
# Initialize lists to store results
dist_to_rew = []
goal_cell_iind = []
goal_cell_prop = []
num_epochs = []
#%%
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]==-1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
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
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))
        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        if 'bordercells' in fall.keys():
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        else:
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
            dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys() and animal!='e145':
            tcs_correct, coms_correct, tcs_fail, coms_fail, com_goal, \
                goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        elif animal=='e145': # something weird about e145 saved df
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size)
        else:# remake tuning curves relative to reward        
            # takes time
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms_correct[0])
        goal_cell_prop.append(goal_cell_p)
        num_epochs.append(len(coms))
        # plot trial by trial rew cells
        F_remap = Fc3[:, goal_cells]
        lick = fall['licks']
        lick = np.squeeze(lick)
        trialstates, licks_trial_by_trial, tcs_trial_by_trial,\
        coms_trial_by_trial = make_tuning_curves_radians_trial_by_trial(eps,rewlocs,
            lick,ybinned,rad,F_remap,trialnum,
            rewards,forwardvel,rewsize,bin_size)
        plt.rc('font', size=16)
        for gc,orggc in enumerate(goal_cells):
            fig, axes = plt.subplots(figsize=(14, 10), nrows=len(eps)-1, ncols=2, sharex=True, gridspec_kw={'width_ratios': [1, 3]})

            for ep in range(len(eps) - 1):
                # Plot licks
                ax = axes[ep, 0]
                for state, color in zip([0, 1, -1], ['Reds', 'Greens', 'Blues']):
                    mask = ~(trialstates[ep] == state)
                    mask = np.ones_like(licks_trial_by_trial[ep]).T*mask
                    sns.heatmap(licks_trial_by_trial[ep], mask=mask.T, cmap=color, cbar=False, ax=ax)
                ax.axvline(bins/2, linestyle='--', color='k', linewidth=2)
                if ep == 0:
                    ax.set_title('Licks')
                    ax.set_ylabel('Trial #')
                ax.set_xticks(np.arange(0, bins+1, 10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5), 2), fontsize=10, rotation=45)

                # Plot ΔF/F
                ax = axes[ep, 1]
                for state, color in zip([0, 1, -1], ['Reds', 'Greens', 'Blues']):
                    mask = ~(trialstates[ep] == state)
                    mask = np.ones_like(tcs_trial_by_trial[ep][gc]).T*mask
                    sns.heatmap(tcs_trial_by_trial[ep][gc], mask=mask.T, cmap=color, ax=ax, cbar_kws={"pad": 0.01})
                ax.axvline(bins/2, linestyle='--', color='k', linewidth=2)
                if ep == 0:
                    ax.set_title('ΔF/F')
                    ax.set_xlabel('Radian position θ')
                ax.set_xticks(np.arange(0, bins+1, 10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5), 2), fontsize=12, rotation=45)

            fig.suptitle(f'{animal},day {day},remap cell # {orggc}')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

pdf.close()
