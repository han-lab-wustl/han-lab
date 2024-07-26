
"""
zahra
june 2024
visualize reward-relative cells across days
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_trial_by_trial, intersect_arrays, make_tuning_curves_radians_by_trialtype
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'trial_by_trial_rew_cells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
radian_alignment = {}
#%%
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal!='e217' and conddf.optoep.values[ii]<2:
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = [] # get radian coordinates
        # same as giocomo preprint - worked with gerardo
        for i in range(len(eps)-1):
            y = ybinned[eps[i]:eps[i+1]]
            rew = rewlocs[i]-rewsize/2
            # convert to radians and align to rew
            rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
        rad = np.concatenate(rad)
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
        rates_all.append(success/total_trials)
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # takes time
            fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
            Fc3 = fall_fc3['Fc3']
            dFF = fall_fc3['dFF']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
            # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            # skew_mask = skew_filter>2
            Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)          
        goal_window = 30*(2*np.pi/track_length) # cm converted to rad
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
        num_epochs.append(len(coms_correct))
        # plot trial by trial rew cells
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3'])
        Fc3 = fall_fc3['Fc3']            
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]            
        F_remap = Fc3[:, goal_cells]
        lick = fall['licks']
        lick = np.squeeze(lick)
        trialstates, licks_trial_by_trial, tcs_trial_by_trial, 
        coms_trial_by_trial = make_tuning_curves_radians_trial_by_trial(eps,rewlocs,
                lick,ybinned,rad,F_remap,trialnum,
            rewards,forwardvel,rewsize,bin_size)
        plt.rc('font', size=10)
        for gc in range(len(goal_cells)):                        
            fig,axes = plt.subplots(nrows=len(eps)-1,ncols=2,sharex=True)
            for ep in range(len(eps)-1):
                ax = axes[ep,0]
                mask = ~(trialstates[ep]==0)
                mask = np.ones_like(licks_trial_by_trial[ep]).T*mask
                sns.heatmap(licks_trial_by_trial[ep], mask=mask.T, cmap='Reds', cbar=False,ax=ax)
                mask = ~(trialstates[ep]==1)
                mask = np.ones_like(licks_trial_by_trial[ep]).T*mask
                sns.heatmap(licks_trial_by_trial[ep], mask=mask.T, cmap='Greens', cbar=False, ax=ax)
                if ep==len(eps)-1: ax.set_title('licks')
                ax = axes[ep,1]
                mask = ~(trialstates[ep]==0)
                mask = np.ones_like(tcs_trial_by_trial[ep][gc]).T*mask
                sns.heatmap(tcs_trial_by_trial[ep][gc], 
                    mask=mask.T,cmap='Reds',ax=ax)
                mask = ~(trialstates[ep]==1)
                mask = np.ones_like(tcs_trial_by_trial[ep][gc]).T*mask
                sns.heatmap(tcs_trial_by_trial[ep][gc],  
                    mask=mask.T,cmap='Greens', ax=ax)
                if ep==len(eps)-1: 
                    ax.set_title('activity')
                    ax.set_xlabel('Position bin')
            fig.suptitle(f'{animal}, day {day}, remap cell # {gc}')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iterations
        num_iterations = 1000
        shuffled_dist = np.zeros((num_iterations))
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
            [random.shuffle(shuf) for shuf in shufs]
            com_shufs = np.zeros_like(coms_correct)
            com_shufs[0,:] = coms_correct[0]
            com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms_correct)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf = intersect_arrays(*com_goal)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        print(p_value)
        total_cells.append(len(coms_correct[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_late, coms, 
                                p_value, trialstates, licks_trial_by_trial, 
                                tcs_trial_by_trial, coms_trial_by_trial]

pdf.close()
#%%