"""
visualize place cells that loop their activity
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians, get_success_failure_trials, consecutive_stretch, intersect_arrays
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'looping_cells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
    
looping_cell_iind = []
looping_cell_prop = []
radian_alignment = {}
#%%
for ii in range(150,len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal!='e217':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'tuning_curves_late_trials', 'stat'])
        VR = fall['VR'][0][0][()]
        try:
            stat = np.array([fall['stat'][0][ii][()] for ii in range(fall['stat'][0].shape[0])])
            skew = np.array([xx[0][0] for xx in np.squeeze(np.hstack(stat['skew']))])
        except: # accept old iscell format?
            stat = np.array([fall['stat'][ii][()] for ii in range(fall['stat'].shape[0])])
            skew = np.array([xx['skew'][0][0][0][0] for xx in np.squeeze(np.hstack(stat))])
        # filter out skewed cells
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))
        # exclude last ep if too little trials
        lastrials = np.unique(trialnum[eps[(len(eps)-2)]:eps[(len(eps)-1)]])[-1]
        if lastrials<8:
            eps = eps[:-1]
        tcs_late = fall['tuning_curves_late_trials'][0]
        goal_window = 30
        coms = np.array([np.hstack(xx) for xx in fall['coms'][0]])
        coms_rewrel = np.array([com-rewlocs[ii] for ii, com in enumerate(coms)])                 
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # get goal cells across all epochs        
        goal_cells_absolute_dist = intersect_arrays(*com_goal)
        rad = [] # get radian coordinates
        # same as giocomo preprint - worked with gerardo
        for i in range(len(eps)-1):
            y = ybinned[eps[i]:eps[i+1]]
            rew = rewlocs[i]-rewsize/2
            # convert to radians and align to rew
            rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
        rad = np.concatenate(rad)
        track_length_rad = track_length*(2*np.pi/track_length)
        bins = 90
        bin_size=track_length_rad/bins

        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3'])
        Fc3 = fall_fc3['Fc3']
        skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew_mask = skew_filter>2
        tcs_late = np.array([xx[skew_mask,:] for xx in tcs_late])
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        Fc3 = Fc3[:, skew_mask] # remove skewed cells
        rates, tcs_late_circ, coms_circ = make_tuning_curves_radians(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size)
        tcs_late_circ = np.array(tcs_late_circ); coms_circ = np.array(coms_circ)            
        goal_window = 30*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_circ])
        perm = list(combinations(range(len(coms_circ)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # get goal cells across all epochs        
        goal_cells_circ = intersect_arrays(*com_goal)
        coms_goal_cells_circ = coms_circ[:, goal_cells_circ]-np.pi
        coms_goal_cells_circ_av = np.mean(coms_goal_cells_circ,axis=0)
        far_remap = 30*(2*np.pi/track_length) # cm converted to rad
        goal_cells_far_remap = goal_cells_circ[coms_goal_cells_circ_av>far_remap]
        colors = ['navy', 'red', 'green', 'k','darkorange']
        for gc in goal_cells_far_remap:
            bin_size = 3
            fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            for ep in range(len(coms_circ)):
                axes[0].plot(tcs_late[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                axes[1].plot(tcs_late_circ[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                axes[1].axvline((bins/2), color='k')
                axes[0].axvline(rewlocs[ep]/bin_size, color=colors[ep])
                axes[0].set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                # ax.set_xticks(np.arange(0,bins+1,10))
                # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),3))
                axes[1].set_xlabel('Radian position (centered at start of rew loc)')
                axes[1].set_ylabel('Fc3')
            axes[0].legend()
            pdf.savefig(fig)
            plt.close(fig)        
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_late_circ, coms_circ]
pdf.close()
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 
#%%