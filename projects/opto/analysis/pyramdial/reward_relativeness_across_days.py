
"""
zahra
june 2024
visualize reward-relative cells across days
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations
from itertools import chain
import matplotlib.backends.backend_pdf
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\thesis_proposal'
savepth = os.path.join(savedst, 'reward_relative_across_days.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = "Z:\saved_datasets\radian_tuning_curves_reward_cell.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates = []
total_cells = []
radian_alignment = {}
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal!='e217':
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
        tcs_early = []; tcs_late = []        
        Fc3 = fall['Fc3']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
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
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_late, coms = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']
        else:# remake tuning curves relative to reward        
            # takes time
            tcs_late, coms = make_tuning_curves_radians(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)
            tcs_late = np.array(tcs_late); coms = np.array(coms)
        goal_window = 30*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms])
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms[0])
        goal_cell_prop.append(goal_cell_p)
        num_epochs.append(len(coms))
        colors = ['navy', 'red', 'green', 'k','darkorange']
        for gc in goal_cells:
            fig, ax = plt.subplots()
            for ep in range(len(coms)):
                ax.plot(tcs_late[ep][gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins/2), color='k')
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                ax.set_xticks(np.arange(0,bins+1,10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),3))
                ax.set_xlabel('Radian position (centered at start of rew loc)')
                ax.set_ylabel('Fc3')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iterations
        num_iterations = 1000
        shuffled_dist = np.zeros((num_iterations))
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms[ii].shape[0])) for ii in range(1, len(coms))]
            [random.shuffle(shuf) for shuf in shufs]
            com_shufs = np.zeros_like(coms)
            com_shufs[0,:] = coms[0]
            com_shufs[1:1+len(shufs),:] = [coms[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-rewlocs_shuf[ii] for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf = intersect_arrays(*com_goal)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        print(p_value)
        total_cells.append(len(coms[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_late, coms]

pdf.close()

# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(radian_alignment, fp) 