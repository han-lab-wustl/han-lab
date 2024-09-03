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
    
looping_cell_iind = []
looping_cell_prop = []
radian_alignment = {}
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
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
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

        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']           
            goal_cells_circ = intersect_arrays(*com_goal)
            coms_goal_cells_circ = coms_correct[:, goal_cells_circ]-np.pi
            coms_goal_cells_circ_av = np.mean(coms_goal_cells_circ,axis=0)
            far_remap = 30*(2*np.pi/track_length) # cm converted to rad
            goal_cells_far_remap = goal_cells_circ#[coms_goal_cells_circ_av>far_remap]
            colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod']
            for gc in goal_cells_far_remap:
                bin_size_rad = 3*(2*np.pi/track_length)
                fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,7))
                for ep in range(len(coms_correct)):
                    axes[0].plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],
                                linewidth='3')
                    axes[1].plot(tcs_correct[ep,gc,:], linewidth='3',
                    label=f'rewloc {rewlocs[ep]:.1f}', color=colors[ep])
                    axes[1].axvline(track_length/bin_size/2, color='k', linestyle='--',linewidth='3')
                    axes[0].axvline((rewlocs[ep]-rewsize/2)/bin_size, color=colors[ep], linestyle='--',linewidth='3')
                    axes[0].set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                    axes[1].set_xlabel(f'Radian position ($\theta$)')
                    axes[0].set_xlabel('Absolute position (cm)')
                    axes[1].set_ylabel('Fc3')
                    tics = np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1)
                    axes[1].set_xticks(np.arange(0,100,len(tics)))
                    axes[1].set_xticklabels(tics)
                    axes[0].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,10))
                    axes[0].set_xticklabels(np.arange(0,track_length+bin_size*10,bin_size*10).astype(int))
                    # axes[1].set_xticklabels(tics)
                    axes[0].spines[['top','right']].set_visible(False)
                    axes[1].spines[['top','right']].set_visible(False)
                axes[0].legend()
                plt.savefig(os.path.join(savedst, f'loopcell_{gc}.png'),bbox_inches='tight')
                pdf.savefig(fig)                                                                            
pdf.close()
#%%