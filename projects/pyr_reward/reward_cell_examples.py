
"""
zahra
pca on tuning curves of reward cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.cluster import KMeans
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'pre_post_rew_assemblies.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
tcs_rew = []
goal_cells_all = []
bins = 90
goal_window_cm=20
epoch_perm=[]
assembly_cells_all_an=[]
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
ii=152

day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if (animal!='e217') & (conddf.optoep.values[ii]<2):
    if animal=='e145' or animal=='e139': pln=2 
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
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
    licks=fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # 9/19/24
        # find correct trials within each epoch!!!!
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
    rewards,forwardvel,rewsize,bin_size)          
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # all cells 
    com_goal_postrew = com_goal
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
    plt.rc('font', size=16)
    #%%
    from matplotlib import colors
    tcs_correct_r = tcs_correct[:,np.unique(np.concatenate(com_goal))]
    tcs_fail_r = tcs_fail[:,np.unique(np.concatenate(com_goal))]
    coms_correct_r = coms_correct[:,np.unique(np.concatenate(com_goal))]
    # Assuming tcs_correct is a list of 2D arrays and coms_correct is defined
    # Determine the global min and max for normalization
    vmin = min(np.min(tcs) for tcs in tcs_correct_r)
    vmax = max(np.max(tcs) for tcs in tcs_correct_r)-.8
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Create subplots
    fig, axes = plt.subplots(ncols=len(tcs_correct_r),
                nrows=2, figsize=(11, 9),sharex=True, sharey=True)
    # Plot each subplot with shared normalization
    for kk, tcs in enumerate(tcs_correct_r):
        ax = axes[0,kk]
        im = ax.imshow(tcs[np.argsort(coms_correct_r[0])] ** 0.2, aspect='auto', norm=norm)
        ax.set_title(f'Correct trials, Epoch {kk + 1}')
        ax.axvline(bins / 2, color='w', linestyle='--')
        if kk==0: ax.set_ylabel('# Reward Cells')

    # Plot each subplot with shared normalization
    for kk, tcs in enumerate(tcs_fail_r):
        ax = axes[1,kk]
        im = ax.imshow(tcs[np.argsort(coms_correct_r[0])] ** 0.2, aspect='auto', norm=norm)
        ax.set_title(f'Incorrect trials, Epoch {kk + 1}')
        ax.axvline(bins / 2, color='w', linestyle='--')
        if kk==len(tcs_fail)-1:
            ax.set_xlabel('Reward-relative distance ($\Theta$)')  
            
    ax.set_xticks(np.arange(0,bins,30))
    ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='$\Delta$ F/F')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(os.path.join(savedst, 'tuning_curve_correct_v_incorrect_eg.svg'),bbox_inches='tight')
    #%%
    # correct only
    # Create subplots
    tcs_correct_r = tcs_correct[:,np.unique(np.concatenate(com_goal))]
    coms_correct_r = coms_correct[:,np.unique(np.concatenate(com_goal))]
    fig, axes = plt.subplots(ncols=len(tcs_correct),
                        figsize=(10, 5),sharex=True, sharey=True)
    # Plot each subplot with shared normalization
    gamma = .4
    for kk, tcs in enumerate(tcs_correct_r):
        ax = axes[kk]
        im = ax.imshow(tcs[np.argsort(coms_correct_r[0])] ** gamma, aspect='auto', norm=norm)
        ax.set_title(f'Correct trials, Epoch {kk + 1}')
        ax.axvline(bins / 2, color='w', linestyle='--')
        # draw line before and after rew at 'near' threshold
        line = (np.pi/4)/(2*np.pi/track_length)
        ax.axvline(line, color='y', linestyle='--')
        line = 90-line
        ax.axvline(line, color='y', linestyle='--')
        # track_length_rad = track_length*(2*np.pi/track_length)
        if kk==0: ax.set_ylabel('Reward cell ID #')
        if kk==len(tcs_correct)-1:
            ax.set_xlabel('Reward-relative distance ($\Theta$)')  
            
    ax.set_xticks(np.arange(0,bins,30))
    ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(os.path.join(savedst, 'tuning_curve_correct_eg.svg'),bbox_inches='tight')

#%%