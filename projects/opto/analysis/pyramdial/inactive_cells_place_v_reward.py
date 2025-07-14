
"""
zahra's analysis for initial com and enrichment of pyramidal cell data
updated aug 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math,  matplotlib as mpl, matplotlib.backends.backend_pdf
from itertools import combinations, chain

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.analysis.pyramdial.placecell import find_differentially_inactivated_cells, find_differentially_activated_cells
mpl.rcParams['svg.fonttype'] = 'none'; mpl.rcParams["xtick.major.size"] = 10; mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,pairwise_distances,\
    get_radian_position_first_lick_after_rew, get_rewzones, cosine_sim_ignore_nan
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.stats import spearmanr

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
#%% - re-run dct making
# uses matlab tuning curves
dcts = []
maindct={}
cm_window=20
# get inactive vs. active cell id and correlate with place vs. reward
for ii in range(len(conddf)):
    # define threshold to detect activation/inactivation
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal=='e145' or animal=='e139': pln=2 
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"    
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
    lick=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>1.2] # only keep cells with skew greateer than 2
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
        rewsize,ybinned,time,lick,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)  
    bin_size=3
    # abs position
    tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)
    # get cells that maintain their coms across at least 2 epochs
    place_window = 20 # cm converted to rad                
    perm = list(combinations(range(len(coms_correct_abs)), 2))     
    com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
    compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
    # get cells across all epochs that meet crit
    pcs = np.unique(np.concatenate(compc))

    lick_correct_abs, _,lick_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick,lick]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
    vel_correct_abs, _,vel_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel,forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
    goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
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
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal[ii])>0]
    com_goal=[com for com in com_goal if len(com)>0]
    if len(com_goal)>0:
        goal_cells = np.unique(np.concatenate(com_goal)).astype(int)     
        # pcs that are not goal cells
        pcs = [xx for xx in pcs if xx not in goal_cells]  
        # Find differentially inactivated cells
        if conddf.optoep.values[dd]<2: 
            eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
        comp = [eptest-2,eptest-1] # eps to compare    
        threshold=5; bin_size=track_length_dt/bins_dt
        differentially_inactivated_cells = find_differentially_inactivated_cells(tcs_correct[eptest-2,:,:], tcs_correct[eptest-1,:,:], threshold, bin_size)
        differentially_activated_cells = find_differentially_activated_cells(tcs_correct[eptest-2,:,:], tcs_correct[eptest-1,:,:], threshold, bin_size)
        affected_gc = [xx for xx in np.concatenate([differentially_inactivated_cells,differentially_activated_cells]) if xx in goal_cells]
        affected_pc = [xx for xx in np.concatenate([differentially_inactivated_cells,differentially_activated_cells]) if xx in pcs]
        total_affected=np.concatenate([differentially_inactivated_cells,differentially_activated_cells])
        if len(total_affected)>0:
            print('####################################\n')
            print(f'% of affected cells that are rew cells: {len(affected_gc)/len(total_affected)*100}')
            print(f'% of affected cells that are place cells: {len(affected_pc)/len(total_affected)*100}')
            print('\n####################################')
        maindct[f'{animal}_{day}']=[pcs,goal_cells,differentially_inactivated_cells,differentially_activated_cells]
    
with open(r'Z:\dcts_active_v_inactive.p', "wb") as fp:   #Pickling
    pickle.dump(maindct, fp)   

#%%
