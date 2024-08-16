
"""
zahra
july 2024
get rew-relative cells in different trial conditions

1st probe trial
other 2 probe trials
initial failed trials of an epoch
failed trials after successful trails
1st correct trial
correct trials
"""
#%%

import numpy as np, random, re, os, scipy, pandas as pd
from itertools import combinations, chain
from placecell import intersect_arrays

def get_radian_position(eps,ybinned,rewlocs,track_length,rewsize):
    rad = [] # get radian coordinates
    # same as giocomo preprint - worked with gerardo
    for i in range(len(eps)-1):
        y = ybinned[eps[i]:eps[i+1]]
        rew = rewlocs[i]-rewsize/2
        # convert to radians and align to rew
        rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
    rad = np.concatenate(rad)
    return rad

def get_goal_cells(track_length,coms_correct,window=30):
    goal_window = window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2))     
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    
    # get goal cells across all epochs        
    goal_cells = intersect_arrays(*com_goal)
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    
    return goal_window, goal_cells, perm, goal_cells_p_per_comparison,goal_cell_p, coms_rewrel

def goal_cell_shuffle(rewlocs, coms_correct, goal_window, num_iterations = 1000):
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        goal_cells_shuf = intersect_arrays(*com_goal); 
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    
    return goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist

def plot_rew_cell():
    colors = ['navy', 'red', 'green', 'k','darkorange']
    for gc in goal_cells:
        fig, ax = plt.subplots()
        for ep in range(len(coms_correct)):
            ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
        ax.axvline((bins/2), color='k')
        ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
        ax.set_xticks(np.arange(0,bins+1,10))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),2))
        ax.set_xlabel('Radian position (centered at start of rew loc)')
        ax.set_ylabel('Fc3')
        ax.legend()
        ax.spines[['top','right']].set_visible(False)

def get_trialtypes(trialnum, rewards, ybinned, coms_correct, eps):
    
    per_ep_trialtypes = []
    
    for i in range(len(coms_correct)):
        eprng = np.arange(eps[i],eps[i+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        trialnum_ep = np.array(trialnum)[eprng]        
        rewards_ep = np.array(rewards)[eprng]
        unique_trials = np.array([xx for xx in np.unique(trialnum_ep) if np.sum(trialnum_ep==xx)>100])
        
        init_fails = [] # initial failed trials
        first_correct = []
        correct_trials_besides_first = []  # success trials
        inbtw_fails = []  # failed trials

        for tt, trial in enumerate(unique_trials):
            if trial >= 3:  # Exclude probe trials
                trial_indices = trialnum_ep == trial
                if np.any(rewards_ep[trial_indices] == 1):                    
                    if trial>3:
                        correct_trials_besides_first.append(trial)
                    else:
                        first_correct.append(trial)
                elif trial==3:                    
                    init_fails.append(trial)
                else:
                    inbtw_fails.append(trial)
                                
        total_trials = np.sum(unique_trials)
        per_ep_trialtypes.append([init_fails, first_correct, correct_trials_besides_first, 
                inbtw_fails, total_trials])
        
    return per_ep_trialtypes
    

def get_days_from_cellreg_log_file(txtpth):
    # Specify the path to your text file
    # Read the file content into a string
    with open(txtpth, 'r') as file:
        data = file.read()

    # Split the data into lines
    lines = data.strip().split('\n')

    # Regular expression pattern to extract session number and day number
    pattern = r'Session (\d+) - .*_day(\d+)_'

    # List to hold the extracted session and day numbers
    sessions = []; days = []

    # Extract session and day numbers using regex
    for line in lines:
        match = re.search(pattern, line)
        if match:
            session_number = match.group(1)
            day_number = match.group(2)
            sessions.append(int(session_number))
            days.append(int(day_number))

    return sessions, days

def find_log_file(pth):
    """for cell track logs

    Args:
        pth (_type_): _description_
    """
    # Find the first file that matches the criteria
    matching_file = None
    for filename in os.listdir(pth):
        if filename.startswith('logFile') and filename.endswith('.txt'):
            matching_file = filename
            break
    
    return matching_file

def get_tracking_vars(params_pth):                
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
        'ybinned', 'VR', 'forwardvel', 
        'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF'])
    # to remove skew cells
    dFF = fall['dFF']
    suite2pind = np.arange(dFF.shape[1])
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    suite2pind_remain = suite2pind[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    # we need to find cells to map back to suite2p indexes
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    suite2pind_remain = suite2pind_remain[skew>2]
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    # mainly for e145
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
    eps = np.append(eps, len(changeRewLoc))
    
    return dFF, suite2pind_remain, VR, scalingf, rewsize, ybinned, forwardvel, changeRewLoc,\
        rewards, eps, rewlocs, track_length

def get_tracked_lut(celltrackpth, animal, pln):
    
    tracked_lut = scipy.io.loadmat(os.path.join(celltrackpth, 
    rf"{animal}_daily_tracking_plane{pln}\Results\commoncells_once_per_week.mat"))
    tracked_lut = tracked_lut['commoncells_once_per_week'].astype(int)
    # CHANGE INDEX TO MATCH SUITE2P INDEX!! -1!!!
    tracked_lut = tracked_lut-1
    # find day match with session        
    txtpth = os.path.join(celltrackpth, rf"{animal}_daily_tracking_plane{pln}\Results")
    txtpth = os.path.join(txtpth, find_log_file(txtpth))
    sessions, days = get_days_from_cellreg_log_file(txtpth)    
    tracked_lut = pd.DataFrame(tracked_lut, columns = days)

    return tracked_lut

def get_shuffled_goal_cell_indices(rewlocs, coms_correct, goal_window, suite2pind_remain,
                num_iterations = 1000):
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cells_shuf_s2pind = []
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, 
                len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct)
        com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) 
                    for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & 
                (comr>-goal_window))[0] for comr in com_remap]            
        goal_cells = intersect_arrays(*com_goal)
        goal_cells_s2p_ind = suite2pind_remain[goal_cells]
        goal_cells_shuf_s2pind.append(goal_cells_s2p_ind)
    return goal_cells_shuf_s2pind, coms_rewrel

def get_reward_cells_that_are_tracked(tracked_lut, goal_cells_s2p_ind, 
        animal, day, tracked_rew_cell_inds, suite2pind_remain):
    tracked_rew_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in goal_cells_s2p_ind]
    tracked_rew_cell_inds[f'{animal}_{day:03d}'] = tracked_rew_cell_ind   
    tracked_cells_that_are_rew_pyr_id = tracked_lut[day].values[tracked_rew_cell_ind]
    rew_cells_that_are_tracked_iind = np.array([np.where(suite2pind_remain==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id])
    
    return tracked_rew_cell_ind, rew_cells_that_are_tracked_iind
                