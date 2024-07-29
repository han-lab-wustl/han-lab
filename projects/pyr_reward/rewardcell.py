
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

import numpy as np
def get_radian_position(eps,ybinned,rewlocs,track_length):
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
    dist_to_rew.append(coms_rewrel)
    # get goal cells across all epochs        
    goal_cells = intersect_arrays(*com_goal)
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    
    return goal_window, goal_cells, perm, goal_cells_p_per_comparison,goal_cell_p

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

def get_trialtypes(trialnum, rewards, coms_correct, eps):
    
    per_ep_trialtypes = []
    
    for ep in range(len(coms_correct)):
        trialnum_ep = np.array(trialnum)[eps[i]:eps[i+1]]
        rewards_ep = np.array(rewards)[eps[i]:eps[i+1]]
        unique_trials = np.unique(trialnum_ep)
        
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
                                
        total_trials = np.sum(unique_trials >= 3)
        per_ep_trialtypes.append([init_fails, first_correct, correct_trials_besides_first, 
                inbtw_fails, total_trials])
        
    return per_ep_trialtypes
    
    
    