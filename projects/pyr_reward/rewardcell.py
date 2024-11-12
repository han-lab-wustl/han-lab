
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

import numpy as np, random, re, os, scipy, pandas as pd, sys, cv2
import matplotlib.pyplot as plt, matplotlib
from itertools import combinations, chain
from scipy.spatial import distance
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype
from projects.opto.behavior.behavior import get_success_failure_trials

def extract_data_nearrew(ii,params_pth,animal,day,bins,radian_alignment,
                radian_alignment_saved,goal_window_cm,pdf,
        num_iterations = 1000):
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins
    success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
    rate = success/total_trials
    # added to get anatomical info
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
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av,pdist = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # find correct trials within each epoch!!!!
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size)          
    fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
    ops = fall_stat['ops']
    stat = fall_stat['stat']
    meanimg=np.squeeze(ops)[()]['meanImg']
    s2p_iind = np.arange(stat.shape[1])
    s2p_iind_filter = s2p_iind[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    s2p_iind_filter = s2p_iind_filter[skew>2]
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    # only get cells near reward        
    perm = list(combinations(range(len(coms_correct)), 2))     
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])                
    # tuning curves that are close to each other across epochs
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com near but after goal
    com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0))] for com in com_goal]
    # get goal cells across all epochs        
    goal_cells = intersect_arrays(*com_goal)
    # also, have to be active in all epochs
    goal_cells = [xx for xx in goal_cells if sum(np.nanmax(tcs_correct[:,xx,:],axis=1)>.05)==len(coms_correct)]
    
    s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
    goal_cell_iind=goal_cells
    goal_cell_p=len(goal_cells)/len(coms_correct[0])
    goal_cell_prop=[goal_cells_p_per_comparison,goal_cell_p]
    num_epochs=len(coms_correct)
    # get x y coords!!
    meanimgm = np.zeros_like(meanimg)        
    stat = np.squeeze(stat)        
    fig,ax = plt.subplots()
    ax.imshow(meanimg, cmap='Greys_r')
    centersgc = []        
    cmap = ['k','yellow']  # Choose your preferred colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)
    cmap.set_under('none')
    for ii,gc in enumerate(s2p_iind_goal_cells):
            ypix=stat[gc][0][0][0][0]    
            xpix=stat[gc][0][0][1][0]     
            coords = np.column_stack((xpix, ypix))  
            mask,cmask,center=create_mask_from_coordinates(coords, 
                    meanimg.shape)                         
            ax.imshow(cmask,cmap=cmap,vmin=1)
            # plot cell id
            ax.text(center[0]+5,center[1]+5,goal_cells[ii],color='w') 
            centersgc.append(center)   
            
    points = np.array(centersgc)
    dist = pairwise_distances(points)
    # Exclude self-pairs (distances to the same point)
    # We do this by flattening the distance matrix and excluding zero distances
    non_self_distances = dist[np.triu_indices_from(dist, k=1)]
    # Compute the average distance
    pdist = non_self_distances

    ax.axis('off')
    fig.suptitle(f'animal: {animal}, day: {day}')
    pdf.savefig(fig)
    plt.close(fig)
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(0,bins+1,20))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
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
        # get goal cells across all epochs
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # (com near goal)
        com_goal =  [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=goal_window) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=0))] for com in com_goal]

        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        goal_cells_shuf = intersect_arrays(*com_goal); shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null=[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells=len(coms_correct[0])
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                    com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av,pdist]
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,perm,goal_cell_prop,num_epochs,goal_cell_null

def acc_corr_cells(forwardvel, timedFF, pln, dFF, eps):
    acccells_per_ep = []
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep], eps[ep+1])
        # get acceleration correlated cells across all trials
        # get goal cells across all epochs      
        acc = np.diff(forwardvel[eprng])/np.diff(timedFF[eprng])
        accdf = pd.DataFrame({'acc': acc})
        acc = np.hstack(accdf.rolling(100).mean().fillna(0).values)
        # cells correlated with acc
        # Calculate phase-shifted correlation
        max_shift = int(np.ceil(31.25/(pln+1)))  # You can adjust the max shift based on your data
        # ~ 1 s phase shifts
        rshiftmax = []
        for i in range(dFF.shape[1]):
            dff = dFF[eprng[:-1],i]
            dff[np.isnan(dff)]=0 # nan to 0
            r=phase_shifted_correlation(acc, dff, max_shift)
            rshiftmax.append(np.max(r))
        # only get top 10% for now
        acccells = np.where(np.array(rshiftmax)>np.nanquantile(rshiftmax,.90))[0]
        acccells_per_ep.append(acccells)        
        
    return acccells_per_ep

def phase_shifted_correlation(acceleration, neural_activity, max_shift):
    """
    Calculate phase-shifted correlation between acceleration and neural activity.
    
    Parameters:
        acceleration (np.array): The acceleration data.
        neural_activity (np.array): The neural activity data.
        max_shift (int): The maximum shift (in samples) to apply for phase-shifting.
        
    Returns:
        shifts (np.array): Array of shift values.
        correlations (np.array): Correlation values for each shift.
    """
    
    # Ensure the signals have the same length
    assert len(acceleration) == len(neural_activity), "Signals must have the same length"
    
    shifts = np.arange(-max_shift, max_shift + 1, 5)
    correlations = np.zeros(len(shifts))
    
    for i, shift in enumerate(shifts):
        if shift < 0:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[shift:] = 0
        else:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[:shift] = 0
        
        # Calculate the correlation for this shift
        correlation, _ = scipy.stats.pearsonr(acceleration, shifted_neural_activity)
        correlations[i] = correlation
    return correlations
    
    
def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = [x[:break_point[0]]]
    for i in range(1, len(break_point)):
        y.append(x[break_point[i - 1] + 1:break_point[i]])
    y.append(x[break_point[-1] + 1:])
    
    return y 

def perireward_binned_activity_early_late(dFF, rewards, timedFF, trialnum, range_val, binsize,
                                          early_trial=2, late_trial=5):
    """Adapts code to align dFF or pose data to rewards within a certain window on a per-trial basis,
    only considering trials with trialnum > 3. Calculates activity for the first 5 and last 5 trials separately.

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        trialnum (_type_): array denoting the trial number per frame
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        dict: Contains mean and normalized activity for first and last 5 trials.
    """
    # Filter rewards for trialnum > 3
    Rewindx = np.where(rewards & (trialnum > 3))[0]
    
    # Calculate separately for first 5 trials
    first_trials = Rewindx[:early_trial]
    last_trials = Rewindx[-late_trial:]
    
    def calculate_activity(TrialIndexes):
        rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(TrialIndexes)))*np.nan
        for rr in range(0, len(TrialIndexes)):
            current_trial = trialnum[TrialIndexes[rr]]
            rewtime = timedFF[TrialIndexes[rr]]
            currentrewchecks = np.where((timedFF > rewtime - range_val) & 
                                        (timedFF <= rewtime + range_val) & 
                                        (trialnum == current_trial))[0]
            currentrewcheckscell = consecutive_stretch(currentrewchecks)
            currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx) > 0]
            currentrewcheckscell = np.array(currentrewcheckscell)
            currentrewardlogical = np.array([sum(TrialIndexes[rr] == x).astype(bool) for x in currentrewcheckscell])
            val = 0
            for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
                val = bin_val + 1
                currentidxt = np.where((timedFF > (rewtime - range_val + (val * binsize) - binsize)) & 
                                       (timedFF <= rewtime - range_val + val * binsize) &
                                    (trialnum == current_trial))[0]
                checks = consecutive_stretch(currentidxt)
                checks = [list(xx) for xx in checks]
                if len(checks[0]) > 0:
                    currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                    for x in checks])
                    for i, cidx in enumerate(currentidxlogical):
                        cidx = [bool(xx) for xx in cidx]
                        if sum(cidx) > 0:
                            checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                            rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

        meanrewdFF = np.nanmean(rewdFF, axis=1)
        normmeanrewdFF = (meanrewdFF - np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
        normrewdFF = np.array([(xx - np.min(xx)) / ((np.max(xx) - np.min(xx))) for xx in rewdFF.T])

        return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF
    
    normmeanrewdFF_first, meanrewdFF_first, normrewdFF_first, rewdFF_first = calculate_activity(first_trials)
    normmeanrewdFF_last, meanrewdFF_last, normrewdFF_last, rewdFF_last = calculate_activity(last_trials)

    return {
        'first_5': {
            'normmeanrewdFF': normmeanrewdFF_first,
            'meanrewdFF': meanrewdFF_first,
            'normrewdFF': normrewdFF_first,
            'rewdFF': rewdFF_first
        },
        'last_5': {
            'normmeanrewdFF': normmeanrewdFF_last,
            'meanrewdFF': meanrewdFF_last,
            'normrewdFF': normrewdFF_last,
            'rewdFF': rewdFF_last
        }
    }

def perireward_binned_activity(dFF, rewards, timedFF, 
        trialnum, range_val, binsize):
    """adaptation of gerardo's code to align IN BOTH TIME AND POSITION, dff or pose data to 
    rewards within a certain window on a per-trial basis, only considering trials with trialnum > 3

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        trialnum (_type_): array denoting the trial number per frame
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    Rewindx = np.where(rewards & (trialnum > 3))[0]  # Filter rewards for trialnum > 3
    rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(Rewindx)))*np.nan

    for rr in range(0, len(Rewindx)):
        current_trial = trialnum[Rewindx[rr]]
        rewtime = timedFF[Rewindx[rr]]
        currentrewchecks = np.where((timedFF > rewtime - range_val) & 
                                    (timedFF <= rewtime + range_val) & 
                                    (trialnum == current_trial))[0]
        currentrewcheckscell = consecutive_stretch(currentrewchecks)  # Get consecutive stretch of reward ind
        # Check for missing vals
        currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx) > 0]
        currentrewcheckscell = np.array(currentrewcheckscell)  # Reformat for Python
        currentrewardlogical = np.array([sum(Rewindx[rr] == x).astype(bool) for x in currentrewcheckscell])
        val = 0
        for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
            val = bin_val + 1
            currentidxt = np.where((timedFF > (rewtime - range_val + (val * binsize) - binsize)) & 
                                   (timedFF <= rewtime - range_val + val * binsize) &
                                (trialnum == current_trial))[0]
            checks = consecutive_stretch(currentidxt)
            checks = [list(xx) for xx in checks]
            if len(checks[0]) > 0:
                currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                for x in checks])
                for i, cidx in enumerate(currentidxlogical):
                    cidx = [bool(xx) for xx in cidx]
                    if sum(cidx) > 0:
                        checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                        rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

    meanrewdFF = np.nanmean(rewdFF, axis=1)
    normmeanrewdFF = (meanrewdFF - np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    normrewdFF = np.array([(xx - np.min(xx)) / ((np.max(xx) - np.min(xx))) for xx in rewdFF.T])
    
    return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF

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

def get_tracking_vars_wo_dff(params_pth):                
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
        'ybinned', 'VR', 'forwardvel', 
        'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF'])
    # to remove skew cells
    dFF = fall['dFF']
    suite2pind = np.arange(fall['iscell'][:,0].shape[0])
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


def get_tracking_vars(params_pth):                
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
        'ybinned', 'VR', 'forwardvel', 
        'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF'])
    # to remove skew cells
    dFF = fall['dFF']
    suite2pind = np.arange(fall['iscell'][:,0].shape[0])
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

    return tracked_lut, days

def get_shuffled_goal_cell_indices(rewlocs, coms_correct, goal_window, suite2pind_remain,
                num_iterations = 1000):
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cells_shuf_s2pind = []; coms_rewrels = []
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
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 
                                            1+len(shufs))]
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
        coms_rewrels.append(coms_rewrel)
    return goal_cells_shuf_s2pind, coms_rewrels

def get_reward_cells_that_are_tracked(tracked_lut, goal_cells_s2p_ind, 
        animal, day,  suite2pind_remain):
    tracked_rew_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in goal_cells_s2p_ind]
    tracked_cells_that_are_rew_pyr_id = tracked_lut[day].values[tracked_rew_cell_ind]
    rew_cells_that_are_tracked_iind = np.array([np.where(suite2pind_remain==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id])
    
    return tracked_rew_cell_ind, rew_cells_that_are_tracked_iind
                
def create_mask_from_coordinates(coordinates, image_shape):
    """Creates a mask from a list of coordinates.

    Args:
        coordinates: A list of (x, y) coordinates defining the region to mask.
        image_shape: A tuple (height, width) defining the shape of the image.

    Returns:
        A numpy array representing the mask, where 1 indicates the masked region.
    """

    mask = np.zeros(image_shape, dtype=np.uint8)
    height,width=image_shape

    # Create a polygon from the coordinates
    polygon = np.array(coordinates, dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2))

    # Fill the polygon with 1s
    cv2.fillPoly(mask, [polygon], 1)
    # Find contours of the filled mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the contours
    contour_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the detected contours on the mask
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), 1)
    # Calculate the moments of the contours to find the center
    M = cv2.moments(contour_mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return mask,contour_mask,(cX,cY)
# Function to compute pairwise distances
def pairwise_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distance.euclidean(points[i], points[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances
