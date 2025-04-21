
"""
zahra
get com histograms
april 2025
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays, make_tuning_curves
from rewardcell import get_radian_position_first_lick_after_rew,create_mask_from_coordinates,pairwise_distances,get_rewzones,\
    normalize_values
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#%%
goal_cm_window = 20
lasttr=8 # last trials
bins=90
coms_mean_rewrel = []
coms_mean_abs = []
com_spatial_tuned_all=[]
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks', 'putative_pcs'])
        putative_pcs=np.vstack(fall['putative_pcs'][0])
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
        lick = fall['licks'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            lick=lick[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                                    trialnum, track_length) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []; norm_pos = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            pos = normalize_values(ybinned[eprng], rewlocs[ep]-rewsize/2, rewlocs[ep]+rewsize/2, 
                    track_length)
            # values are a list of positions
            # b is the reward start location
            # c is the reward end location
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
            norm_pos.append(pos)
        rate=np.nanmean(np.array(rates))
        norm_pos=np.concatenate(norm_pos)
        
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
        # get tuning curves trial by trial and get calculate radians
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # 9/19/24
            # find correct trials within each epoch!!!!
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size)      
        # allocentric ref
        bin_size=track_length/bins 
        tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
            Fc3,trialnum,rewards,forwardvel,
            rewsize,bin_size)
        coms_rewrel_abs = np.array([xx-rewlocs[ii] for ii,xx in enumerate(coms_correct_abs)])
        # just get ep 1
        goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
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
        ######### REWARD
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        #only get perms with non zero cells
        perm=[p for ii,p in enumerate(perm) if len(com_goal[ii])>0]
        rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal[ii])>0]
        com_goal=[com for com in com_goal if len(com)>0]
        # get goal cells across all epochs   
        if len(com_goal)>0:
            goal_cells = intersect_arrays(*com_goal); 
        else:
            goal_cells=[]
        ########## PLACE
        # get cells that maintain their coms across at least 2 epochs
        place_window = 20 # cm 
        perm = list(combinations(range(len(coms_correct_abs)), 2))     
        com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
        # get cells across all epochs that meet crit
        pcs = np.unique(np.concatenate(compc))
        pcs_all = intersect_arrays(*compc)
        # get mean com for rew vs place
        pc_com_mean = np.nanmean(coms_rewrel_abs[:, pcs_all],axis=0)
        rew_com_mean = np.nanmean(coms_rewrel_abs[:, goal_cells],axis=0)
        coms_mean_rewrel.append(rew_com_mean)
        coms_mean_abs.append(pc_com_mean)

        ######## OTHER SPATIAL TUNED CELLS
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        # boolean with border cells bc thats how spatial info shuffle is done
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
        # skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
        # get tc again
        tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
        Fc3,trialnum,rewards,forwardvel,
        rewsize,bin_size)
        coms_rewrel_abs = np.array([xx-rewlocs[ii] for ii,xx in enumerate(coms_correct_abs)])
        # get com of cell in spatial tuned ep
        coms_spatial_tuned = []        
        for cll in range(putative_pcs.shape[1]):
            ep_st = np.where(putative_pcs[:,cll])[0]
            try:
                coms_spatial_tuned.append(coms_rewrel_abs[ep_st, cll])
            except Exception as e:
                print(e)
        coms_spatial_tuned=np.concatenate(coms_spatial_tuned)
        # add spatially tuned cells
        com_spatial_tuned_all.append(coms_spatial_tuned)


#%%
# plot com distributions
plt.rc('font', size=24) 
fig,axes = plt.subplots(ncols = 4,nrows = 1,figsize=(18,4),sharex=True,sharey=True)
ax=axes[0]
vmin=0
vmax = .01
ax.hist(np.concatenate(com_spatial_tuned_all),density=True,color='slategray')
ax.set_ylabel('Density of COM')
ax.set_title(f'Spatially tuned cells\n n={len(np.concatenate(com_spatial_tuned_all))} cells')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim([vmin,vmax])
ax=axes[1]
ax.hist(np.concatenate(coms_mean_rewrel),density=True,color='cornflowerblue')
ax.set_title(f'Reward cells\n n={len(np.concatenate(coms_mean_rewrel))} cells')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim([vmin,vmax])
ax=axes[2]
ax.hist(np.concatenate(coms_mean_abs),density=True,color='indigo')
ax.set_title(f'Place cells\n n={len(np.concatenate(coms_mean_abs))} cells')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim([vmin,vmax])
ax=axes[3]
alpha=0.4
ax.hist(np.concatenate(com_spatial_tuned_all),density=True,
        color='slategray',alpha=alpha,
        label='Spatially tuned cells')
ax.hist(np.concatenate(coms_mean_rewrel),density=True,color='cornflowerblue',alpha=alpha,
        label='Reward cells')
ax.hist(np.concatenate(coms_mean_abs),density=True,color='indigo',alpha=alpha,
        label='Place cells')
ax.set_xlabel('Center-of-mass (COM)-rew. loc.(cm)')
ax.set_ylim([vmin,vmax])
ax.legend()
ax.spines[['top','right']].set_visible(False)
savedst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures"
plt.savefig(os.path.join(savedst, 'com_hist_rew_place.svg'),bbox_inches='tight')

#%%