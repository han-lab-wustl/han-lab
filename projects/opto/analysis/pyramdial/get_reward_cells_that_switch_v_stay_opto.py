"""
get reward distance cells between opto and non opto conditions
sept 2025
% reward cells
% of cells near rew that don't switch
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
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, \
        intersect_arrays, make_tuning_curves_radians_by_trialtype, make_tuning_curves_all_trialtypes, make_tuning_curves_radians_by_trialtype_early,make_tuning_curves_all_trialtypes_early, make_tuning_curves_by_trialtype_w_darktime_early
from projects.opto.behavior.behavior import smooth_lick_rate

import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
results_all=[]
radian_alignment = {}
cm_window = 20
def normalize_rows(arr):
   arr_new = np.copy(arr)
   rowmins = np.nanmin(arr_new, axis=1, keepdims=True)
   rowmaxs = np.nanmax(arr_new, axis=1, keepdims=True)
   denom = rowmaxs - rowmins
   # Prevent division by zero
   denom[denom == 0] = 1
   arr_new = (arr_new - rowmins) / denom
   # Set entire row to 0 where max == min
   mask = (rowmaxs == rowmins).flatten()
   arr_new[mask, :] = 0
   # return arr_new
   return arr_new # do not norm

#%%
# iterate through all animals 
iis=[60,166,49] # ctrl, vip inhib, vip ex
# iis=[46]
for ii in iis:
    day = int(conddf.days.values[ii])
    animal = conddf.animals.values[ii]
    # skip e217 day
    if ii!=202:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2  
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
        time = fall['timedFF'][0]
        lick = fall['licks'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            time=time[:-1]
            lick=lick[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        # only test opto vs. ctrl
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: 
            eptest = random.randint(2,3)   
            if len(eps)<4: eptest = 2 # if no 3 epochs 
        eptest=int(eptest)-1   
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins

        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3_org = fall_fc3['Fc3']
        dFF_org = fall_fc3['dFF']
        Fc3_org = Fc3_org[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF_org = dFF_org[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF_org, nan_policy='omit', axis=0)
        dFF=dFF_org[:, skew>1.2]
        Fc3=Fc3_org[:, skew>1.2]
        # tc w/ dark time
        print('making tuning curves...\n')
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,dFF,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt,lasttr=8) 
        # early tc
        tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,dFF,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=3)  
        goal_window = 20*(2*np.pi/track_length)
        
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2)) 
        print(perm)
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
        # remove empty epochs
        com_goal = [xx for xx in com_goal if len(xx)>0]
        # get cells cells near rew in ep1
        if len(com_goal)>0:
            goal_cells = intersect_arrays(*com_goal)
        else:
            goal_cells=[]
        # get tuning curves based on all trials
        bin_size=2
        lasttr=8
        firsttr=3
        
        tcs_correct_abs, coms_correct_abs = make_tuning_curves_all_trialtypes(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size)) # last 5 trials
        tcs_correct_abs_early, coms_correct_abs_early = make_tuning_curves_all_trialtypes_early(eps,rewlocs,ybinned, Fc3,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size),lasttr=firsttr) # last 5 trials
        # lick and vel
        dt=np.nanmedian(np.diff(time))
        lick_rate=smooth_lick_rate(lick,dt)
        lick_tcs_correct_abs, lick_coms_correct_abs = make_tuning_curves_all_trialtypes(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size)) # last 5 trials
        lick_tcs_correct_abs_early, lick_coms_correct_abs_early = make_tuning_curves_all_trialtypes_early(eps,rewlocs,ybinned,np.array([lick_rate]).T,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size),lasttr=firsttr) # last 5 trials
        
        vel_tcs_correct_abs, vel_coms_correct_abs = make_tuning_curves_all_trialtypes(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size)) # last 5 trial
        vel_tcs_correct_abs_early, vel_coms_correct_abs_early = make_tuning_curves_all_trialtypes_early(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_size,bins=int(track_length/bin_size),lasttr=firsttr) # last 5 trials
        # last 5 trials
        fig, axes=plt.subplots(ncols=3,nrows=3,height_ratios=[3,1,1],sharey='row',sharex=True)
        ax=axes[0,0]
        mat = normalize_rows(tcs_correct_abs[eptest-1][np.argsort(coms_correct_abs[eptest-1])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'Last {lasttr} trials, epoch 1')
        ax=axes[1,0]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax.set_ylabel('Mean $\Delta F/F$')
        ax=axes[2,0]
        mat = lick_tcs_correct_abs[eptest-1][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.set_ylabel('Lick rate (Hz)')
        
        ax=axes[0,1]
        mat = normalize_rows(tcs_correct_abs_early[eptest][np.argsort(coms_correct_abs[eptest-1])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'First {firsttr} trials, epoch 2')
        ax=axes[1,1]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[2,1]
        mat = lick_tcs_correct_abs_early[eptest][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[0,2]
        mat = normalize_rows(tcs_correct_abs[eptest][np.argsort(coms_correct_abs[eptest-1])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'Last {lasttr} trials, epoch 2')
        ax=axes[1,2]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[2,2]
        mat = lick_tcs_correct_abs[1][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        fig.suptitle(f'{animal},{day},All pyramidal cells')

        # reward cells
        # show new vs. old rew zone
        fig, axes=plt.subplots(ncols=3,nrows=3,height_ratios=[3,1,1],sharey='row',sharex=True)
        ax=axes[0,0]
        mat = normalize_rows(tcs_correct_abs[eptest-1][goal_cells][np.argsort(coms_correct_abs[eptest-1][goal_cells])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'Last {lasttr} trials, epoch 1')
        ax=axes[1,0]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax.set_ylabel('Mean $\Delta F/F$')
        ax=axes[2,0]
        mat = lick_tcs_correct_abs[eptest-1][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.set_ylabel('Lick rate (Hz)')
        
        ax=axes[0,1]
        mat = normalize_rows(tcs_correct_abs_early[eptest][goal_cells][np.argsort(coms_correct_abs[eptest-1][goal_cells])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'First {firsttr} trials, epoch 2')
        ax=axes[1,1]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[2,1]
        mat = lick_tcs_correct_abs_early[eptest][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[0,2]
        mat = normalize_rows(tcs_correct_abs[eptest][goal_cells][np.argsort(coms_correct_abs[eptest-1][goal_cells])])
        ax.imshow(mat,aspect='auto')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)+(rewsize/bin_size/2),color='y',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='w',linestyle='--')
        ax.axvline((rewlocs[1]/bin_size)+(rewsize/bin_size/2),color='w',linestyle='--')
        ax.set_title(f'Last {lasttr} trials, epoch 2')
        ax=axes[1,2]
        ax.plot(np.nanmean(mat,axis=0))
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')

        ax=axes[2,2]
        mat = lick_tcs_correct_abs[1][0]
        ax.plot(mat,color='k')
        ax.axvline((rewlocs[1]/bin_size)-(rewsize/bin_size/2),color='k',linestyle='--')
        ax.axvline((rewlocs[0]/bin_size)-(rewsize/bin_size/2),color='y',linestyle='--')
        fig.suptitle(f'{animal},{day},All reward cells')


