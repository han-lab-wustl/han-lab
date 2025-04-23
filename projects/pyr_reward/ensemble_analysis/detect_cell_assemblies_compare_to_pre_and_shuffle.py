
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
savepth = os.path.join(savedst, 'post_rew_assemblies.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
# initialize var
# radian_alignment_saved = {} # overwri       te
tcs_rew = []
goal_cells_all = []
bins = 90
goal_window_cm=20
epoch_perm=[]
assembly_cells_all_an=[]
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
        # if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        #     tcs_correct, coms_correct, tcs_fail, coms_fail, \
        #     com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        # else:# remake tuning curves relative to reward        
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
        # all cells after 0
        com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0)] if len(com)>0 else [] for com in com_goal]
        #only get perms with non zero cells
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
        rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
        assembly_cells_all = {}
        try: # if enough neurons
            goal_all = np.unique(np.concatenate(com_goal_postrew))
        # significant_components, assembly_patterns, assembly_activities = detect_cell_assemblies(Fc3[:,goal_all].T, 
        #     significance_threshold=None, plot=False)
            from ensemble import detect_assemblies_with_ica,cluster_neurons_from_ica,\
            get_cells_by_assembly
            patterns, activities, labels, n = detect_assemblies_with_ica(Fc3[:,goal_all].T)
            print(f"{n} assemblies detected")

            labels = cluster_neurons_from_ica(patterns)
            assembly_cells = get_cells_by_assembly(labels)

            # Print cell indices for each assembly
            for assembly_id, cells in assembly_cells.items():
                time_bins = np.arange(90)
                # Calculate weighted average time bin (center of mass)
                activity = tcs_correct[0,goal_all[cells],:]
                center_of_mass = np.sum(activity * time_bins) / np.sum(activity) if np.sum(activity) > 0 else np.nan
                com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in activity]
                com_com_asm = com_per_cell-center_of_mass
                if np.nanmean(com_com_asm)<np.pi/6:
                    fig, ax = plt.subplots()
                    ax.plot(np.nanmean(tcs_correct[:,goal_all[cells],:],axis=0).T)
                    ax.plot(tcs_correct[0,goal_all[cells],:].T)
                    ax.set_title(f'{animal}, {day}, Assembly ID: {assembly_id}')
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    # only get cells with close assembly seq
                    # save tc
                    assembly_cells_all[f'assembly {assembly_id}']=tcs_correct[:,goal_all[cells],:]
        except Exception as e:
            print(e)
        assembly_cells_all_an.append(assembly_cells_all)

# Assembly activity is a time series that measures how active a particular 
# neuronal ensemble (identified via PCA) is at each time point. It reflects 
# coordinated activity, not just individual spikes.
pdf.close()

#%%
from projects.pyr_reward.rewardcell import cosine_sim_ignore_nan
# look through all the assemblies
df = conddf.copy()
df = df[(df.animals!='e217') & (df.optoep.values<2)]
an_plt = 'e190' # 1 eg animal
plt.close('all')
for ii,ass in enumerate(assembly_cells_all_an):
    if df.iloc[ii].animals==an_plt:
        print(f'{df.iloc[ii].animals}, {df.iloc[ii].days}')
        ass_all = list(ass.values()) # all assemblies
        cs_per_ep = []
        for jj,asm in enumerate(ass_all):
            perm = list(combinations(range(len(asm)), 2)) 
            # consecutive ep only
            perm = [p for p in perm if p[0]-p[1]==-1]
            cs = [cosine_sim_ignore_nan(asm[p[0]], asm[p[1]]) for p in perm]
            cs_per_ep.append(cs)
            fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5))
            for kk,tcs in enumerate(asm):
                ax = axes[kk]
                if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                ax.imshow(tcs[np.argsort(com_per_cell)]**.4,aspect='auto')
                ax.set_title(f'epoch {kk+1}')
                ax.axvline(bins/2, color='w', linestyle='--')
            # fig.suptitle(f'Cosine similar b/wn epochs: \n\
            #     Epoch combinations: {perm}\n\
            #         CS: {np.round(cs,2)}, average: {np.nanmean(cs)}')
            fig.suptitle(f'{df.iloc[ii].animals}, {df.iloc[ii].days} \n\
                Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}')

            # plt.figure()
            # plt.plot(tcs[np.argsort(com_per_cell)].T)
# %%
