
"""
zahra
july 2024
quantify reward-relative cells
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
from placecell import make_tuning_curves_by_trialtype, make_tuning_curves_trial_by_trial, \
    intersect_arrays, make_tuning_curves_probes
from rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_relative_probetr_skewfilt.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto_20240919.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
evolution_com_rewrel = []
radian_alignment = {}
cm_window = 10
# cm_window = [10,20,30,40,50,60,70,80] # cm
#%%
# iterate through all animals
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
            'stat', 'licks'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        lick = fall['licks']
        lick = np.squeeze(lick)
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
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length/bins 
        # import
        tcs_correct, coms_correct, tcs_fail, coms_fail, com_goal, \
                goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        goal_window = 10*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)

        # probes
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # 9/19/24
        # find correct trials within each epoch!!!!
        if len(goal_cells)>0:
            F_rewrel = Fc3[:, ~goal_cells] # NON GOAL CELLS
            # F_rewrel = Fc3[:, goal_cells] # GOAL CELLS
            tcs_probe, coms_probe=make_tuning_curves_probes(eps,rewlocs,ybinned,rad,F_rewrel,trialnum,
                rewards,forwardvel,rewsize,bin_size,probe=[0])
            tcs_probe_other, coms_probe_other=make_tuning_curves_probes(eps,rewlocs,ybinned,rad,F_rewrel,trialnum,
                rewards,forwardvel,rewsize,bin_size,probe=[1,2])  
            trialstates, licks_trial_by_trial, tcs_trial_by_trial,\
            coms_trial_by_trial = make_tuning_curves_trial_by_trial(eps,rewlocs,
                lick,ybinned,rad,F_rewrel,trialnum,
                rewards,forwardvel,rewsize,bin_size)        # remake tcs without circular alignment
        
            coms_probe_rewrel = [com-rewlocs[ep] for ep,com in enumerate(coms_probe)]
            coms_probe_other_rewrel = [com-rewlocs[ep] for ep,com in enumerate(coms_probe_other)]
            coms_per_correcttr = [[com[cl][trialstates[ep]==1]-rewlocs[ep] for cl in range(com.shape[0])] for ep,
                            com in enumerate(coms_trial_by_trial)]
        else:
            coms_probe_rewrel=np.nan;coms_probe_other_rewrel=np.nan;coms_per_correcttr=np.nan
        # learning trials --> end probe 1 --> end probe 2,3
        # animal, trial type, epoch, cell
        evolution_com_rewrel.append([coms_per_correcttr,coms_probe_rewrel,coms_probe_other_rewrel])
        

# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 
#%%
# coms across trials
plt.rc('font', size=22)      
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod']
fig,axes=plt.subplots(ncols=2,sharey=True,figsize=(20,7),width_ratios=[6,1])

for an in range(len(evolution_com_rewrel)):
    try:
        org = [[evolution_com_rewrel[an][0][ep][cll] for cll in range(len(evolution_com_rewrel[an][0][0]))] \
            for ep in range(len(evolution_com_rewrel[an][0]))]
        org_pr = [[np.concatenate([[evolution_com_rewrel[an][1][ep][cll]],
            [evolution_com_rewrel[an][2][ep][cll]]]) for cll in range(len(evolution_com_rewrel[an][0][0]))] \
            for ep in range(len(evolution_com_rewrel[an][0]))]
        for i,arr in enumerate(org):
            for cll in arr:
                # print(len(cll))
                ax=axes[0]
                # add noise
                x=np.arange(len(cll));y=cll
                x_jitter = x + np.random.normal(0, 0.2, size=len(x))
                y_jitter = y + np.random.normal(0, 0.2, size=len(y))
                ax.scatter(x_jitter,y_jitter,color=colors[i],alpha=0.2)
                # ax.plot(cll,color=colors[i])
            ax.set_xlabel('Correct Trial #')   
        for i,arr in enumerate(org_pr):
            for cll in arr:
                # print(len(cll))
                ax=axes[1]
                x=np.arange(len(cll));y=cll
                x_jitter = x + np.random.normal(0, 0.2, size=len(x))
                y_jitter = y + np.random.normal(0, 0.2, size=len(y))
                if i>0: j=i-1 
                else: j=0
                ax.scatter(x_jitter,y_jitter,color=colors[j],alpha=0.2)
                # ax.plot(cll,color=colors[i])
            ax.set_xlabel('Probe Trial #')
    except:
        print('e')
fig.suptitle('All other cells')
fig.tight_layout()