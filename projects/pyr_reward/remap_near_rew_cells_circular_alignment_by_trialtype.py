
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
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, make_tuning_curves_radians_trial_by_trial
from rewardcell import get_radian_position, goal_cell_shuffle, get_goal_cells, get_trialtypes
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'rew_cells_quant_by_trialtypes.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
to_save = r"Z:\saved_datasets\rew_cells_split_by_trialtype.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
lasttr=8 # last trials
bins=90

dct = {}
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
total_cells = []
epoch_perm = []
per_day_trialtypes = []
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal!='e217' and conddf.optoep.values[ii]<2:
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'licks',
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        lick = np.squeeze(fall['licks'])
        # set vars
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))        
        # save some vars
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        # get tuning curves relative to reward per trial
        # import fluor
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # get correct tuning curves 
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,
                rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size) 
            
        # get goal cells
        goal_window, goal_cells, perm, goal_cells_p_per_comparison,\
            goal_cell_p,coms_rewrel = get_goal_cells(track_length,coms_correct,window=30)
        goal_cell_iind.append(goal_cells); epoch_perm.append(perm)
        goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p]);
        num_epochs.append(len(coms_correct))
        dist_to_rew.append(coms_rewrel[:, goal_cells])
    
        # split by trial type
        # init_fails, first_correct, correct_trials_besides_first, 
        #         inbtw_fails, total_trials
        per_ep_trialtypes = get_trialtypes(trialnum, rewards, ybinned,
                coms_correct, eps)
        per_day_trialtypes.append(per_ep_trialtypes)
        F_remap = Fc3[:, goal_cells]
        # cells x trial x bin        
        trialstates, licks_trial_by_trial, tcs_trial_by_trial, coms_trial_by_trial = make_tuning_curves_radians_trial_by_trial(eps,rewlocs,
        lick,ybinned,rad,F_remap,trialnum,
        rewards,forwardvel,rewsize,bin_size)
        # goal cells by trial type
        gc_init_fails = [tcs_trial_by_trial[ep][:,per_ep_trialtypes[ep][0],:] for ep in range(len(coms_correct))]
        gc_init_corr = [tcs_trial_by_trial[ep][:,per_ep_trialtypes[ep][1],:] for ep in range(len(coms_correct))]
        gc_init_probe = [[tcs_trial_by_trial[ep][:,0,:] ]for ep in range(len(coms_correct))]
        gc_other_probe = [tcs_trial_by_trial[ep][:,1:2,:] for ep in range(len(coms_correct))]
        gc_corr = [tcs_trial_by_trial[ep][:,per_ep_trialtypes[ep][2],:] for ep in range(len(coms_correct))]
        gc_incorr = [tcs_trial_by_trial[ep][:,per_ep_trialtypes[ep][3],:] for ep in range(len(coms_correct))]
        
        dct[f'{animal}_{day:03d}_index{ii:03d}'] = [gc_init_fails,gc_init_corr,
            gc_corr,gc_incorr]
        plt.rc('font', size=12)
        for celln in range(len(goal_cells)):            
            for ep in range(len(coms_correct)):
                fig, ax = plt.subplots()
                gc_arr = np.array(gc_init_probe[ep])[:,celln,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['init_probe'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='init_probe',color='darkorange',
                    ax=ax, label='init_probe')
                
                gc_arr = gc_other_probe[ep][celln,:,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['other_probes'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='other_probes',color='olive',
                    ax=ax, label='other_probes')
                
                gc_arr = gc_corr[ep][celln,:,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['correct_trials'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='correct_trials',color='k',
                    ax=ax, label='correct_trials')
                
                gc_arr = gc_init_corr[ep][celln,:,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['first_correct'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='first_correct',color='teal',
                            ax=ax,label='first_correct')
            
                gc_arr = gc_init_fails[ep][celln,:,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['first_incorrect'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='first_incorrect',color='blue',
                            ax=ax,label='first_incorrect')

                gc_arr = gc_incorr[ep][celln,:,:]
                arr = np.concatenate(gc_arr.T)
                lbl = np.concatenate([[xx]*gc_arr.shape[0] for xx in np.arange(gc_arr.shape[1])])
                df = pd.DataFrame()
                df['incorrect_trials'] = arr
                df['position'] = lbl
                sns.lineplot(data=df, x='position',y='incorrect_trials',color='magenta',
                        ax=ax, label='incorrect_trials')
                ax.set_xticks(np.arange(0,bins+1,10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),2))
                ax.set_xlabel('Position (rad. rel. rew.)')
                ax.axvline(bins/2, color='slategray',linestyle='--')
                ax.spines[['top','right']].set_visible(False)
                ax.legend()
                ax.set_ylabel('Fc3')
                ax.set_title(f'animal: {animal}, day: {day}\nepoch {ep+1}, cell # {goal_cells[celln]}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


# save pickle of dcts
with open(to_save, "wb") as fp:   #Pickling
    pickle.dump(dct, fp)
    
pdf.close()