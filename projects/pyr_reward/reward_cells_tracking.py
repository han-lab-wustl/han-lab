
"""
zahra
june 2024
visualize reward-relative cells across days
idea 1: find all the reward relative cells per day and see if they map onto the same 
subset of cells
idea 2: find reward relative cells on the last day (or per week, or per 5 days)
and see what their activity was like on previous days

"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations
from itertools import chain
import matplotlib.backends.backend_pdf
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

animals = ['e216','e218','e200','e201','e186','e189','e190']
days_tracked_per_an = {'e216':np.concatenate([[32,33],range(35,64),[65]]),
                    # 'e217': np.concatenate([range(2,21),[26,27,29,31,32,34,39,43,47]]),
                    'e218':np.arange(20,51),
                    'e200':np.concatenate([range(62,71),[72,73,74,76],range(80,91)]),
                    'e201':np.arange(55,76),
                    'e186':np.arange(1,52),
                    'e189':np.concatenate([[7,8],range(10,16),range(17,22),range(24,43),[44,45,46]]),
                    'e190':np.concatenate([range(6,10),[11,13],range(15,20),[21,22,24,27,28,29,33,34,35],
                                range(40,44),[45]])
                    }
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_relative_across_days.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
radian_tuning_dct = r"Z:\saved_datasets\radian_tuning_curves_reward_cell.p"
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
radian_alignment = {}
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_cell_tracking.csv", index_col=None)

tracked_rew_cell_inds = {}
tracked_rew_activity = {}
for dd,day in enumerate(conddf.days.values):
    animal = conddf.animals.values[dd]
    if animal in animals:
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = days_tracked_per_an[animal]
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 
            'trialnum', 'rewards', 'iscell', 'bordercells'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = [] # get radian coordinates
        # same as giocomo preprint - worked with gerardo
        for i in range(len(eps)-1):
            y = ybinned[eps[i]:eps[i+1]]
            rew = rewlocs[i]-rewsize/2
            # convert to radians and align to rew
            rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
        rad = np.concatenate(rad)
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
        rates_all.append(success/total_trials)
        key = [k for k,v in radian_alignment_saved.items() if f'{animal}_{day:03d}' in k ]
        if len(key)>0:
            tcs_late, coms = radian_alignment_saved[key[0]]            
        else:# remake tuning curves relative to reward        
            # takes time
            fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3'])
            Fc3 = fall_fc3['Fc3']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            rates, tcs_late, coms = make_tuning_curves_radians(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)
            tcs_late = np.array(tcs_late); coms = np.array(coms)            
        goal_window = 30*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms])
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        pyr_tc_s2p_cellind = fall['pyr_tc_s2p_cellind'][0]
        # suite2p indices of rew cells
        goal_cells_s2p_ind = pyr_tc_s2p_cellind[goal_cells]
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms[0])
        goal_cell_prop.append(goal_cell_p)
        num_epochs.append(len(coms))
        # get shuffled iterations
        num_iterations = 1000
        shuffled_dist = np.zeros((num_iterations))
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms[ii].shape[0])) for ii in range(1, len(coms))]
            [random.shuffle(shuf) for shuf in shufs]
            com_shufs = np.zeros_like(coms)
            com_shufs[0,:] = coms[0]
            com_shufs[1:1+len(shufs),:] = [coms[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf = intersect_arrays(*com_goal)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        print(p_value)
        total_cells.append(len(coms[0]))
        radian_alignment[f'{animal}_{day:03d}'] = [tcs_late, coms]
        
        try:
            tracked_rew_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in goal_cells_s2p_ind]
            tracked_rew_cell_inds[f'{animal}_{day:03d}'] = tracked_rew_cell_ind   
            tracked_cells_that_are_rew_pyr_id = tracked_lut[day].values[tracked_rew_cell_ind]
            rew_cells_that_are_tracked_iind = [np.where(pyr_tc_s2p_cellind==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id]
            # includes s2p indices of inactive cells so you can find them in the tracked lut
            tracked_rew_activity[f'{animal}_{day:03d}'] = [[tcs_late[c][rew_cells_that_are_tracked_iind] for c in range(len(coms))],\
                    rewlocs,coms_rewrel[:,[rew_cells_that_are_tracked_iind]]]
        except Exception as e:
            print(e)

# pdf.close()
dct = {}; dct['rew_cells_tracking'] = [tracked_rew_cell_inds,tracked_rew_activity]
# save pickle of dcts
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_rew_cells.p"
with open(rew_cells_tracked_dct, "wb") as fp:   #Pickling
    pickle.dump(dct, fp) 

#%%
# p-values for sanity check
plt.rc('font', size=16)          # controls default text sizes
# goal cells across epochs
df = conddf.copy()
df = df[df.animals!='e217']
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals

fig,ax = plt.subplots(figsize=(5,5))
ax.hist(df.loc[df.opto==False,'p_value'].values)
ax.spines[['top','right']].set_visible(False)

#%%

# tracked cell activity
tc_tracked_per_cond = {}

for k,v in tracked_rew_cell_inds.items():
    if k in tracked_rew_activity.keys():
        tcs = tracked_rew_activity[k][0]
        rewlocs = tracked_rew_activity[k][1]
        coms = tracked_rew_activity[k][2]
        animal = k[:-4]
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane0\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
        days_tracked = days_tracked_per_an[animal]
        tracked_lut = pd.DataFrame(tracked_lut, columns = days_tracked)
        tc_tracked = np.ones((len(coms), len(tracked_lut), bins))*np.nan
        tracked_cell_id = v    
        tc_tracked[:,tracked_cell_id,:] = tcs
        tc_tracked_per_cond[k] = tc_tracked

#%%
# plot
# compile per animal tuning curves
annm = 'e189'
an = np.array([v[:2,:,:] for k,v in tc_tracked_per_cond.items() if k[:-4]==annm and v.shape[0]>1])
# remove cells that are nan every tracked day
mask = (np.sum(np.sum(np.isnan(an[:,0,:,:]),axis=2),axis=0)<((an.shape[3]*an.shape[0])-(1*an.shape[3]))) # not nan in all positions across all days
an = an[:,:,mask,:]
shp = int(np.ceil(np.sqrt(an.shape[2])))
fig, axes = plt.subplots(ncols=shp,
                    nrows=shp,sharex=True,
                    figsize=(30,20))
import matplotlib as mpl
name = "tab20"
cmap = mpl.colormaps[name]  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

for dy in range(an.shape[0]):  
    plt.rc('font', size=10)
    rr=0;cc=0
    for ii in range(an.shape[2]):
        ax=axes[rr,cc]
        # for jj in range(an.shape[1]): # epochs
        jj=0;ax.plot(an[dy,jj,ii,:], color=colors[dy]) # just plot epoch 1
        cell_days_tracked = np.sum(np.sum(np.isnan(an[:,jj,ii,:]),axis=1)<90)
        ax.set_title(f'cell {ii}, days tracked {cell_days_tracked}')
        ax.spines[['top','right']].set_visible(False)
        ax.axvline(x = int(bins/2), color='k', linestyle='--')
        ax.set_xticks(np.arange(0,bins+1,10))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
        if ii==an.shape[0]-1:
            ax.set_xlabel('Radian position \n(centered at start of rew loc)')
        rr+=1
        if rr>np.ceil(np.sqrt(an.shape[2]))-1:cc+=1;rr=0        
    # fig.suptitle(f'Day {dy}')
fig.suptitle(f'{annm}, all days')
fig.tight_layout()
plt.savefig(os.path.join(savedst, f'{annm}.svg'))
# %%
