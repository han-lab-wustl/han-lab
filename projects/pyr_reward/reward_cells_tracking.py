
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
import pickle, seaborn as sns, random, math, os
from collections import Counter
from itertools import combinations, chain
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
from rewardcell import get_days_from_cellreg_log_file, find_log_file
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

animals = ['e218','e216','e217','e201','e186','e189','e190', 'e145']

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
radian_alignment = {}
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)

tracked_rew_cell_inds = {}
tracked_rew_activity = {}
#%%
for dd,day in enumerate(conddf.days.values):
    animal = conddf.animals.values[dd]
    if animal in animals and animal!='e217' and conddf.optoep.values[dd]==-1:
        if animal=='e145': pln=2
        else: pln=0
        # get lut
        tracked_lut = scipy.io.loadmat(os.path.join(celltrackpth, 
        rf"{animal}_daily_tracking_plane{pln}\Results\commoncells_once_per_week.mat"))
        tracked_lut = tracked_lut['commoncells_once_per_week']
        # find day match with session        
        txtpth = os.path.join(celltrackpth, rf"{animal}_daily_tracking_plane{pln}\Results")
        txtpth = os.path.join(txtpth, find_log_file(txtpth))
        sessions, days = get_days_from_cellreg_log_file(txtpth)    
        tracked_lut = pd.DataFrame(tracked_lut, columns = days)
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 
            'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF'])
        # to remove skew cells
        dFF = fall['dFF']
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        if f'{animal}_{day:03d}_index{dd:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,\
            goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{dd:03d}']            

        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        pyr_tc_s2p_cellind = fall['pyr_tc_s2p_cellind'][0][skew>2]
        # make sure dims are correct for transform
        assert pyr_tc_s2p_cellind.shape[0]==tcs_correct.shape[1]
        # suite2p indices of rew cells
        if len(goal_cells)>0:
            goal_cells_s2p_ind = pyr_tc_s2p_cellind[goal_cells]
            # change to relative value 
            coms_rewrel = np.array([com-np.pi for com in coms_correct])
            try:
                tracked_rew_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in goal_cells_s2p_ind]
                tracked_rew_cell_inds[f'{animal}_{day:03d}'] = tracked_rew_cell_ind   
                tracked_cells_that_are_rew_pyr_id = tracked_lut[day].values[tracked_rew_cell_ind]
                rew_cells_that_are_tracked_iind = [np.where(pyr_tc_s2p_cellind==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id]
                # includes s2p indices of inactive cells so you can find them in the tracked lut
                tracked_rew_activity[f'{animal}_{day:03d}'] = [[tcs_correct[c][rew_cells_that_are_tracked_iind] for c in range(len(coms_correct))],\
                        rewlocs,coms_rewrel[:,[rew_cells_that_are_tracked_iind]]]
            except Exception as e:
                print(e)

dct = {}; dct['rew_cells_tracking'] = [tracked_rew_cell_inds,tracked_rew_activity]
# save pickle of dcts
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_rew_cells.p"
with open(rew_cells_tracked_dct, "wb") as fp:   #Pickling
    pickle.dump(dct, fp) 

#%%
bins = 90
# tracked cell activity
tc_tracked_per_cond = {}
com_tracked_per_cond = {}
for k,v in tracked_rew_cell_inds.items():
    if k in tracked_rew_activity.keys():
        tcs = tracked_rew_activity[k][0]
        rewlocs = tracked_rew_activity[k][1]
        coms = tracked_rew_activity[k][2]        
        animal = k[:-4]
        if animal=='e145': pln=2
        else: pln=0
        tracked_lut = scipy.io.loadmat(rf"Y:\analysis\celltrack\{animal}_daily_tracking_plane{pln}\Results\commoncells_once_per_week.mat")
        tracked_lut = tracked_lut['commoncells_once_per_week']
                # find day match with session        
        txtpth = os.path.join(celltrackpth, rf"{animal}_daily_tracking_plane{pln}\Results")
        txtpth = os.path.join(txtpth, find_log_file(txtpth))
        sessions, days = get_days_from_cellreg_log_file(txtpth)    
        tracked_lut = pd.DataFrame(tracked_lut, columns = days)
        tc_tracked = np.ones((len(coms), len(tracked_lut), bins))*np.nan        
        tracked_cell_id = v    
        tc_tracked[:,tracked_cell_id,:] = tcs
        tc_tracked_per_cond[k] = tc_tracked
        coms_tracked = np.ones((len(coms), len(tracked_lut)))*np.nan
        if len(coms.shape)>2 and coms.shape[2]>1: coms = np.squeeze(coms)
        elif coms.shape[2]==1: coms = np.reshape(coms, (coms.shape[0],coms.shape[2]))
        coms_tracked[:,tracked_cell_id] = coms
        com_tracked_per_cond[k] = coms_tracked
#%%
# plot
# compile per animal tuning curves
dfs = []
animals = ['e218','e216','e200','e201','e186','e189','e190', 'e145']
days_per_animal = [k[:4] for k,v in tracked_rew_cell_inds.items()]
days_per_animal = Counter(days_per_animal)
ancoms={}
for annm in animals:
    # TODO: nan pad so that we can get all epochs!!
    an = np.array([v[:3,:,:] for k,v in tc_tracked_per_cond.items() if k[:-4]==annm and v.shape[0]>2])
    ancom = np.array([v[:3] for k,v in com_tracked_per_cond.items() if k[:-4]==annm and v.shape[0]>2])

    # remove cells that are nan every tracked day
    least_tracked_days = 1
    if len(an)>0:
        mask = (np.sum(np.sum(np.isnan(an[:,0,:,:]),axis=2),
                axis=0)<((an.shape[3]*an.shape[0])-((least_tracked_days-1)*an.shape[3]))) # not nan in all positions across all days
        an = an[:,:,mask,:]    

        ancom = ancom[:,:,mask]
        median_com_across_ep = np.nanmedian(ancom,axis=1)
        # remember that this is in radian 
        median_com_across_ep_and_days = np.nanmedian(median_com_across_ep,axis=0)
        num_days_tracked = np.sum((np.sum(np.isnan(an[:,0,:,:]),axis=2)<90), axis=0)
        df = pd.DataFrame()
        df['median_com_across_ep_and_days'] = median_com_across_ep_and_days
        df['num_days_tracked'] = num_days_tracked
        df['animal'] = [annm]*len(num_days_tracked)
        dfs.append(df)
        # alternatively, keep epochs intact
        ancoms[annm]=ancom
dfs = pd.concat(dfs)
# num tracked days vs. median com
#%%
plt.rc('font', size=22) 
# optional = per animal
# annm = 'e186'
# dfs = dfs.loc[dfs.animal==annm]
dfs_av = dfs.groupby(['animal', 'num_days_tracked']).median(numeric_only=True)

dfs = dfs.sort_values(by=['animal'])
dfs_av = dfs_av.sort_values(by=['animal'])
fig,ax=plt.subplots(figsize=(4,7))
ax = sns.stripplot(y='median_com_across_ep_and_days',x='num_days_tracked',
            hue='animal',data=dfs,s=8,alpha=0.4)
ax = sns.stripplot(y='median_com_across_ep_and_days',x='num_days_tracked',
            hue='animal',data=dfs_av,s=10)
ax = sns.boxplot(y='median_com_across_ep_and_days',x='num_days_tracked',
            color='k',data=dfs, fill=False, width=.5) 
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.00, 1.00))
ax.set_ylabel('Median COM (rad.)\ncentered at rew. loc.')
ax.set_xlabel('# of days tracked')

plt.savefig(os.path.join(savedst, f'rewcom_v_days_tracked.svg'), bbox_inches='tight')
#%%
# plot coms across epochs per day
an='e218'
ancom = ancoms[an]

dfs=[]
for dy in range(ancom.shape[0]):   
    df = pd.DataFrame()     
    df['com'] = np.ravel(ancom[dy])
    df['day'] = [dy]*len(df)
    df['cell'] = np.concatenate([range(ancom[dy].shape[1]),
                        range(ancom[dy].shape[1]),range(ancom[dy].shape[1])])
    df['epoch'] = np.concatenate([[1]*ancom[dy].shape[1], [2]*ancom[dy].shape[1],
                        [3]*ancom[dy].shape[1]])
    dfs.append(df)
ancomdf = pd.concat(dfs)
ancomdf = ancomdf.reset_index()
#%%
for c in ancomdf.cell.unique():
    fig, ax = plt.subplots()
    ancomdfc = ancomdf[ancomdf.cell==c]
    sns.stripplot(y='com',x='epoch',hue='day',
                data=ancomdfc, palette='colorblind', s=10, ax=ax)
    sns.lineplot(x=ancomdfc.epoch.values-1, y='com',hue='day',
                data=ancomdfc, palette='colorblind',ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.legend().set_visible(False)
    ax.set_title(f'Cell: {c}')
#%%
ancomdf = ancomdf[ancomdf.day==1]
ax = sns.stripplot(y='com',x='epoch',hue='cell',
            data=ancomdf, palette='colorblind', s=10)
ax = sns.lineplot(x=ancomdf.epoch.values-1, y='com',hue='cell',
            data=ancomdf, palette='colorblind')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)

# plot individual cell traces
# shp = int(np.ceil(np.sqrt(an.shape[2])))
# fig, axes = plt.subplots(ncols=shp,
#                     nrows=shp,sharex=True,
#                     figsize=(30,20))
# import matplotlib as mpl
# name = "tab20"
# cmap = mpl.colormaps[name]  # type: matplotlib.colors.ListedColormap
# colors = cmap.colors  # type: list

# for dy in range(an.shape[0]):  
#     plt.rc('font', size=10)
#     rr=0;cc=0
#     for ii in range(an.shape[2]):
#         ax=axes[rr,cc]
#         # for jj in range(an.shape[1]): # epochs
#         jj=0;ax.plot(an[dy,jj,ii,:], color=colors[dy]) # just plot epoch 1
#         cell_days_tracked = np.sum(np.sum(np.isnan(an[:,jj,ii,:]),axis=1)<90)
#         ax.set_title(f'cell {ii}, days tracked {cell_days_tracked}')
#         ax.spines[['top','right']].set_visible(False)
#         ax.axvline(x = int(bins/2), color='k', linestyle='--')
#         ax.set_xticks(np.arange(0,bins+1,10))
#         ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
#         if ii==an.shape[0]-1:
#             ax.set_xlabel('Radian position \n(centered at start of rew loc)')
#         rr+=1
#         if rr>np.ceil(np.sqrt(an.shape[2]))-1:cc+=1;rr=0        
#     # fig.suptitle(f'Day {dy}')
# fig.suptitle(f'{annm}, all days')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, f'{annm}.svg'))
# %%
# plot com of a cell vs. number of tracked days