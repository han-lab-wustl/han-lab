
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
import pickle, seaborn as sns, random, math, os, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf
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
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)

tracked_rew_cell_inds = {}
tracked_rew_activity = {}
#%%
coms = {}
# redo across days analysis but init array per animal
for animal in animals:
    dys = conddf.loc[conddf.animals==animal, 'days'].values
    # index
    dds = list(conddf[conddf.animals==animal].index)
    for ii, day in enumerate(dys):
        if animal!='e217' and conddf.optoep.values[dds[ii]]==-1:
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
            if ii==0:
                # init with min 4 epochs
                coms_rewrel_tracked = np.ones((4,tracked_lut.shape[0],
                                tracked_lut.shape[1]))*np.nan
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
            eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
            if f'{animal}_{day:03d}_index{dds[ii]:03d}' in radian_alignment_saved.keys():
                tcs_correct, coms_correct, tcs_fail, coms_fail, \
                com_goal, goal_cell_shuf_ps_per_comp_av,\
                goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{dds[ii]:03d}']            

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
                rew_cells_that_are_tracked_iind = np.array([np.where(pyr_tc_s2p_cellind==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id])
                # includes s2p indices of inactive cells so you can find them in the tracked lut
                # build a mat the same size as the tracked cell dataframe
                # nan out other cells
                # ep x cell x bin x days
                # tcs_rew_tracked = np.ones((tcs_correct.shape[0],tracked_lut.shape[0], 
                #                 tcs_correct.shape[2],
                #                 tracked_lut.shape[1]))*np.nan
                # tcs_rew_tracked[:, tracked_rew_cell_ind, :, 
                #         np.where(days==day)[0]]=tcs_correct[:,rew_cells_that_are_tracked_iind,:]                
                if len(tracked_rew_cell_ind)>0:                    
                    coms_rewrel_tracked[:coms_correct.shape[0],tracked_rew_cell_ind,
                        np.where(days==day)[0]] = coms_rewrel[:, rew_cells_that_are_tracked_iind]
            except Exception as e:
                print(e)
    coms[animal] = coms_rewrel_tracked

dct = {}; dct['rew_cells_coms_tracked'] = [coms]
# save pickle of dcts
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_rew_cells.p"
with open(rew_cells_tracked_dct, "wb") as fp:   #Pickling
    pickle.dump(dct, fp) 

#
#%%
# plot
# compile per animal tuning curves
dfs = []; df2s = []; df3s = []
animals = ['e218','e216','e200','e201','e186','e189','e190', 'e145']
for annm in animals:
    # TODO: nan pad so that we can get all epochs!!
    ancom = np.nanmedian(coms[annm],axis=0)    
    if len(ancom.shape)>0:
        tracked_cell_ind = np.arange(ancom.shape[0])
        mask1 = np.nansum(ancom,axis=1)>0
        ancom = ancom[mask1,:]
        if ancom.shape[0]>0:
            tracked_cell_ind = tracked_cell_ind[mask1]
            if annm=='e145': pln=2
            else: pln=0
            # find day match with session        
            txtpth = os.path.join(celltrackpth, rf"{annm}_daily_tracking_plane{pln}\Results")
            txtpth = os.path.join(txtpth, find_log_file(txtpth))
            sessions, days = get_days_from_cellreg_log_file(txtpth)    
            days_per_animal = conddf.loc[((conddf.animals.values==annm) & (conddf.optoep.values==-1)),'days'].values
            sessions_rec = np.hstack(np.array([np.where(np.array(days)==xx)[0] for xx in days_per_animal]))
            # cells x days
            # get only recorded sessions            
            ancom = ancom[:, sessions_rec]
            df = pd.DataFrame()
            df['median_com_across_ep'] = np.ravel(ancom)
            df['day'] = np.concatenate([np.arange(ancom.shape[1])]*ancom.shape[0])+1
            df['animal'] = [annm]*len(df)        
            df['cell'] = np.repeat(tracked_cell_ind, ancom.shape[1])+1
            df['animal_cell'] = [str(xx)+'_'+str(df.cell.values[ii]) for ii,xx in enumerate(df.animal.values)]
            dfs.append(df)
                    
            df2  = pd.DataFrame()
            df2['median_com_across_ep_days'] = np.nanmedian(ancom,axis=1)
            df2['num_days_tracked'] = [np.sum(~np.isnan(xx)) for xx in ancom]
            df2['animal'] = [annm]*len(df2)        
            df2['cell'] =tracked_cell_ind
            df2['animal_cell'] = [str(xx)+'_'+str(df2.cell.values[ii]) for ii,xx in enumerate(df2.animal.values)]
            df2s.append(df2)
            # get epochs separately        
            ancom=coms[annm]
            tracked_cell_ind = np.arange(ancom.shape[1])
            # mask of epoch 1 only
            mask1 = np.nansum(ancom[0,:,:],axis=1)>0
            ancom = ancom[:,mask1,:]
            tracked_cell_ind = tracked_cell_ind[mask1]
            ancom = ancom[:, :, sessions_rec]
            print(f'animal: {annm}, days rec: {ancom.shape[2]}')
            # ravel  = day 1,2,3... x cell 1,2,3 ... x epoch CHECKED THIS
            df3  = pd.DataFrame()
            df3['com'] = np.ravel(ancom)
            df3['day'] =  np.concatenate([np.concatenate([list(range(ancom.shape[2]))]*(ancom.shape[1]))]*ancom.shape[0])
            df3['cell'] = np.repeat(np.repeat(tracked_cell_ind, ancom.shape[2])+1, ancom.shape[0])
            df3['epoch'] = np.repeat(list(range(ancom.shape[0])), ancom.shape[2]*ancom.shape[1])
            df3['epoch_day'] = [str(xx)+'_'+str(df3.day.values[ii]) for ii,xx in enumerate(df3.epoch.values)]
            df3['animal'] = [annm]*len(df3)   
            df3['epoch_animal_day'] = [str(xx)+'_'+str(df3.day.values[ii])+'_'+str(df3.animal.values[ii]) for ii,
                    xx in enumerate(df3.epoch.values)]   
            df3['animal_cell'] = [str(xx)+'_'+str(df3.cell.values[ii]) for ii,xx in enumerate(df3.animal.values)]    
            df3s.append(df3)  
dfs = pd.concat(dfs)
df2s = pd.concat(df2s)
df3s = pd.concat(df3s)

# num tracked days vs. median com
#%%
plt.rc('font', size=22) 
dfs_av = dfs.groupby(['animal_cell', 'day',]).median(numeric_only=True)
# optional = per animal
# annm = 'e216'
# dfsplt = dfs.loc[dfs.animal==annm]

dfsplt = dfs.sort_values(by=['animal_cell'])
dfsplt.index = np.arange(len(dfsplt))
fig,ax=plt.subplots(figsize=(4,7))
ax = sns.stripplot(y='median_com_across_ep',x='day',
            hue='animal_cell',data=dfsplt,s=8,alpha=0.4)

sns.lineplot(y='median_com_across_ep',x=dfsplt.day.values-1,
            hue='animal_cell',data=dfsplt,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.00, 1.00)).set_visible(False)
ax.set_ylabel('Median COM across epochs (rad.)\ncentered at rew. loc.')
ax.set_xlabel('Day')
dfsplt.to_csv(r'C:\Users\Han\Desktop\test.csv')
#%%
# per epoch
plt.rc('font', size=12) 
dfsplt = df3s.sort_values(by=['epoch_day', 'animal_cell'])
dfsplt.index = np.arange(len(dfsplt))
# optional = per animal
annm = 'e216'
dfsplt = dfsplt.loc[dfsplt.animal==annm]

fig,ax=plt.subplots(figsize=(10,7))
ax = sns.stripplot(y='com',x='epoch_day',
            hue='animal_cell',data=dfsplt,s=8,alpha=0.4)
sns.lineplot(y='com',x='epoch_day',
            hue='animal_cell',data=dfsplt,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.00, 1.00)).set_visible(False)
ax.set_ylabel('COM\ncentered at rew. loc.')
ax.tick_params(axis='x', rotation=90)

#%%
sns.histplot(dfsplt.median_com_across_ep,bins=50)
# plt.savefig(os.path.join(savedst, f'rewcom_v_days_tracked.svg'), bbox_inches='tight')
#%%
# plot median across days
plt.rc('font', size=22) 
dfs_av = df2s.groupby(['animal_cell']).median(numeric_only=True)
# optional = per animal
# annm = 'e216'
# dfsplt = dfs.loc[dfs.animal==annm]

dfsplt = df2s.sort_values(by=['animal_cell'])
dfsplt.index = np.arange(len(dfsplt))
fig,ax=plt.subplots()
ax = sns.stripplot(y='median_com_across_ep_days',x='num_days_tracked',
            hue='animal',data=dfsplt,s=8)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.00, 1.00)).set_visible(False)
ax.set_ylabel('Median COM across ep. & days\ncentered at rew. loc.')
ax.set_xlabel('# of days tracked')

#%%

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