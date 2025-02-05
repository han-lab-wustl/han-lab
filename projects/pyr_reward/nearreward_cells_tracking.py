
"""
zahra
aug 2024
main idea: find near rew cells on the first day and see if they continue to be 
near reward

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
from placecell import intersect_arrays
from rewardcell import get_days_from_cellreg_log_file, find_log_file
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

animals = ['e218','e216','e217','e201','e186','e189','e190', 'e145']

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r"Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto.p"
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
                coms_rewrel_tracked = np.ones((tcs_correct.shape[0],tracked_lut.shape[0],
                                tracked_lut.shape[1]))*np.nan
                if len(tracked_rew_cell_ind)>0:                    
                    coms_rewrel_tracked[:,tracked_rew_cell_ind,np.where(days==day)[0]] = coms_rewrel[:, rew_cells_that_are_tracked_iind]
                tracked_rew_activity[f'{animal}_{day:03d}'] = [rewlocs,coms_rewrel_tracked]
            except Exception as e:
                print(e)

dct = {}; dct['rew_cells_tracking'] = [tracked_rew_cell_inds,tracked_rew_activity]
# save pickle of dcts
rew_cells_tracked_dct = r"Z:\saved_datasets\tracked_nearrew_cells.p"
with open(rew_cells_tracked_dct, "wb") as fp:   #Pickling
    pickle.dump(dct, fp) 

#
#%%
# plot
# compile per animal tuning curves
dfs = []; df2s = []
animals = ['e218','e216','e200','e201','e186','e189','e190', 'e145']
days_per_animal = [k[:4] for k,v in tracked_rew_cell_inds.items()]
days_per_animal = Counter(days_per_animal)
ancoms={}
for annm in animals:
    # TODO: nan pad so that we can get all epochs!!
    ancom = np.nansum(np.array([np.nanmedian(v[1],axis=0) for k,v in tracked_rew_activity.items() if k[:-4]==annm]),axis=0)    
    if len(ancom.shape)>0:
        ancom[ancom==0]=np.nan # replace 0's (due to nan sum) with nans again
        mask1 = np.nansum(ancom,axis=1)>0
        ancom = ancom[mask1,:]
        if annm=='e145': pln=2
        else: pln=0
        # find day match with session        
        txtpth = os.path.join(celltrackpth, rf"{annm}_daily_tracking_plane{pln}\Results")
        txtpth = os.path.join(txtpth, find_log_file(txtpth))
        sessions, days = get_days_from_cellreg_log_file(txtpth)    
        days_per_animal = [int(k[5:]) for k,v in tracked_rew_cell_inds.items() if k[:4]==annm]
        sessions_rec = np.hstack(np.array([np.where(np.array(days)==xx)[0] for xx in days_per_animal]))
        # cells x days
        ancom = ancom[:, sessions_rec]

        df = pd.DataFrame()
        df['median_com_across_ep'] = np.ravel(ancom)
        df['day'] = np.repeat(np.arange(ancom.shape[1]), ancom.shape[0])+1
        df['animal'] = [annm]*len(df)        
        df['cell'] = np.concatenate([np.arange(ancom.shape[0])]*ancom.shape[1])
        df['animal_cell'] = [str(xx)+'_'+str(df.cell.values[ii]) for ii,xx in enumerate(df.animal.values)]
        dfs.append(df)
        df2  = pd.DataFrame()
        df2['median_com_across_ep_days'] = np.nanmedian(ancom,axis=1)
        df2['num_days_tracked'] = [np.sum(np.isnan(xx)) for xx in ancom]
        df2['animal'] = [annm]*len(df2)        
        df2['cell'] = np.arange(ancom.shape[0])
        df2['animal_cell'] = [str(xx)+'_'+str(df2.cell.values[ii]) for ii,xx in enumerate(df2.animal.values)]
        df2s.append(df2)
dfs = pd.concat(dfs)
df2s = pd.concat(df2s)
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
ax.set_xlabel('# of days tracked')

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
            hue='animal_cell',data=dfsplt,s=8)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.00, 1.00)).set_visible(False)
ax.set_ylabel('Median COM across ep. & days\ncentered at rew. loc.')
ax.set_xlabel('# of days tracked')
