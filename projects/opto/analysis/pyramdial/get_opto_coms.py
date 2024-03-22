"""plot regular coms - rewloc
for opto vs. ledoff epochs
"""


import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto, normalize_2d_array
import matplotlib.backends.backend_pdf

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\conddf_neural_dark_time_only.csv", index_col=None)
#%%
figcom, axcom = plt.subplots()
figcom2, axcom2 = plt.subplots()
figcom3, axcom3 = plt.subplots()
figcom4, axcom4 = plt.subplots()

inactive = []
active = []
pre_post_tc = []
rewzones_comps = []
for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    if conddf.in_type.values[ii]=='vip':# and conddf.animals.values[ii]!='e217':
        
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
        #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_early_trials',
            'tuning_curves_late_trials', 'coms', 'coms_early_trials'])        
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: eptest = random.randint(2,3)    
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)        
        eps = np.append(eps, len(changeRewLoc))    
        if len(eps)<4: eptest = 2 # if no 3 epochs
        comp = [eptest-2,eptest-1] # eps to compare    
        rewzones_comps.append(rewzones[comp])
        bin_size = 3    
        tcs_early = fall['tuning_curves_early_trials'][0]
        tcs_late = fall['tuning_curves_late_trials'][0]
        tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0]]]))
        tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1]]]))
        tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0]]]))
        tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1]]]))        
        # Find differentially inactivated cells
        threshold=10
        differentially_inactivated_cells = find_differentially_inactivated_cells((tc1_late[:, :int(rewlocs[comp[1]]/bin_size)]), (tc2_late[:, :int(rewlocs[comp[1]]/bin_size)]), threshold, bin_size)
        differentially_activated_cells = find_differentially_activated_cells((tc1_late[:, :int(rewlocs[comp[1]]/bin_size)]),(tc2_late[:, :int(rewlocs[comp[1]]/bin_size)]), threshold, bin_size)
        # coms = fall['coms_pc_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]        
        coms1 = np.hstack(coms[comp[0]])
        coms2 = np.hstack(coms[comp[1]])
        coms1_early = np.hstack(coms_early[comp[0]])
        coms2_early = np.hstack(coms_early[comp[1]])
        # # replace nan coms
        # for jj,tc in enumerate(fall['tuning_curves_circular_late_trials'][0]):
        #     peak = np.nanmax(tc,axis=1)
        #     coms_max = np.array([np.where(tc[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
        #     coms[jj][np.isnan(coms[jj])]=coms_max[np.isnan(coms[jj])]
        if len(differentially_inactivated_cells)>0 and len(differentially_activated_cells)>0:
            pre = sum((((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])<=0)))/len(coms2[differentially_inactivated_cells])
            pre_post = sum((((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])>0)))/len(coms2[differentially_inactivated_cells])
            pre_post_tc1 = tc1_late[differentially_inactivated_cells][(((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])>0)),:]
            pre_post_tc2 = tc2_late[differentially_inactivated_cells][(((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])>0)),:]
            pre_post_tc.append([pre_post_tc1,pre_post_tc2])
            post = sum((((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])>0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])>0)))/len(coms2[differentially_inactivated_cells])
            post_pre = sum((((coms1[differentially_inactivated_cells]-rewlocs[comp[0]])>0) & ((coms2[differentially_inactivated_cells]-rewlocs[comp[1]])<=0)))/len(coms2[differentially_inactivated_cells])
            inactive.append([pre, pre_post, post, post_pre])
            pre = sum((((coms1[differentially_activated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_activated_cells]-rewlocs[comp[1]])<=0)))/len(coms2[differentially_activated_cells])
            pre_post = sum((((coms1[differentially_activated_cells]-rewlocs[comp[0]])<=0) & ((coms2[differentially_activated_cells]-rewlocs[comp[1]])>0)))/len(coms2[differentially_activated_cells])
            post = sum((((coms1[differentially_activated_cells]-rewlocs[comp[0]])>0) & ((coms2[differentially_activated_cells]-rewlocs[comp[1]])>0)))/len(coms2[differentially_activated_cells])
            post_pre = sum((((coms1[differentially_activated_cells]-rewlocs[comp[0]])>0) & ((coms2[differentially_activated_cells]-rewlocs[comp[1]])<=0)))/len(coms2[differentially_activated_cells])
            active.append([pre, pre_post, post, post_pre])
            if (conddf.optoep.values[ii]>1):
                axcom2.scatter(coms1[differentially_inactivated_cells]-rewlocs[comp[0]], coms2[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='red')
                # axcom2.scatter(coms1_early[differentially_inactivated_cells]-rewlocs[comp[0]], coms2_early[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='blue')                                      
            elif (conddf.optoep.values[ii]<2):
                axcom.scatter(coms1[differentially_inactivated_cells]-rewlocs[comp[0]], coms2[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='black')       
                # axcom.scatter(coms1_early[differentially_inactivated_cells]-rewlocs[comp[0]], coms2_early[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='blue')       
            if (conddf.optoep.values[ii]>1):
                axcom4.scatter(coms1[differentially_activated_cells]-rewlocs[comp[0]], coms2[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='red')       
                # axcom4.scatter(coms1_early[differentially_activated_cells]-rewlocs[comp[0]], coms2_early[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='blue')       
            elif (conddf.optoep.values[ii]<2):
                axcom3.scatter(coms1[differentially_activated_cells]-rewlocs[comp[0]], coms2[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='black') 
                # axcom3.scatter(coms1_early[differentially_activated_cells]-rewlocs[comp[0]], coms2_early[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='blue')       
        else:   
            inactive.append([np.nan,np.nan,np.nan,np.nan])
            active.append([np.nan,np.nan,np.nan,np.nan])

axcom.plot(axcom.get_xlim(), axcom.get_ylim(), color='orange', linestyle='--')
axcom.axvline(0, color='yellow', linestyle='--')
axcom.axhline(0, color='yellow', linestyle='--')
axcom.spines['top'].set_visible(False)
axcom.spines['right'].set_visible(False)
axcom.set_xlabel('Prev Ep COM')
axcom.set_ylabel('Target Ep COM')
axcom.set_title('Inactivated cells, LED off')

axcom2.plot(axcom2.get_xlim(), axcom2.get_ylim(), color='k', linestyle='--')
axcom2.axvline(0, color='slategray', linestyle='--')
axcom2.axhline(0, color='slategray', linestyle='--')
axcom2.spines['top'].set_visible(False)
axcom2.spines['right'].set_visible(False)
axcom2.set_xlabel('Prev Ep COM')
axcom2.set_ylabel('Target Ep COM')
axcom2.set_title('Inactivated cells, LED on')

axcom3.plot(axcom3.get_xlim(), axcom3.get_ylim(), color='orange', linestyle='--')
axcom3.axvline(0, color='yellow', linestyle='--')
axcom3.axhline(0, color='yellow', linestyle='--')
axcom3.spines['top'].set_visible(False)
axcom3.spines['right'].set_visible(False)
axcom3.set_xlabel('Prev Ep COM')
axcom3.set_ylabel('Target Ep COM')
axcom3.set_title('Activated cells, LED off')

axcom4.plot(axcom4.get_xlim(), axcom4.get_ylim(), color='k', linestyle='--')
axcom4.axvline(0, color='slategray', linestyle='--')
axcom4.axhline(0, color='slategray', linestyle='--')
axcom4.spines['top'].set_visible(False)
axcom4.spines['right'].set_visible(False)
axcom4.set_xlabel('Prev Ep COM')
axcom4.set_ylabel('Target Ep COM')
axcom4.set_title('Activated cells, LED on')
#%%
inactive_opto = np.array(inactive)#[(conddf.optoep.values[(conddf.in_type.values=='vip')]<2), :]
active_opto = np.array(active)#[(conddf.optoep.values[(conddf.in_type.values=='vip')]<2), :]
intypecond = (conddf.in_type.values=='vip')
rewzones_comp = np.array(rewzones_comps)
df = pd.DataFrame(inactive_opto, columns = ['pre', 'pre_post', 'post', 'post_pre'])
df['opto'] = conddf.optoep.values[intypecond]>1
df['animal'] = conddf.animals.values[intypecond]
df['cond'] = ['inactive']*len(df)
df['rewzones_transition'] = [f'{int(xx[0])}_{int(xx[1])}' for xx in rewzones_comps]
df2 = pd.DataFrame(active_opto, columns = ['pre', 'pre_post', 'post', 'post_pre'])
df2['opto'] = conddf.optoep.values[intypecond]>1
df2['animal'] = conddf.animals.values[intypecond]
df2['cond'] = ['active']*len(df2)
df2['rewzones_transition'] = [f'{int(xx[0])}_{int(xx[1])}' for xx in rewzones_comps]
dforg = pd.concat([df,df2])
dforg.reset_index(drop=True, inplace=True) 
#%%  
plt.figure()
df = dforg.groupby(['animal', 'cond','opto']).mean()
quadrant = 'post_pre'
ax = sns.barplot(x="opto", y=quadrant,hue='cond', data=df,fill=False)
ax = sns.stripplot(x="opto", y=quadrant,hue='cond', data=df)
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

pre_post_1 = df.loc[(df.index.get_level_values('opto')==True) & (df.index.get_level_values('cond')=='active'), quadrant].values
pre_post_2 = df.loc[(df.index.get_level_values('opto')==False) & (df.index.get_level_values('cond')=='active'), quadrant].values

# pre_post_1 = df.loc[(df.opto==True) & (df.cond=='inactive'), quadrant].values
# pre_post_2 = df.loc[(df.opto==False) & (df.cond=='inactive'), quadrant].values


scipy.stats.ttest_ind(pre_post_1[~np.isnan(pre_post_1)], pre_post_2[~np.isnan(pre_post_2)])

# %%
