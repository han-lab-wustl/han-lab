"""plot regular coms - rewloc
for opto vs. ledoff epochs
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto
import matplotlib.backends.backend_pdf
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com.csv", index_col=None)
#%%
figcom, axcom = plt.subplots()
figcom2, axcom2 = plt.subplots()
figcom3, axcom3 = plt.subplots()
figcom4, axcom4 = plt.subplots()

inactive = []
active = []
pre_post_tc = []
inactive_cells_remap = []
inactive_cells_stable = []
active_cells_remap = []
active_cells_stable = []
rewzones_comps = []

for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    if True:#conddf.in_type.values[ii]=='vip': #and conddf.animals.values[ii]=='e218':#and conddf.optoep.values[ii]==2:# and conddf.animals.values[ii]=='e218':
        plane=0 #TODO: make modular        
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{plane}_Fall.mat"
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
        threshold=7
        differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late, tc2_late, threshold, bin_size)
        differentially_activated_cells = find_differentially_activated_cells(tc1_late,tc2_late, threshold, bin_size)
        # coms = fall['coms_pc_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]        
        coms1 = np.hstack(coms[comp[0]])
        coms2 = np.hstack(coms[comp[1]])
        coms1_early = np.hstack(coms_early[comp[0]])
        coms2_early = np.hstack(coms_early[comp[1]])
        com_remap = (coms1-rewlocs[comp[0]])-(coms2-rewlocs[comp[1]])
        window = 5 # cm
        # get proportion of remapping vs. stable cells
        remap = np.where((com_remap<window) & (com_remap>-window))[0]
        remap = np.array([cl for cl in remap if np.nanmax(tc1_late[cl,:])>0.2])
        stable = np.where(((coms1-coms2)<window) & ((coms1-coms2)>-window))[0]
        stable = np.array([cl for cl in stable if np.nanmax(tc1_late[cl,:])>0.2])
        inactive_stable = [xx for xx in stable if xx in differentially_inactivated_cells]
        if len(inactive_stable)>0:
            inactive_stable_prop = len(inactive_stable)/len(differentially_inactivated_cells)
        else:
            inactive_stable_prop = 0
        inactive_remap = [xx for xx in remap if xx in differentially_inactivated_cells]
        if len(inactive_remap)>0:
            inactive_remap_prop = len(inactive_remap)/len(differentially_inactivated_cells)
        else:
            inactive_remap_prop = 0
    
        active_stable = [xx for xx in stable if xx in differentially_activated_cells]
        if len(active_stable)>0:
            active_stable_prop = len(active_stable)/len(differentially_activated_cells)
        else:
            active_stable_prop = 0
        active_remap = [xx for xx in remap if xx in differentially_activated_cells]
        if len(active_remap)>0:
            active_remap_prop = len(active_remap)/len(differentially_activated_cells)
        else:
            active_remap_prop = 0       
        active_cells_stable.append(active_stable_prop)
        active_cells_remap.append(active_remap_prop)

        stable_prop = len(stable)/len(coms1)
        remap_prop = len(remap)/len(coms1)
        inactive_cells_stable.append([stable_prop, inactive_stable_prop])
        inactive_cells_remap.append([remap_prop, inactive_remap_prop])
        # TODO: look at trial by trial tuning
        if conddf.animals.values[ii]=='e216':
            # per trial examples
            # for cl in inactive_remap:            
            #     if np.nanmax(tc1_late[cl,:])>0.2:
            #         fig, ax = plt.subplots()           
            #         ax.plot(tc1_late[cl,:],color='k',label='previous_ep')
            #         ax.plot(tc2_late[cl,:],color='red',label='led_on')
                    
            #         ax.axvline(rewlocs[comp[0]]/bin_size,color='k', linestyle='dotted')
            #         ax.axvline(rewlocs[comp[1]]/bin_size,color='red', linestyle='dotted')
                    
            #         # ax.set_axis_off()  
            #         ax.set_title(f'Goal remap cell \n animal: {animal}, day: {day}, optoep: {conddf.optoep.values[ii]}')
            #         ax.spines['top'].set_visible(False)
            #         ax.spines['right'].set_visible(False) 
            #         ax.legend()
            # for cl in inactive_stable:
            #     if np.nanmax(tc1_late[cl,:])>0.2:
            #         fig, ax = plt.subplots()           
            #         ax.plot(tc1_late[cl,:],color='k',label='previous_ep')
            #         ax.plot(tc2_late[cl,:],color='red',label='led_on')
                    
            #         ax.axvline(rewlocs[comp[0]]/bin_size,color='k', linestyle='dotted')
            #         ax.axvline(rewlocs[comp[1]]/bin_size,color='red', linestyle='dotted')
                    
            #         # ax.set_axis_off()  
            #         ax.set_title(f'Stable tuning cell \n animal: {animal}, day: {day}, optoep: {conddf.optoep.values[ii]}')
            #         ax.spines['top'].set_visible(False)
            #         ax.spines['right'].set_visible(False) 
            #         ax.legend()
        # # replace nan coms
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
inactive_cells_remap = np.array(inactive_cells_remap)#[(conddf.optoep.values[(conddf.in_type.values=='vip')]<2), :]
inactive_cells_stable = np.array(inactive_cells_stable)#[(conddf.optoep.values[(conddf.in_type.values=='vip')]<2), :]
rewzones_comp = np.array(rewzones_comps)
df = pd.DataFrame(inactive_cells_remap[:,1], columns = ['inactive_remap_prop'])
df['remap_prop'] = inactive_cells_remap[:,0]
df['stable_prop'] = inactive_cells_stable[:,0]
df['inactive_stable_prop'] = inactive_cells_stable[:,1]
df['active_stable_prop'] = active_cells_stable
df['active_remap_prop'] = active_cells_remap
df['opto'] = conddf.optoep.values>1
df['animal'] = conddf.animals.values
cond = conddf.in_type.values=="vip"
cond = [['vip']*xx if xx==True else ['ctrl'] for xx in cond]
df['cond'] = np.concatenate(cond)
df['rewzones_transition'] = [f'{int(xx[0])}_{int(xx[1])}' for xx in rewzones_comps]

fig, ax = plt.subplots()
ax.scatter(conddf.days.values[conddf.optoep.values<2], inactive_cells_remap[:,0][conddf.optoep.values<2], color='limegreen', label='goal_remapping')
ax.scatter(conddf.days.values[conddf.optoep.values<2], inactive_cells_stable[:,0][conddf.optoep.values<2], color='k', label='track_relative')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Days of recording')
ax.set_ylabel('Proportion of all cells')

df = df.groupby(['animal', 'opto', 'cond']).mean(numeric_only=True)

# diff_remap_prop_ctrl = (df.loc[((df.index.get_level_values('opto')==True) & (df.index.get_level_values('cond')==False)), 
#         'inactive_remap_prop']).values-(df.loc[((df.index.get_level_values('opto')==False) & (df.index.get_level_values('cond')==False)), 'inactive_remap_prop']).values
# diff_remap_prop_vip = (df.loc[((df.index.get_level_values('opto')==True) & (df.index.get_level_values('cond')==True)), 
#         'inactive_remap_prop']).values-(df.loc[((df.index.get_level_values('opto')==False) & (df.index.get_level_values('cond')==True)), 'inactive_remap_prop']).values
# df2 = pd.DataFrame()
# df2['diff_remap_prop'] = np.concatenate([diff_remap_prop_ctrl, diff_remap_prop_vip])
# df2['condition'] = np.concatenate([['ctrl']*len(diff_remap_prop_ctrl), ['vip']*len(diff_remap_prop_vip)])
# plt.figure()
# ax = sns.barplot(x="condition", y='diff_remap_prop',hue='condition', data=df2,fill=False)
# ax = sns.stripplot(x="condition", y='diff_remap_prop',hue='condition', data=df2)
# ax.tick_params(axis='x', labelrotation=90)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# scipy.stats.ttest_ind(diff_remap_prop_ctrl, diff_remap_prop_vip)

# #%%
# diff_stable_prop_ctrl = (df.loc[((df.index.get_level_values('opto')==True) & (df.index.get_level_values('cond')==False)), 
#         'inactive_stable_prop']).values-(df.loc[((df.index.get_level_values('opto')==False) & \
#         (df.index.get_level_values('cond')==False)), 'inactive_stable_prop']).values
# diff_stable_prop_vip = (df.loc[((df.index.get_level_values('opto')==True) & \
#     (df.index.get_level_values('cond')==True)), 
#         'inactive_stable_prop']).values-(df.loc[((df.index.get_level_values('opto')==False) & (df.index.get_level_values('cond')==True)), 'inactive_stable_prop']).values
# df2 = pd.DataFrame()
# df2['diff_stable_prop'] = np.concatenate([diff_stable_prop_ctrl, diff_stable_prop_vip])
# df2['condition'] = np.concatenate([['ctrl']*len(diff_remap_prop_ctrl), ['vip']*len(diff_remap_prop_vip)])
# plt.figure()
# ax = sns.barplot(x="condition", y='diff_stable_prop',hue='condition', data=df2,fill=False)
# ax = sns.stripplot(x="condition", y='diff_stable_prop',hue='condition', data=df2)
# ax.tick_params(axis='x', labelrotation=90)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

plt.figure()
ax = sns.barplot(x="opto", y='inactive_stable_prop',hue='cond', data=df,fill=False,
        errorbar='se',palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='inactive_stable_prop',hue='cond', data=df,
    palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figure()
ax = sns.barplot(x="opto", y='inactive_remap_prop',hue='cond', data=df,fill=False,
    errorbar='se', palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='inactive_remap_prop',hue='cond', data=df,
    palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figure()
ax = sns.barplot(x="opto", y='active_stable_prop',hue='cond', data=df,fill=False,
        errorbar='se',palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='active_stable_prop',hue='cond', data=df,
    palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figure()
ax = sns.barplot(x="opto", y='active_remap_prop',hue='cond', data=df,fill=False,
    errorbar='se', palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='active_remap_prop',hue='cond', data=df,
    palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.figure()
ax = sns.barplot(x="opto", y='remap_prop',hue='cond', data=df,fill=False,
        errorbar='se',palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='remap_prop',hue='cond', data=df,
        palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


plt.figure()
ax = sns.barplot(x="opto", y='stable_prop',hue='cond', data=df,fill=False,
        errorbar='se',palette={'ctrl': "slategray", 'vip': "red"})
ax = sns.stripplot(x="opto", y='stable_prop',hue='cond', data=df,palette={'ctrl': "slategray", 'vip': "red"})
ax.tick_params(axis='x', labelrotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

inactive_remap_t = df.loc[((df.index.get_level_values('opto') == True) & (df.index.get_level_values('cond') == True)), 'inactive_remap_prop'].values
inactive_remap_f = df.loc[((df.index.get_level_values('opto') == False) & (df.index.get_level_values('cond') == True)), 'inactive_remap_prop'].values
scipy.stats.ttest_ind(inactive_remap_t, inactive_remap_f)

# %%
