"""identify subsets of cells based on tuning properties
zahra
april 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
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
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savepth = r'Z:\opto_analysis_stable_vs_remap_all_an.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# import raw data
with open("Z:\dcts_com_opto_inference_wcomp.p", "rb") as fp: #unpickle
        dcts = pickle.load(fp)

#%%
figcom, axcom = plt.subplots()
figcom2, axcom2 = plt.subplots()
figcom3, axcom3 = plt.subplots()
figcom4, axcom4 = plt.subplots()

pre_post_tc = []
inactive_cells_remap = []
inactive_cells_stable = []
active_cells_remap = []
active_cells_stable = []
rewzones_comps = []

for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    if True:# conddf.in_type.values[ii]=='vip': #and conddf.animals.values[ii]=='e218':#and conddf.optoep.values[ii]==2:# and conddf.animals.values[ii]=='e218':
        plane=0 #TODO: make modular  
        dct = dcts[ii]      
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{plane}_Fall.mat"
        # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
        #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_early_trials',
            'tuning_curves_late_trials', 'coms', 'coms_early_trials'])        
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[ii]
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)        
        eps = np.append(eps, len(changeRewLoc))    
        comp = dct['comp'] # eps to compare 
        other_eps = [xx for xx in range(len(eps)-1) if xx not in comp]   
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
        differentially_inactivated_cells = dct['inactive']
        differentially_activated_cells = dct['active']
        # coms = fall['coms_pc_late_trials'][0]
        # coms_early = fall['coms_pc_early_trials'][0]
        coms = fall['coms'][0]
        coms_early = fall['coms_early_trials'][0]        
        coms1 = np.hstack(coms[comp[0]])
        coms2 = np.hstack(coms[comp[1]])
        coms1_early = np.hstack(coms_early[comp[0]])
        coms2_early = np.hstack(coms_early[comp[1]])
        com_remap = (coms1-rewlocs[comp[0]])-(coms2-rewlocs[comp[1]])
        window = 30 # cm
        goal_window = 10 # cm
        # get proportion of remapping vs. stable cells
        remap = np.where((com_remap<goal_window) & (com_remap>-goal_window))[0]
        remap = np.array([cl for cl in remap if np.nanmax(tc1_late[cl,:])>0.2])
        stable = np.where(((coms1-coms2)<window) & ((coms1-coms2)>-window))[0]
        stable = np.array([cl for cl in stable if np.nanmax(tc1_late[cl,:])>0.2])
        # get others?
        
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
        # tuning curves for goal remap cells 
        for other_ep in other_eps:
            if len(remap)>0:
                tc_other = tcs_late[other_ep]
                coms_other = coms[other_ep]
                tc_other = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tc_other]))        
                arr = tc_other[remap]
                tc3 = arr[np.argsort(coms1[remap])] # np.hstack(coms_other)
                arr = tc2_late[remap]    
                tc2 = arr[np.argsort(coms1[remap])]
                arr = tc1_late[remap]
                tc1 = arr[np.argsort(coms1[remap])]
                fig, ax1 = plt.subplots()            
                if other_ep>comp[1]:
                    ax1.imshow(np.concatenate([tc1,tc2,tc3]))
                    ax1.axvline(rewlocs[comp[0]]/bin_size, color='w')
                    ax1.axvline(rewlocs[comp[1]]/bin_size, color='w', linestyle='--')
                    ax1.axvline(rewlocs[other_ep]/bin_size, color='w', linestyle='dotted')
                    ax1.axhline(tc1.shape[0], color='yellow')
                    ax1.axhline(tc1.shape[0]+tc2.shape[0], color='yellow')
                    ax1.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[ii]} \n goal remapping cells')
                    ax1.set_ylabel('Cells')
                    ax1.set_xlabel('Spatial bins (3cm)')
                    fig.tight_layout()                
                else:
                    ax1.imshow(np.concatenate([tc3,tc1,tc2]))
                    ax1.axvline(rewlocs[other_ep]/bin_size, color='w')
                    ax1.axvline(rewlocs[comp[0]]/bin_size, color='w', linestyle='--')
                    ax1.axvline(rewlocs[comp[1]]/bin_size, color='w', linestyle='dotted')                    
                    ax1.axhline(tc3.shape[0], color='yellow')
                    ax1.axhline(tc3.shape[0]+tc1.shape[0], color='yellow')
                    ax1.set_title(f'animal: {animal}, day: {day}, optoep: {conddf.optoep.values[ii]}\n goal remapping cells')
                    ax1.set_ylabel('Cells')
                    ax1.set_xlabel('Spatial bins (3cm)')
                    fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)    
            # per cell examples
            for cl in remap:            
                if np.nanmax(tc1_late[cl,:])>0.2:
                    fig, ax = plt.subplots()           
                    ax.plot(tc1_late[cl,:],color='k',label='previous_ep')
                    ax.plot(tc1_early[cl,:],color='k',label='previous_ep_early',linestyle='--')
                    ax.plot(tc2_late[cl,:],color='red',label='led_on')
                    ax.plot(tc2_early[cl,:],color='red',label='led_on_early',linestyle='--')
                    ax.axvline(rewlocs[comp[0]]/bin_size,color='k', linestyle='dotted')
                    ax.axvline(rewlocs[comp[1]]/bin_size,color='red', linestyle='dotted')
                    
                    # ax.set_axis_off()  
                    ax.set_title(f'Distance-to-goal cell \n animal: {animal}, day: {day}, optoep: {conddf.optoep.values[ii]}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False) 
                    ax.legend()
                    pdf.savefig(fig)
                    plt.close(fig)
            # for cl in stable:
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
                    ax.legend()
        # # replace nan coms
        if len(remap)>0 and len(stable)>0:
            if (conddf.optoep.values[ii]>1):
                axcom2.scatter(coms1[remap]-rewlocs[comp[0]], coms2[remap]-rewlocs[comp[1]], s=4, color='red')
                # axcom2.scatter(coms1_early[differentially_inactivated_cells]-rewlocs[comp[0]], coms2_early[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='blue')                                      
            elif (conddf.optoep.values[ii]<2):
                axcom.scatter(coms1[remap]-rewlocs[comp[0]], coms2[remap]-rewlocs[comp[1]], s=4, color='black')       
                # axcom.scatter(coms1_early[differentially_inactivated_cells]-rewlocs[comp[0]], coms2_early[differentially_inactivated_cells]-rewlocs[comp[1]], s=4, color='blue')       
            if (conddf.optoep.values[ii]>1):
                axcom4.scatter(coms1[stable]-rewlocs[comp[0]], coms2[stable]-rewlocs[comp[1]], s=4, color='red')       
                # axcom4.scatter(coms1_early[differentially_activated_cells]-rewlocs[comp[0]], coms2_early[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='blue')       
            elif (conddf.optoep.values[ii]<2):
                axcom3.scatter(coms1[stable]-rewlocs[comp[0]], coms2[stable]-rewlocs[comp[1]], s=4, color='black') 
                # axcom3.scatter(coms1_early[differentially_activated_cells]-rewlocs[comp[0]], coms2_early[differentially_activated_cells]-rewlocs[comp[1]], s=4, color='blue')       

axcom.plot(axcom.get_xlim(), axcom.get_ylim(), color='orange', linestyle='--')
axcom.axvline(0, color='yellow', linestyle='--')
axcom.axhline(0, color='yellow', linestyle='--')
axcom.spines['top'].set_visible(False)
axcom.spines['right'].set_visible(False)
axcom.set_xlabel('Prev Ep COM')
axcom.set_ylabel('Target Ep COM')
axcom.set_title('Distance-to-goal coding, LED off')
figcom.tight_layout()
axcom2.plot(axcom2.get_xlim(), axcom2.get_ylim(), color='k', linestyle='--')
axcom2.axvline(0, color='slategray', linestyle='--')
axcom2.axhline(0, color='slategray', linestyle='--')
axcom2.spines['top'].set_visible(False)
axcom2.spines['right'].set_visible(False)
axcom2.set_xlabel('Prev Ep COM')
axcom2.set_ylabel('Target Ep COM')
axcom2.set_title('Distance-to-goal coding, LED on')
figcom2.tight_layout()
axcom3.plot(axcom3.get_xlim(), axcom3.get_ylim(), color='orange', linestyle='--')
axcom3.axvline(0, color='yellow', linestyle='--')
axcom3.axhline(0, color='yellow', linestyle='--')
axcom3.spines['top'].set_visible(False)
axcom3.spines['right'].set_visible(False)
axcom3.set_xlabel('Prev Ep COM')
axcom3.set_ylabel('Target Ep COM')
axcom3.set_title('Stable place coding, LED off')
figcom3.tight_layout()
axcom4.plot(axcom4.get_xlim(), axcom4.get_ylim(), color='k', linestyle='--')
axcom4.axvline(0, color='slategray', linestyle='--')
axcom4.axhline(0, color='slategray', linestyle='--')
axcom4.spines['top'].set_visible(False)
axcom4.spines['right'].set_visible(False)
axcom4.set_xlabel('Prev Ep COM')
axcom4.set_ylabel('Target Ep COM')
axcom4.set_title('Stable place coding, LED on')
figcom4.tight_layout()

pdf.savefig(figcom)
pdf.savefig(figcom2)
pdf.savefig(figcom3)
pdf.savefig(figcom4)

pdf.close()
plt.close('all')