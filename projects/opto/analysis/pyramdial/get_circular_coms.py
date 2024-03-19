"""plot circular coms
"""


import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto
import matplotlib.backends.backend_pdf

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\conddf_neural_com.csv", index_col=None)
#%%
figcom, axcom = plt.subplots()
figcom2, axcom2 = plt.subplots()

figcom3, axcom3 = plt.subplots()
figcom4, axcom4 = plt.subplots()
for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    if conddf.in_type.values[ii]=='vip':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_circular_early_trials',
            'tuning_curves_pc_circular_late_trials', 'coms_pc_circular_late_trials'])
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: eptest = random.randint(2,3)    
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))    
        if len(eps)<4: eptest = 2 # if no 3 epochs
        comp = [eptest-2,eptest-1] # eps to compare    
        bin_size = 3    
        tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_pc_circular_early_trials'][0][comp[0]]]))
        tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_pc_circular_early_trials'][0][comp[1]]]))
        tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_pc_circular_late_trials'][0][comp[0]]]))
        tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_pc_circular_late_trials'][0][comp[1]]]))        
        # Find differentially inactivated cells
        # differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
        # differentially_activated_cells = find_differentially_activated_cells(tc1_late[:, :int(rewlocs[comp[0]]/bin_size)], tc2_late[:, :int(rewlocs[comp[1]]/bin_size)], threshold, bin_size)
        differentially_inactivated_cells = find_differentially_inactivated_cells(tc1_late, tc2_late, threshold, bin_size)
        differentially_activated_cells = find_differentially_activated_cells(tc1_late, tc2_late, threshold, bin_size)
        coms = fall['coms_pc_circular_late_trials'][0]
        coms = np.array([np.hstack(xx)-np.pi for xx in coms])
        coms1 = coms[comp[0]]
        coms2 = coms[comp[1]]
        # # replace nan coms
        # for jj,tc in enumerate(fall['tuning_curves_circular_late_trials'][0]):
        #     peak = np.nanmax(tc,axis=1)
        #     coms_max = np.array([np.where(tc[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
        #     coms[jj][np.isnan(coms[jj])]=coms_max[np.isnan(coms[jj])]
        if (conddf.optoep.values[ii]>1):
            axcom2.scatter(coms1[differentially_inactivated_cells], coms2[differentially_inactivated_cells], s=2, color='red')       
        elif (conddf.optoep.values[ii]<2):
            axcom.scatter(coms1[differentially_inactivated_cells], coms2[differentially_inactivated_cells], s=2, color='black')       
        if (conddf.optoep.values[ii]>1):
            axcom4.scatter(coms1[differentially_activated_cells], coms2[differentially_activated_cells], s=2, color='red')       
        elif (conddf.optoep.values[ii]<2):
            axcom3.scatter(coms1[differentially_activated_cells], coms2[differentially_activated_cells], s=2, color='black')       

axcom.plot(axcom.get_xlim(), axcom.get_ylim(), color='orange', linestyle='--')
axcom.axvline(0, color='yellow', linestyle='--')
axcom.axhline(0, color='yellow', linestyle='--')
axcom.spines['top'].set_visible(False)
axcom.spines['right'].set_visible(False)
axcom.set_ylabel('Circular COM Ep 2')
axcom.set_xlabel('Circular COM Ep 1')
axcom.set_title('Inactivated cells, LED off')

axcom2.plot(axcom2.get_xlim(), axcom2.get_ylim(), color='k', linestyle='--')
axcom2.axvline(0, color='slategray', linestyle='--')
axcom2.axhline(0, color='slategray', linestyle='--')
axcom2.spines['top'].set_visible(False)
axcom2.spines['right'].set_visible(False)
axcom2.set_ylabel('Circular COM Ep 2')
axcom2.set_xlabel('Circular COM Ep 1')
axcom2.set_title('Inactivated cells, LED on')

axcom3.plot(axcom3.get_xlim(), axcom3.get_ylim(), color='orange', linestyle='--')
axcom3.axvline(0, color='yellow', linestyle='--')
axcom3.axhline(0, color='yellow', linestyle='--')
axcom3.spines['top'].set_visible(False)
axcom3.spines['right'].set_visible(False)
axcom3.set_ylabel('Circular COM Ep 2')
axcom3.set_xlabel('Circular COM Ep 1')
axcom3.set_title('Activated cells, LED off')

axcom4.plot(axcom4.get_xlim(), axcom4.get_ylim(), color='k', linestyle='--')
axcom4.axvline(0, color='slategray', linestyle='--')
axcom4.axhline(0, color='slategray', linestyle='--')
axcom4.spines['top'].set_visible(False)
axcom4.spines['right'].set_visible(False)
axcom4.set_ylabel('Circular COM Ep 2')
axcom4.set_xlabel('Circular COM Ep 1')
axcom4.set_title('Activated cells, LED on')
