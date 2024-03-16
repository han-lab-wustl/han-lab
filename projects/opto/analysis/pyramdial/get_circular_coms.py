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
conddf = pd.read_csv(r"Z:\conddf_neural.csv", index_col=None)
#%%
figcom, axcom = plt.subplots()
for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    if conddf.animals.values[ii]=='e218' and conddf.optoep.values[ii]<0:
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc',
        'coms_circular_late_trials', 'tuning_curves_circular_late_trials'])
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: eptest = random.randint(2,3)    
        eps = np.where(changeRewLoc>0)[0]
        rewlocs = changeRewLoc[eps]*1.5
        rewzones = get_rewzones(rewlocs, 1.5)
        eps = np.append(eps, len(changeRewLoc))    
        if len(eps)<4: eptest = 2 # if no 3 epochs
        comp = [eptest-2,eptest-1] # eps to compare    
        coms = fall['coms_circular_late_trials'][0]
        coms = np.array([np.hstack(xx)-np.pi for xx in coms])
        bin_size = 3    
        # replace nan coms
        for jj,tc in enumerate(fall['tuning_curves_circular_late_trials'][0]):
            peak = np.nanmax(tc,axis=1)
            coms_max = np.array([np.where(tc[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
            coms[jj][np.isnan(coms[jj])]=coms_max[np.isnan(coms[jj])]

        axcom.scatter(coms[comp[0]], coms[comp[1]], s=5, color='black')       

axcom.plot(axcom.get_xlim(), axcom.get_ylim(), color='slategray', linestyle='--')
axcom.spines['top'].set_visible(False)
axcom.spines['right'].set_visible(False)
axcom.set_ylabel('Circular COM Ep 1')
axcom.set_xlabel('Circular COM Ep 2')
