"""
plot tuning curves side by side
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
import seaborn as sns
import matplotlib.backends.backend_pdf
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.colors as colors
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
dd=4
track_length = 270
dct = {}
pc=True
animal = conddf.animals.values[dd]
day = conddf.days.values[dd]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
if not pc:
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials'])
    coms = fall['coms'][0]
    coms_early = fall['coms_early_trials'][0]
    tcs_early = fall['tuning_curves_early_trials'][0]
    tcs_late = fall['tuning_curves_late_trials'][0]
else:
    fall = scipy.io.loadmat(params_pth, variable_names=['coms_pc_late_trials', 'changeRewLoc', 'tuning_curves_pc_early_trials',\
        'tuning_curves_pc_late_trials', 'coms_pc_early_trials'])
    coms = fall['coms_pc_late_trials'][0]
    coms_early = fall['coms_pc_early_trials'][0]
    tcs_early = fall['tuning_curves_pc_early_trials'][0]
    tcs_late = fall['tuning_curves_pc_late_trials'][0]
changeRewLoc = np.hstack(fall['changeRewLoc'])
eptest = conddf.optoep.values[dd]    
eps = np.where(changeRewLoc>0)[0]
rewlocs = changeRewLoc[eps]*1.5
eps = np.append(eps, len(changeRewLoc))  
if conddf.optoep.values[dd]<2: 
    eptest = random.randint(2,3)      
    if len(eps)<4: eptest = 2 # if no 3 epochs
comp = [eptest-2,eptest-1] # eps to compare    
bin_size = 3    
tc1_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[0]]]))
tc2_early = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_early[comp[1]]]))
tc1_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[0]]]))
tc2_late = np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[comp[1]]]))    
# replace nan coms
# peak = np.nanmax(tc1_late,axis=1)
# coms1_max = np.array([np.where(tc1_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
# peak = np.nanmax(tc2_late,axis=1)
# coms2_max = np.array([np.where(tc2_late[ii,:]==peak[ii])[0][0] for ii in range(len(peak))])    
coms1 = np.hstack(coms[comp[0]])
coms2 = np.hstack(coms[comp[1]])
coms1_early = np.hstack(coms_early[comp[0]])
coms2_early = np.hstack(coms_early[comp[1]])

#%%
fig, axes = plt.subplots(ncols = len(tcs_late),sharex=True,sharey=True,
                    figsize=(6,6))
tc =  np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[0]]))    
vmin = np.nanmin(tc); vmax = np.nanmax(tc)-1.5 # get min max values to set
for ep in range(len(tcs_late)):
    if eptest-1==ep:
        axes[ep].set_title('VIP Inhibition')
    tc =  np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in tcs_late[ep]]))    
    axes[ep].imshow(tc[np.argsort(np.hstack(coms[0])),:], vmin=vmin, vmax=vmax)
    axes[ep].axvline(x=rewlocs[ep]/bin_size,color='w')
axes[len(tcs_late)-1].set_xlabel('Spatial bins')
axes[0].set_ylabel('Cells')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data'
plt.savefig(os.path.join(savedst, f"{animal}_{day}_pc_tuning_curve.svg"))