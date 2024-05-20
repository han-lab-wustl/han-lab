
# look at enriched cells across rewlocs
# only in control days
import matplotlib.backends.backend_pdf

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
from sklearn.cluster import KMeans
import seaborn as sns
from placecell import get_rewzones, find_differentially_activated_cells, \
find_differentially_inactivated_cells, convert_com_to_radians, get_pyr_metrics_opto, intersect_arrays

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone

# import condition df
conddf = pd.read_csv(r"Z:\conddf_activated.csv", index_col=None)

pdf = matplotlib.backends.backend_pdf.PdfPages(r'Z:\enriched_tuning_curves_per_animal.pdf')
figures = False
frac_act = []; frac_inact = []; rewzone_transitions = []
for dd,day in enumerate(conddf.days.values):
    animal = conddf.animals.values[dd]
    track_length = 270
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials', 'pyr_tc_s2p_cellind'])
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]
    if conddf.optoep.values[dd]<2: eptest = random.randint(2,3)    
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    rewzones = get_rewzones(rewlocs, 1.5)
    rewzone_transitions.append(rewzones)
    eps = np.append(eps, len(changeRewLoc))    
    if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    
    bin_size = 3    
    tcs_early = []; tcs_late = []
    for ii,tc in enumerate(fall['tuning_curves_early_trials'][0]):
        tcs_early.append(np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_early_trials'][0][ii]])))
        tcs_late.append(np.squeeze(np.array([pd.DataFrame(xx).rolling(3).mean().values for xx in fall['tuning_curves_late_trials'][0][ii]])))
    tcs_early = np.array(tcs_early)
    tcs_late = np.array(tcs_late)
    coms = fall['coms'][0]
    # replace nan coms
    for tt in range(len(tcs_early)):
        peak = np.nanmax(tcs_late[tt],axis=1)
        coms[tt] = np.hstack(coms[tt])
        coms_max = np.array([np.where(tcs_late[tt][ii,:]==peak[ii])[0][0] for ii in range(len(peak))])
        coms[tt][np.isnan(coms[tt])]=coms_max[np.isnan(coms[tt])]
        
    threshold=5
    differentially_activated_cells = [find_differentially_activated_cells(tcs_early[tt], tcs_late[tt], threshold, bin_size) for tt in range(len(tcs_early))]
    differentially_inactivated_cells = [find_differentially_inactivated_cells(tcs_early[tt], tcs_late[tt], threshold, bin_size) for tt in range(len(tcs_early))]
    same_activated_cells = intersect_arrays(*differentially_activated_cells)
    same_inactivated_cells = intersect_arrays(*differentially_inactivated_cells)
    frac_same_activated_compared_to_mean_act = len(same_activated_cells)/np.mean(np.array([len(xx) for xx in differentially_activated_cells]))
    frac_same_inactivated_compared_to_mean_inact = len(same_inactivated_cells)/np.mean(np.array([len(xx) for xx in differentially_inactivated_cells]))
    frac_act.append(frac_same_activated_compared_to_mean_act)
    frac_inact.append(frac_same_inactivated_compared_to_mean_inact)
    if figures:
        # compare early vs. late tuning
        arr_early = tc1_early[differentially_activated_cells]
        arr_late = tc1_late[differentially_activated_cells]
        arr_early = arr_early[np.argsort(coms1[differentially_activated_cells])]
        arr_late = arr_late[np.argsort(coms1[differentially_activated_cells])]
        fig, axes = plt.subplots(2,1)
        axes[0].imshow(arr_early); axes[0].axvline(rewlocs[comp[0]]/bin_size, color = 'w')
        axes[1].imshow(arr_late); axes[1].axvline(rewlocs[comp[0]]/bin_size, color = 'w'); 
        axes[1].set_xlabel('Spatial bins')
        axes[0].set_ylabel('Cells')
        fig.suptitle(f"{animal}, day {day}, \n Enriched cells TC1, early vs. late trials")
        pdf.savefig(fig)
        # compare activated cells in next epoch
        arr_early = tc2_early[differentially_activated_cells]
        arr_late = tc2_late[differentially_activated_cells]
        arr_early = arr_early[np.argsort(coms2[differentially_activated_cells])]
        arr_late = arr_late[np.argsort(coms2[differentially_activated_cells])]
        fig, axes = plt.subplots(2,1)
        axes[0].imshow(arr_early); axes[0].axvline(rewlocs[comp[1]]/bin_size, color = 'w')
        axes[1].imshow(arr_late); axes[1].axvline(rewlocs[comp[1]]/bin_size, color = 'w'); 
        axes[1].set_xlabel('Spatial bins')
        axes[0].set_ylabel('Cells')
        fig.suptitle(f"{animal}, day {day}, \n Enriched cells TC2, early vs. late trials")
        pdf.savefig(fig)
        # compare early vs. late tuning - deenrichment
        arr_early = tc1_early[differentially_inactivated_cells]
        arr_late = tc1_late[differentially_inactivated_cells]
        arr_early = arr_early[np.argsort(coms1[differentially_inactivated_cells])]
        arr_late = arr_late[np.argsort(coms1[differentially_inactivated_cells])]
        fig, axes = plt.subplots(2,1)
        axes[0].imshow(arr_early); axes[0].axvline(rewlocs[comp[0]]/bin_size, color = 'w')
        axes[1].imshow(arr_late); axes[1].axvline(rewlocs[comp[0]]/bin_size, color = 'w'); 
        axes[1].set_xlabel('Spatial bins')
        axes[0].set_ylabel('Cells')
        fig.suptitle(f"{animal}, day {day}, \n De-enriched cells TC1, early vs. late trials")
        pdf.savefig(fig)
        # compare activated cells in next epoch
        arr_early = tc2_early[differentially_inactivated_cells]
        arr_late = tc2_late[differentially_inactivated_cells]
        arr_early = arr_early[np.argsort(coms2[differentially_inactivated_cells])]
        arr_late = arr_late[np.argsort(coms2[differentially_inactivated_cells])]
        fig, axes = plt.subplots(2,1)
        axes[0].imshow(arr_early); axes[0].axvline(rewlocs[comp[1]]/bin_size, color = 'w')
        axes[1].imshow(arr_late); axes[1].axvline(rewlocs[comp[1]]/bin_size, color = 'w'); 
        axes[1].set_xlabel('Spatial bins')
        axes[0].set_ylabel('Cells')
        fig.suptitle(f"{animal}, day {day}, \n De-enriched cells TC2, early vs. late trials")
        pdf.savefig(fig)
        plt.close('all')

pdf.close()
#%%
# plot

# plot fraction of activated cells that are the same across eps
optoep = conddf.optoep.values; animals = conddf.animals.values; in_type = conddf.in_type.values
conddf['frac_same_active'] = frac_act
conddf['frac_same_inactive'] = frac_inact
rz = []
for rzt in rewzone_transitions:
    string_list = [str(xx.astype(int)) for xx in rzt]
    rz.append('_'.join(string_list))
conddf['rz_transition'] = rz

# look at ctrl days
df = conddf#[conddf.optoep.values<2]
# df = conddf.groupby(['animals', 'rz_transition']).mean()
df = df.sort_values('rz_transition')
fig,ax = plt.subplots()
ax = sns.stripplot(x = 'rz_transition', y='frac_same_active', hue = 'animals', data=df)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelrotation=90)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# %%
