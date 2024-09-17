"""
aug 2024
"""

import os, sys, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
# formatting for figs
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
import matplotlib.backends.backend_pdf, matplotlib as mpl

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
savepth = os.path.join(savedst, 'sst_per_epoch_earlyvlate_allep.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late
# Define source directory and mouse name
srcs = [r'J:', r'F:']
mice = ['E135', 'E136']
days_all = [[1,2,3,4], [2, 3, 4, 5, 6, 7, 8]]
range_val, binsize = 8, 0.2 # s
for ii,days in enumerate(days_all):
    mouse_name = mice[ii]
    src = srcs[ii]
    for dy in days:
        # for e135
        if mouse_name=='E135' and dy == 4: planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
        else: planelut = {0: 'SR', 1: 'SP', 2: 'SO'}
        if mouse_name=='E135': day_dir = os.path.join(src, mouse_name, f'Day{dy}')
        else: day_dir = os.path.join(src, mouse_name, f'D{dy}')
        for root, dirs, files in os.walk(day_dir):
            for file in files:
                if 'plane' in root and file.endswith('Fall.mat'):
                    f = loadmat(os.path.join(root, file))
                    
                    # Extract necessary variables
                    dFF_iscell = f['dFF_iscell']
                    stat = f['stat'][0]
                    iscell = f['iscell'][:, 0].astype(bool)
                    eps = np.where(f['changeRewLoc']>0)[1]
                    eps = np.append(eps,dFF_iscell.shape[1])
                    rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0]>0]
                    # Determine the cells to keep, excluding merged ROIs
                    statiscell = [stat[i] for i in range(len(stat)) if iscell[i]]
                    garbage = []
                    for st in statiscell:
                        if 'imerge' in st.dtype.names and len(st['imerge'][0]) > 0:
                            garb =  st['imerge'][0].flatten().tolist()
                            garbage.extend(garb)
                    arr = [x[0] for x in garbage if len(x)>0]
                    if len(arr)>0:
                        garbage = np.unique(np.concatenate(arr))
                    else:
                        garbage = []
                    cllsind = np.arange(f['F'].shape[0])
                    cllsindiscell = cllsind[iscell]
                    keepind = ~np.isin(cllsindiscell, garbage)

                    # Filter dFF_iscell
                    dFF_iscell_filtered = dFF_iscell[keepind, :]
                    
                    # run glm
                    dff_res = []; perirew = []
                    
                    for cll in range(dFF_iscell_filtered.shape[0]):
                        # for uneven plane nums
                        X = np.array([f['forwardvel'][0]]).T # Predictor(s)
                        y = dFF_iscell_filtered[cll,:] # Outcome                        
                        if X.shape[0]>y.shape[0]: X = X[:-1]
                        X = sm.add_constant(X) # Adds a constant term to the predictor(s)                        
                        ############## GLM ##############
                        # Fit a regression model
                        model = sm.GLM(y, X, family=sm.families.Gaussian())
                        result = model.fit()
                        dff_res.append(result.resid_pearson)    
                        # peri reward
                        # dff = dFF_iscell_filtered[cll,:]
                        dff = result.resid_pearson
                        early_v_late = perireward_binned_activity_early_late(dff, 
                                (f['rewards'][0]==1).astype(int), 
                                f['timedFF'][0], f['trialnum'][0],range_val, binsize)
                        perirew.append([[early_v_late['first_5']['meanrewdFF'], 
                                early_v_late['first_5']['rewdFF']], [early_v_late['last_5']['meanrewdFF'], 
                                early_v_late['last_5']['rewdFF']]])
                    dff_res = np.array(dff_res)
                    # dff_res = dFF_iscell_filtered
                    
                    # Plotting
                    # mean image
                    fig, ax = plt.subplots()
                    ax.imshow(f['ops']['max_proj'][0][0]**10e-5, cmap='Greys_r')
                    ax.set_axis_off()
                    for cll in range(dff_res.shape[0]):
                        ycoord = f['stat'][0][cll][0][0][0]
                        ypos = np.median(ycoord)
                        xcoord = f['stat'][0][cll][0][0][1]
                        xpos = np.median(xcoord)
                        # coords = np.squeeze(np.array([xcoord, ycoord]).T)
                        # # Calculate the convex hull
                        # hull = ConvexHull(coords)
                        # for simplex in hull.simplices:
                        #     ax.plot(coords[simplex, 0], coords[simplex, 1], 'y-')
                        ax.text(xpos, ypos, f'Cell {cll+1}', color='w', fontsize=8)
                    fig.suptitle(f'{mouse_name}, Day={dy}, {planelut[int(root.split("plane")[1])]}')
                    pdf.savefig(fig)
                    plt.close(fig)

                    clls = dff_res.shape[0]
                    subpl = int(np.ceil(np.sqrt(clls)))
                    fig, axes = plt.subplots(subpl, subpl, figsize=(20, 10))
                    axes = axes.flatten()  # Flatten the array to easily index axes
                    for cll in range(clls):
                        axes[cll].plot(dff_res[cll, :])
                        axes[cll].set_title(f'Cell {cll + 1}')
                    # Hide any unused subplots
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')
                    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.suptitle(f'GLM residuals \n {mouse_name}, Day={dy}, {planelut[int(root.split("plane")[1])]}')
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                    subpl = int(np.ceil(np.sqrt(clls)))
                    fig, axes = plt.subplots(subpl, subpl, figsize=(30, 15))
                    axes = axes.flatten()  # Flatten the array to easily index axes
                    
                    for cll in range(clls):
                        ax = axes[cll]
                        # first 5 
                        meanrew = perirew[cll][0][0]
                        rewall = perirew[cll][0][1]
                        ax.plot(meanrew, 'slategray', label='early_trials')
                        ax.fill_between(
                            range(0, int(range_val / binsize) * 2),
                            meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            alpha=0.5, color='slategray'
                        )
                        # last 5
                        meanrew = perirew[cll][1][0]
                        rewall = perirew[cll][1][1]

                        ax.plot(meanrew, 'darkcyan', label='late_trials')
                        ax.fill_between(
                            range(0, int(range_val / binsize) * 2),
                            meanrew - scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            meanrew + scipy.stats.sem(rewall, axis=1, nan_policy='omit'),
                            alpha=0.5, color='darkcyan'
                        )                            
                        ax.set_title(f'Cell {cll + 1}')
                        ax.axvline(int(range_val / binsize), color='k', linestyle='--')
                        ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1, 20))
                        ax.set_xticklabels(np.arange(-range_val, range_val + 1, 4))
                        ax.spines[['top', 'right']].set_visible(False)
                        if cll==clls-1 and int(root.split("plane")[1])==0: ax.legend(bbox_to_anchor=(1.01, 1.05))
                    # Hide any unused subplots
                    for i in range(clls, len(axes)):
                        axes[i].axis('off')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust 'rect' to accommodate any titles if necessary
                    fig.suptitle(f'Peri-reward, \n {mouse_name}, Day={dy}, {planelut[int(root.split("plane")[1])]}')
                    plt.show()
                    pdf.savefig(fig)
                    plt.close(fig)

pdf.close()