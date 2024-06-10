"""
calculate proportion of goal cells
zahra
june 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.backends.backend_pdf
from itertools import combinations
import matplotlib as mpl
from placecell import intersect_arrays
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_modeling.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data'
savepth = os.path.join(savedst, 'goal_cells_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%

goal_cell_iind = []
goal_cell_prop = []
num_epochs = []
pvals = []
for ii in range(len(conddf)):
    animal = conddf.animals.values[ii]
    day = conddf.days.values[ii]
    plane=0 #TODO: make modular  
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{plane}_Fall.mat"
    # fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_pc_early_trials',
    #     'tuning_curves_pc_late_trials', 'coms_pc_late_trials', 'coms_pc_early_trials'])
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 'tuning_curves_early_trials',
        'tuning_curves_late_trials', 'coms', 'coms_early_trials'])        
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[ii]
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    eps = np.append(eps, len(changeRewLoc))    
    bin_size = 3    
    tcs_early = fall['tuning_curves_early_trials'][0]
    tcs_late = fall['tuning_curves_late_trials'][0]
    # coms_early = fall['coms_pc_early_trials'][0]
    coms = fall['coms'][0]
    coms_early = fall['coms_early_trials'][0]    
    window = 20 # cm
    goal_window = 10 # cm
    coms = np.array([np.hstack(xx) for xx in coms])
    # relative to reward
    coms_rewrel = np.array([com-rewlocs[ii] for ii, com in enumerate(coms)])                 
    perm = list(combinations(range(len(coms)), 2))     
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    # get goal cells across all epochs
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    goal_cells = intersect_arrays(*com_goal)
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms[0])
    goal_cell_prop.append(goal_cell_p)
    num_epochs.append(len(coms))
    colors = ['navy', 'red', 'green', 'k','darkorange']
    for gc in goal_cells:
        fig, ax = plt.subplots()
        for ep in range(len(coms)):
            ax.plot(tcs_late[ep][gc,:], label=f'epoch {ep}', color=colors[ep])
            ax.axvline(rewlocs[ep]/bin_size, color=colors[ep])
            ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iterations
    num_iterations = 3000
    shuffled_dist = np.zeros((num_iterations))
    for i in range(num_iterations):
        rewlocs_shuf = [random.randint(100,250) for iii in range(len(eps))]
        # relative to reward
        coms_rewrel = np.array([com-rewlocs_shuf[ii] for ii, com in enumerate(coms)])             
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        goal_cells_shuf = intersect_arrays(*com_goal)
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value)
    print(p_value)

pdf.close()
# %%
df = conddf
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='condition',data=df[df.opto==False],
        s=10)
ax.spines[['top','right']].set_visible(False)

#%%
# histograms of p-values
# shuffled reward loc
fig,ax = plt.subplots()
ax.hist(pvals,bins=80)
ax.axvline(0.05,color='k',linewidth=2,linestyle='--')
ax.set_ylabel('Sessions')
ax.set_xlabel('P-value')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Reward cell proportion compared\nto shuffled rew loc cells')
# %%
df['p_value'] = pvals

dfagg = df.groupby(['animals', 'opto', 'condition']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.barplot(x='opto', y='goal_cell_prop',
        hue='condition',data=dfagg, 
        fill=False)
ax = sns.stripplot(x='opto', y='goal_cell_prop',
        hue='condition',data=dfagg, 
        s=10,dodge=True)
ax.spines[['top','right']].set_visible(False)
# ax.axhline(0.05,color='k',linewidth=2,linestyle='--')

