
"""
zahra
july 2024
quantify reward-relative cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.circular import get_circular_data
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'circular_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
meanangles_all = []
rvals_all = []
radian_alignment = {}
goal_cm_window = 20
lasttr=8 # last trials
bins=90
dists = []

# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                meanangles_abs,rvals_abs,meanangles_rad,rvals_rad,tc_mean,com_mean_rewrel,\
                tcs_abs_mean,com_abs_mean=get_circular_data(ii,params_pth,animal,day,bins,radian_alignment,
                        radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
                        goal_cell_null,pvals,total_cells,
                        num_iterations=1000)
                meanangles_all.append([meanangles_abs,meanangles_rad])
                rvals_all.append([rvals_abs,rvals_rad])

pdf.close()
#%%


def plot_polar_mean_angle(mean_angle, R):
    """
    Plots a polar plot with the mean firing angle and resultant vector length (R).
    
    Parameters:
    - mean_angle: Circular mean in radians.
    - R: Resultant vector length (0 to 1, representing concentration).
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plot the mean angle as a vector
    ax.arrow(mean_angle, 0, 0, R, 
             head_width=0.1, head_length=0.1, fc='r', ec='r', linewidth=2)

    # Set radial limits and labels
    ax.set_ylim(0, 1)  # Since R ranges from 0 to 1
    ax.set_yticklabels([])  # Hide radial labels
    ax.set_title("Polar Plot of Mean Angle and R")

    # Convert radians to degrees for readable angle labels
    ax.set_xticks(np.linspace(0, 2*np.pi, 8))  # 8 Major Ticks
    ax.set_xticklabels([f"{int(np.degrees(a))}°" for a in np.linspace(0, 2*np.pi, 8)])

    plt.show()

# Example values
mean_angle = -np.pi/4  # -45 degrees
R = 0.8  # High concentration

# Convert negative angles to [0, 2π] range
mean_angle = mean_angle % (2 * np.pi)

plot_polar_mean_angle(mean_angle, R)