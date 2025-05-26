
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from reward_shuffle import get_com_v_persistence
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

#%%
####################################### RUN CODE #######################################
# initialize var
ep_dicts = []
bins = 90
goal_window_cm=20
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        ep_dict = get_com_v_persistence(params_pth, animal, day, ii)
        ep_dicts.append(ep_dict)
pdf.close()
####################################### RUN CODE #######################################
#%%
# key = ep
com_ep2_comb = [xx[1] for xx in ep_dicts if 1 in xx.keys()]
com_ep3_comb = [xx[2] for xx in ep_dicts if 2 in xx.keys()]
com_ep4_comb = [xx[3] for xx in ep_dicts if 3 in xx.keys()]
# com_ep5_comb = [xx[4] for xx in ep_dicts if 4 in xx.keys()]
#%%
from scipy.stats import gaussian_kde

# plot histograms
fig, ax = plt.subplots()
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
a = 0.2
lw = 3
labels = [2,3,4,5]
# Plot histogram and confidence intervals for each epoch
data_sets = [com_ep2_comb, com_ep3_comb, com_ep4_comb]
for i, data in enumerate(data_sets):
    all_data = np.concatenate(data)

    # Plot histogram
    ax.hist(
        all_data, bins=20, density=True, alpha=a,
        label=f'{labels[i]}, {len(all_data)} cells',
        color=colors[i], edgecolor=colors[i], linewidth=lw
    )

    # Smooth Gaussian KDE
    kde = gaussian_kde(all_data[np.isfinite(all_data)])  # remove NaNs
    x_vals = np.linspace(-np.pi, np.pi, 500)
    ax.plot(x_vals, kde(x_vals), color=colors[i], linewidth=2)

    # 95% CI lines
    ci_low = np.nanpercentile(all_data, 2.5)
    ci_high = np.nanpercentile(all_data, 97.5)

    vline_low = ax.axvline(ci_low, color=colors[i], linewidth=lw, linestyle='--',alpha=0.4)
    vline_low.set_dashes([10, 8])
    vline_high = ax.axvline(ci_high, color=colors[i], linewidth=lw, linestyle='--',alpha=0.4)
    vline_high.set_dashes([10, 8])


# # Add label for one CI line only
# ci_ref = np.concatenate(com_ep3_comb)
# ci_high = np.nanpercentile(ci_ref, 97.5)
# ax.axvline(ci_high, color=colors[1], linewidth=lw, linestyle='--', label='95% CI').set_dashes([10, 8])

# Style and labels
ax.set_ylabel('Relative cell density\n(across all sessions)')
ax.set_xticks([-np.pi, -np.pi/4,0, np.pi/4,np.pi])
ax.set_xticklabels(["$-\\pi$", '$-\\pi/4$', "0",  '$\\pi/4$', "$\\pi$"])
ax.set_xlabel('Reward-relative distance')
ax.spines[['top', 'right']].set_visible(False)

# Legend
h_strip, l_strip = ax.get_legend_handles_labels()
if ax.legend_: ax.legend_.remove()
ax.legend(
    h_strip, l_strip,
    title='Epoch #',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.
)
ax.axvline(0,color='dimgrey')

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
plt.savefig(os.path.join(savedst, 'com_hist_across_ep.svg'),bbox_inches='tight')