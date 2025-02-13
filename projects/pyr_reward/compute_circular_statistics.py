
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
coms_mean_rewrel = []
coms_mean_abs = []
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
                coms_mean_rewrel.append(com_mean_rewrel)
                coms_mean_abs.append(com_abs_mean)

pdf.close()
#%%
# plot r val distributions as a function of reward relative distance vs. track distance
dfc = conddf.copy()
dfc = dfc[((dfc.animals!='e217')) & (dfc.optoep<2)]

df = pd.DataFrame()
df['com_mean_rewardrel'] = np.concatenate(coms_mean_rewrel) # add pi to align with place map?
df['com_mean_abs'] = np.concatenate(coms_mean_abs)
df['meanangles_abs'] = np.concatenate([xx[0] for xx in meanangles_all])
df['meanangles_rewardrel'] = np.concatenate([xx[1] for xx in meanangles_all])
df['rval_abs'] = np.concatenate([xx[0] for xx in rvals_all])
df['rval_rewardrel'] = np.concatenate([xx[1] for xx in rvals_all])
df['animal'] = np.concatenate([[xx]*len(coms_mean_rewrel[ii]) for ii,xx in enumerate(dfc.animals.values)])
df['days'] = np.concatenate([[xx]*len(coms_mean_rewrel[ii]) for ii,xx in enumerate(dfc.days.values)])

df.to_csv(r'C:\Users\Han\Desktop\circular_stats.csv')
#%%

# Create a 2D density plot
fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='com_mean_rewardrel', y='rval_rewardrel', data=df, cmap="Purples", 
        fill=True, thresh=0)
ax.axvline(0, color='k', linestyle='--')
ax.axvline(-np.pi/4, color='r', linestyle='--')
ax.axvline(np.pi/4, color='r', linestyle='--')
ax.set_xlabel("Reward-relative distance")
ax.set_ylabel("r value")
ax.set_title(f"Reward-relative map")
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='com_mean_abs', y='rval_abs', data=df,cmap="Blues", fill=True, thresh=0)
ax.axvline(0, color='k', linestyle='--')
ax.axvline(270, color='k', linestyle='--')
plt.xlabel("Allocentric distance")
ax.set_ylabel("r value")
plt.title(f"Place map")
plt.show()
#%%
for animal in df.animal.unique():
        # Create a 2D density plot
        fig, ax = plt.subplots(figsize=(6,5))
        sns.kdeplot(x='com_mean_rewardrel', y='rval_rewardrel', data=df[df.animal==animal], cmap="Purples", 
                fill=True, thresh=0)
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel("Reward-relative distance")
        ax.set_ylabel("r value")
        ax.set_title(f"{animal}, Reward-relative map")
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))
        sns.kdeplot(x='com_mean_abs', y='rval_abs', data=df[df.animal==animal],cmap="Blues", fill=True, thresh=0)
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(270, color='k', linestyle='--')
        plt.xlabel("Allocentric distance")
        ax.set_ylabel("r value")
        plt.title(f"{animal}, Place map")
        plt.show()
#%%
# proportion of cells that have r > .8
thres=.99
rewrel_consistent_all = []
for animal in df.animal.unique():
        dfan=df[df.animal==animal]
        days = dfan.days.unique()
        rewrel_consistent_=[]
        for day in days:
                dfandy = dfan[dfan.days==day]
                rewrel_consistent = len(dfandy['com_mean_rewardrel'][dfandy['rval_rewardrel']>thres])/len(dfandy['com_mean_rewardrel'])
                com_rewrel_consistent = np.nanmean(dfandy['com_mean_rewardrel'][dfandy['rval_rewardrel']>thres].values)

                rewrel_consistent_.append([rewrel_consistent,com_rewrel_consistent])
        rewrel_consistent_all.append([np.nanmean(np.array([xx[0] for xx in rewrel_consistent_])),
                np.nanmean(np.array([xx[1] for xx in rewrel_consistent_]))])

abs_consistent_all = []
for animal in df.animal.unique():
        dfan=df[df.animal==animal]
        days = dfan.days.unique()
        abs_consistent_=[]
        for day in days:
                dfandy = dfan[dfan.days==day]
                rewrel_consistent = len(dfandy['com_mean_abs'][dfandy['rval_abs']>thres])/len(dfandy['com_mean_abs'])
                com_rewrel_consistent = np.nanmean(dfandy['com_mean_abs'][dfandy['rval_abs']>thres].values)

                abs_consistent_.append([rewrel_consistent,com_rewrel_consistent])
        abs_consistent_all.append([np.nanmean(np.array([xx[0] for xx in abs_consistent_])),np.nanmean(np.array([xx[1] for xx in abs_consistent_]))])

#%%
dfp = pd.DataFrame()
dfp['proportion'] = np.concatenate([[xx[0] for xx in rewrel_consistent_all], [xx[0] for xx in abs_consistent_all]])
dfp['alignment_type'] = np.concatenate([['reward-centric']*len([xx[0] for xx in rewrel_consistent_all]), ['allocentric']*len([xx[0] for xx in abs_consistent_all])])
dfp['mean_com'] = np.concatenate([[xx[1] for xx in rewrel_consistent_all], [xx[1] for xx in abs_consistent_all]])
dfp = dfp[dfp.proportion>0]
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='alignment_type', y='proportion',data=dfp,color='k',s=10,alpha=.7)
sns.barplot(x='alignment_type', y='proportion',data=dfp,color='k',fill=False)
ax.spines[['top','right']].set_visible(False)

x1 = dfp.loc[dfp.alignment_type=='reward-centric', 'proportion']
x2 = dfp.loc[dfp.alignment_type=='allocentric', 'proportion']
t,pval = scipy.stats.ranksums(x1,x2)
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='alignment_type', y='mean_com',data=dfp,color='k',s=10,alpha=.7)
sns.barplot(x='alignment_type', y='mean_com',data=dfp,color='k',fill=False)

#%%
# polar plot per session
def plot_polar_mean_angle(mean_angle, R, ax1, ax2, com):
        """
        Plots a polar plot with the mean firing angle and resultant vector length (R).

        Parameters:
        - mean_angle: Circular mean in radians.
        - R: Resultant vector length (0 to 1, representing concentration).
        """
        ax=ax1
        # Plot the mean angle as a vector
        ax.arrow(mean_angle, 0, 0, R, 
                head_width=0.1, head_length=0.1, fc='r', ec='r', linewidth=2)
        # Compute text position slightly beyond the arrow tip
        text_x = mean_angle
        text_y = R + 0.2  # Offset the text slightly outside the arrow tip

        # Add text next to the arrow
        ax.text(text_x, text_y, f"{com:.1f}", color='red', fontsize=12, 
                ha='center', va='bottom', fontweight='bold')

        # Set radial limits and labels
        ax.set_ylim(0, 1)  # Since R ranges from 0 to 1
        ax.set_yticklabels([])  # Hide radial labels
        ax.set_title("Polar Plot of Mean Angle and R")

        # Convert radians to degrees for readable angle labels
        ax.set_xticks(np.linspace(0, 2*np.pi, 8))  # 8 Major Ticks
        ax.set_xticklabels([f"{int(np.degrees(a))}°" for a in np.linspace(0, 2*np.pi, 8)])


# Example values
mean_angle = df['meanangles_rewardrel'].values[:30]  # -45 degrees
R = df['rval_rewardrel'].values[:30]  # High concentration
coms = df['com_mean_abs'].values[:30]
fig, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})

for ii,ma in enumerate(mean_angle):
        plot_polar_mean_angle(ma, R[ii], ax1, ax2, coms[ii])