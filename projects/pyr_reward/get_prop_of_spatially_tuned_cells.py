
"""
zahra
2025
get # of spatially tuned cells
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
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_relative_correcttr_skewfilt.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
spatial_tuned_per_ep_all = []
rates_all = []
goal_window_cm = 20
lasttr=8 # last trials
bins=90
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"

# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks', 'putative_pcs'])
        putative_pcs=np.vstack(fall['putative_pcs'][0])
        
        # get prop of spatially tuned cells per ep
        spatial_tuned_per_ep = [sum(xx)/len(xx) for xx in putative_pcs]
        spatial_tuned_per_ep=np.array(spatial_tuned_per_ep)
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        licks=fall['licks'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            licks=licks[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))

        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []
        total_trials_all = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            total_trials_all.append(total_trials)
            rates.append(success/total_trials)
        rate=np.nanmean(np.array(rates))
        total_trials_all=np.array(total_trials_all)
        # only get for epochs trials > 10
        rates_all.append(np.array(rates)[total_trials_all>10])
        spatial_tuned_per_ep_all.append(spatial_tuned_per_ep[total_trials_all>10])
#%%
plt.rc('font', size=20)          # controls default text sizes
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2)]
df = pd.DataFrame()
df['rates_all']=np.concatenate(rates_all)
df['spatial_tuned_per_ep_all']=np.concatenate(spatial_tuned_per_ep_all)*100
df['epoch']=np.concatenate([np.arange(len(xx))+1 for xx in rates_all])
df=df[df.epoch<5]

fig,ax=plt.subplots(figsize=(4,5))
sns.boxplot(x='epoch', y='spatial_tuned_per_ep_all', data=df, hue='epoch', palette=sns.color_palette('colorblind'))
# sns.stripplot(x='epoch', y='spatial_tuned_per_ep_all', data=df)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('% of Spatially tuned cells')
ax.set_xlabel('Epoch #')

plt.savefig(os.path.join(savedst, 'p_spatially_tuned_cells.svg'),bbox_inches='tight')

#%%
# not signficant!!
# ANOVA results:
#                 sum_sq     df         F    PR(>F)
# C(epoch)   1461.124075    3.0  1.877126  0.132943
# Residual  97816.848799  377.0       NaN       NaN

# Post-hoc t-tests (Bonferroni corrected):
# 1 vs 2: raw p = 0.3027, corrected p = 1.0000, significant: False
# 1 vs 3: raw p = 0.9944, corrected p = 1.0000, significant: False
# 1 vs 4: raw p = 0.0428, corrected p = 0.2569, significant: False
# 2 vs 3: raw p = 0.4017, corrected p = 1.0000, significant: False
# 2 vs 4: raw p = 0.0292, corrected p = 0.1753, significant: False
# 3 vs 4: raw p = 0.1168, corrected p = 0.7008, significant: False