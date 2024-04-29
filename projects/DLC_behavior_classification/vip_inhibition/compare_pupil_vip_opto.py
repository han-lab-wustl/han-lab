"""
zahra
"""

import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy, pandas as pd, seaborn as sns
import eye
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from eye import consecutive_stretch_vralign, get_area_circumference_opto, perireward_binned_activity
# path to pickle
pdst = r"D:\PupilTraining-Matt-2023-07-07\opto-pickles\E218_06_Dec_2023_vr_dlc_align.p"
range_val=5
binsize=0.05
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# fix vars
rewards = vralign["rewards"]
changeRewLoc = vralign["changeRewLoc"]
crl = consecutive_stretch_vralign(np.where(changeRewLoc>0)[0])
crl = np.array([min(xx) for xx in crl])
eps = np.array([xx for ii,xx in enumerate(crl[1:]) if np.diff(np.array([crl[ii],xx]))[0]>5000])
eps = np.append(eps, 0)
eps = np.append(eps, len(changeRewLoc))
eps = np.sort(eps)
velocity = np.hstack(vralign['forwardvel'])
areas, areas_res = get_area_circumference_opto(pdst, range_val, binsize)
# plot
fig, axes = plt.subplots(nrows=(len(eps)-1), ncols=2)
pre_reward = []
areas_per_ep = []
for i in range(len(eps)-1):
    input_peri = areas_res[eps[i]:eps[i+1]]
    areas_per_ep.append(input_peri)
    rewards_ = rewards[eps[i]:eps[i+1]]
    normmeanrew_t, meanrew, normrewall_t, \
    rewall = perireward_binned_activity(np.array(input_peri), \
                            rewards_.astype(int), 
                            vralign['timedFF'][eps[i]:eps[i+1]], range_val, binsize)
    pre_reward.append(np.nanmean(rewall[:int(range_val/binsize),:],axis=0))
    ax = axes[i,0]
    ax.imshow(rewall.T)
    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(np.arange(-range_val, range_val+1))
    ax.set_title(f'Epoch {i+1}')
    ax.axvline(int(range_val/binsize), color = 'w', linestyle = '--')
    ax.axvline(int(range_val/binsize)+10, color = 'lightgray', linestyle = '--')
    ax = axes[i,1]
    ax.plot(meanrew)
    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(np.arange(-range_val, range_val+1))
    ax.axvline(int(range_val/binsize), color = 'k', linestyle = '--')
    ax.axvline(int(range_val/binsize)+10, color = 'gray', linestyle = '--')
fig.suptitle("Residual pupil area / trial")

mean_areas_per_ep = [np.nanmean(xx) for xx in areas_per_ep]
std_areas_per_ep = [np.nanstd(xx) for xx in areas_per_ep]
# all ep
plt.figure()
plt.bar(range(0, len(mean_areas_per_ep)), mean_areas_per_ep, color = 'slategray')
plt.errorbar(range(0, len(mean_areas_per_ep)), mean_areas_per_ep, yerr=std_areas_per_ep,
            color='k')

comparison = [1,2] # change per session
#remember to use 0 indexing
fig, ax = plt.subplots()
df = pd.DataFrame()
df['pupil'] = np.concatenate([pre_reward[comparison[0]], pre_reward[comparison[1]]])
df['condition'] = np.concatenate([['previous_epoch']*len(pre_reward[comparison[0]]), ['opto_epoch']*len(pre_reward[comparison[1]])])
ax = sns.stripplot(x='condition', y='pupil', data=df, color = 'k')
ax = sns.barplot(x='condition', y='pupil', data=df, color = 'k', fill=False)
# comp
mean_comp = np.array(mean_areas_per_ep)[comparison]
std_comp = np.array(std_areas_per_ep)[comparison]
plt.figure()
plt.bar(range(0, len(mean_comp)), mean_comp, color = 'slategray')
plt.errorbar(range(0, len(std_comp)), mean_comp, yerr=std_comp,
            color='k')
