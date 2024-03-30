"""
zahra
"""

import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy
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
pdst = r"Z:\E201_26_Apr_2023_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# fix vars
rewards = vralign["rewards"]
changeRewLoc = vralign["changeRewLoc"]
crl = consecutive_stretch_vralign(np.where(changeRewLoc>0)[0])
crl = np.array([min(xx) for xx in crl])
eps = np.array([xx for ii,xx in enumerate(crl[1:]) if np.diff(np.array([crl[ii-1],xx]))>1000])
eps = np.append(eps, 0)
eps = np.append(eps, len(changeRewLoc))
eps = np.sort(eps)
velocity = np.hstack(vralign['forwardvel'])
areas, areas_res = get_area_circumference_opto(pdst, range_val, binsize)
# plot
fig, axes = plt.subplots(nrows=(len(eps)-1), ncols=2)
for i in range(len(eps)-1):
    input_peri = areas_res[eps[i]:eps[i+1]]
    rewards_ = rewards[eps[i]:eps[i+1]]
    normmeanrew_t, meanrew, normrewall_t, \
    rewall = perireward_binned_activity(np.array(input_peri), \
                            rewards_.astype(int), 
                            vralign['timedFF'][eps[i]:eps[i+1]], range_val, binsize)
    ax = axes[i,0]
    ax.imshow(normrewall_t)
    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(np.arange(-range_val, range_val+1, 2))
    ax.set_title(f'Epoch {i+1}')
    ax.axvline(int(range_val/binsize), color = 'w', linestyle = '--')
    ax.axvline(int(range_val/binsize)+5, color = 'lightgray', linestyle = '--')
    ax = axes[i,1]
    ax.plot(normmeanrew_t)
    ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
    ax.set_xticklabels(np.arange(-range_val, range_val+1, 2))
    ax.axvline(int(range_val/binsize), color = 'k', linestyle = '--')
    ax.axvline(int(range_val/binsize)+5, color = 'gray', linestyle = '--')

fig.suptitle("Residual pupil area / trial")
