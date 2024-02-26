"""zahra's dopamine hrz analysis
feb 2024
for chr2 experiments
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye

def consecutive_stretch(x):
    z = np.diff(x)
    break_point = np.where(z != 1)[0]

    if len(break_point) == 0:
        return [x]

    y = []
    if 0 in break_point: y.append([x[break_point[0]]]) # since it does not iterate from 0
    for i in range(1, len(break_point)):
        xx = x[break_point[i - 1] + 1:break_point[i]]
        if len(xx)==0: xx = [x[break_point[i]]]
        y.append(xx)
    y.append(x[break_point[-1] + 1:])
    
    return y
#%%
params_pth = r"Z:\chr2_grabda\e232\4\240221_ZD_000_002\suite2p\plane2\reg_tif\params.mat"
params = scipy.io.loadmat(params_pth)
params_keys = params.keys()
keys = params['params'].dtype
# dff is in row 7 - roibasemean3/basemean
dff = np.hstack(params['params'][0][0][6][0][0])/np.hstack(params['params'][0][0][9])
dffdf = pd.DataFrame({'dff': dff})
dff = np.hstack(dffdf.rolling(2).mean().values)
rewards = np.hstack(params['solenoid2'])
ybinned = np.hstack(params['ybinned'])/(2/3)

# plot pre-first reward dop activity
rewloc = 123*1.5
firstrew = np.where(rewards==1)[0][0]
rews_centered = np.zeros_like(ybinned[:firstrew])
rews_centered[(ybinned[:firstrew] > rewloc-2) & (ybinned[:firstrew] < rewloc+2)]=1
rews_iind = consecutive_stretch(np.where(rews_centered)[0])
min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
rews_centered = np.zeros_like(ybinned[:firstrew])
rews_centered[min_iind]=1
timedFF = np.hstack(params['timedFF'])

# # visualize
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(dff[:firstrew], 'k')
# ax2.plot(ybinned[:firstrew])


range_val = 5; binsize=0.2
normmeanrewdFF, meanrewdFF, normrewdFF, \
    rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, timedFF[:firstrew], range_val, binsize)

fig, ax = plt.subplots()
ax.imshow(normrewdFF)
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(range(-range_val, range_val+1, 1))
fig, ax = plt.subplots()
ax.plot(meanrewdFF)
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(range(-range_val, range_val+1, 1))

# all subsequent rews
range_val = 5; binsize=0.2
normmeanrewdFF, meanrewdFF, normrewdFF, \
    rewdFF = eye.perireward_binned_activity(dff, rewards, timedFF, range_val, binsize)

fig, ax = plt.subplots()
ax.imshow(normrewdFF)
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(range(-range_val, range_val+1, 1))

fig, ax = plt.subplots()
ax.plot(meanrewdFF)
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(range(-range_val, range_val+1, 1))
