#%%
import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign, perireward_binned_activity, consecutive_stretch, nan_helper
import statsmodels.api as sm

pdst = r"D:\PupilTraining-Matt-2023-07-07\light world pickles\E217-updated\E217_18_Jan_2024_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)

range_val = 5
binsize = 0.05
areas, circumferences, centroids_x, centroids_y, \
        meanrew, rewall, meanlicks, meanvel = get_area_circumference_from_vralign(pdst, range_val, binsize)
# removes repeated frames of reward delivery (to not double the number of trials)
#%%
# plot peri reward
fig, axes=plt.subplots(2,1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
# normalize
trials_norm = scaler.fit_transform(rewall)
meanrew_norm = scaler.fit_transform(meanrew.reshape(-1,1))
axes[0].imshow(trials_norm.T)
axes[0].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[0].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].plot(np.hstack(meanrew_norm))
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')
#%%
# look at centroid
input_peri = centroids_x
rewards = vralign["rewards"]
range_val=5; binsize=0.05 # s
normmeanrew_t, meanrew, normrewall_t, \
rewall = perireward_binned_activity(np.array(input_peri), \
                        rewards.astype(int), 
                        vralign['timedFF'], range_val, binsize)

fig, axes=plt.subplots(2,1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
# normalize
trials_norm = scaler.fit_transform(rewall)
meanrew_norm = scaler.fit_transform(meanrew.reshape(-1,1))
axes[0].imshow(trials_norm.T)
# axes[0].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
# axes[0].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].plot(np.hstack(meanrew_norm))
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')

#%%

normmeanlicks_t, meanlicks, normlickall_t, \
lickall = perireward_binned_activity(vralign['licks'], \
                cs.astype(int),
                vralign['timedFF'], range_val, binsize)
normmeanvel_t, meanvel, normvelall_t, \
velall = perireward_binned_activity(vralign['forwardvel'], \
                cs.astype(int),
                vralign['timedFF'], range_val, binsize)
 
normmeanvel_t, meanvel, normvelall_t, \
velall = perireward_binned_activity(vralign['forwardvel'], \
                cs.astype(int), 
                vralign['timedFF'], range_val, binsize)
plt.figure(); plt.imshow(normlickall_t, cmap="Reds")
plt.figure(); plt.imshow(normvelall_t, cmap="Greys")
plt.figure(); plt.plot(normmeanrew_t)
#%%
plt.figure()
r = np.random.randint(1000, len(areas))
# plt.plot(areas[r:r+3000])
plt.plot(areas[r:r+3000]/7, 'k')
plt.plot(areas_res[r:r+3000], 'grey')
plt.plot(areas_pred[r:r+3000]/4, 'slategray')
plt.plot(vralign['forwardvel'][r:r+3000])

plt.plot(vralign['licks'][r:r+3000]*100)
plt.plot((vralign['rewards'])[r:r+3000]/8)
plt.plot((vralign['ybinned']<3)[r:r+3000]*120)
#ffmpeg -i I:\eye_videos\240120_E217.avi -c:v rawvideo I:\eye_videos\240120_E217_conv.avi

plt.figure()
plt.plot(areas)
plt.plot(vralign['forwardvel']*2)

plt.figure()
plt.plot(normmeanrew_t)
plt.figure()
plt.imshow(normrewall_t)
#plt.plot(normmeanvel_t)
plt.plot(normmeanlicks_t)

ffmpeg -i \\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos\230621_E200.avi -c:v rawvideo \\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos\230621_E200.avi_conv.avi