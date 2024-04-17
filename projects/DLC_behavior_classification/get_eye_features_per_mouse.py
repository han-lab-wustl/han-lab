#%%
import pickle, os, sys, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np, scipy
from eye import get_area_circumference_from_vralign, perireward_binned_activity, consecutive_stretch, nan_helper, get_area_circumference_from_vralign_with_fails
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# path to pickle
pdst = r"D:\PupilTraining-Matt-2023-07-07\failed_trials\hide\E217_02_Feb_2024_vr_dlc_align.p"
vrfl = r"D:\PupilTraining-Matt-2023-07-07\failed_trials\hide\E217_02_Feb_2024_time(13_53_42).mat"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
rewards = vralign["rewards"]
range_val = 10 #s
binsize = 0.1 #s
areas, areas_res, circumferences, centroids_x, centroids_y, \
        meanrew, rewall, meanlicks, meanvel = get_area_circumference_from_vralign(pdst, range_val, binsize)

# get success and fail trials (excludes probes)
areas, areas_res, circumferences, meanrew, rewall, \
meanrewfail, rewallfail = get_area_circumference_from_vralign_with_fails(pdst, vrfl,
        range_val, binsize)

vralign['areas_residual'] = areas_res
with open(pdst, "wb") as fp: #unpickle
        pickle.dump(vralign, fp)
# fix nans
nans, x= nan_helper(rewall)
rewall[nans]= np.interp(x(nans), x(~nans), rewall[~nans])
nans, x= nan_helper(rewallfail)
rewallfail[nans]= np.interp(x(nans), x(~nans), rewallfail[~nans])
#%%
# plot peri reward
fig, axes=plt.subplots(2,1,sharex=True)
scaler = MinMaxScaler(feature_range=(0, 1))
# normalize
trials_norm = rewall#scaler.fit_transform(rewall)
meanrew_norm = meanrew#scaler.fit_transform(meanrew.reshape(-1,1))
# currently plotting non normalized
im = axes[0].imshow(trials_norm.T, cmap = 'cividis')
axes[0].set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
axes[0].set_xticklabels(np.arange(-range_val, range_val+1, 2))
axes[0].set_title("Residual pupil area / trial")
for i in range(trials_norm.T.shape[0]):
        axes[1].plot(trials_norm.T[i,:], color='slategray', alpha=0.3)        
axes[1].plot(np.hstack(meanrew_norm), color='k')
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')
axes[1].set_xlabel('Time from CS (s)')
axes[0].axvline(int(range_val/binsize), color = 'w', linestyle = '--')
axes[0].axvline(int(range_val/binsize)+5, color = 'lightgrey', linestyle = '--')
axes[1].axvline(int(range_val/binsize), color = 'k', linestyle = '--')
axes[1].axvline(int(range_val/binsize)+5, color = 'gray', linestyle = '--')
axes[1].set_ylim(-70,70)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes('right', size='5%', pad=0)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.show()
fig.tight_layout()
#%%
# plot failed trials
# plot peri reward
fig, axes=plt.subplots(2,1,sharex=True)
scaler = MinMaxScaler(feature_range=(0, 1))
# normalize
trials_norm = rewallfail#scaler.fit_transform(rewallfail)
meanrew_norm = meanrewfail#scaler.fit_transform(meanrewfail.reshape(-1,1))
# currently plotting non normalized
im = axes[0].imshow(trials_norm, cmap = 'cividis')
axes[0].set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
axes[0].set_xticklabels(np.arange(-range_val, range_val+1, 2))
axes[0].set_title("Residual pupil area / failed trial (centered at rewloc)")
for i in range(trials_norm.shape[0]):
        axes[1].plot(trials_norm[i,:], color='slategray', alpha=0.3)        
axes[1].plot(meanrew_norm, color='k')
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')
axes[1].set_xlabel('Time from Center of RL (s)')
axes[0].axvline(int(range_val/binsize), color = 'w', linestyle = '--')
axes[1].axvline(int(range_val/binsize), color = 'k', linestyle = '--')
axes[1].set_ylim(-70,70)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes('right', size='5%', pad=0)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.show()
fig.tight_layout()
#%%
licks = vralign['lickVoltage']<=-0.065 # manually threshold licks bc interpolation messes up lick binary
normmeanlicks_t, meanlicks, normlickall_t, \
lickall = perireward_binned_activity(licks, \
                rewards.astype(int),
                vralign['timedFF'], range_val, binsize)
velocity = vralign['forwardvel']
nans, x= nan_helper(vralign['forwardvel'])
velocity[nans]= np.interp(x(nans), x(~nans), vralign['forwardvel'][~nans])

normmeanvel_t, meanvel, normvelall_t, \
velall = perireward_binned_activity(velocity, \
                rewards.astype(int), 
                vralign['timedFF'], range_val, binsize)
#%%
# plot all
fig, axes=plt.subplots(3,1,sharex=True)
axes[0].imshow(rewall.T, cmap='cividis')
axes[1].imshow(normlickall_t, cmap="Reds")
# plt.figure(); plt.imshow(normvelall_t, cmap="Greys")
axes[2].imshow(normvelall_t, cmap="Greys")
axes[0].axvline(int(range_val/binsize), color = 'w', linestyle = '--')
axes[0].axvline(int(range_val/binsize)+5, color = 'lightgrey', linestyle = '--')
axes[1].axvline(int(range_val/binsize), color = 'k', linestyle = '--')
axes[1].axvline(int(range_val/binsize)+5, color = 'slategray', linestyle = '--')

axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[2].set_xlabel('Time from CS (s)')
axes[2].axvline(int(range_val/binsize), color = 'k', linestyle = '--')
axes[2].axvline(int(range_val/binsize)+5, color = 'slategray', linestyle = '--')

fig.suptitle('Pupil residual, licks, and velocity')
fig.tight_layout()
#%%
# look at centroid
input_peri = centroids_x
range_val=10; binsize=0.1 # s
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
axes[0].set_xticks(range(0, (int(range_val/binsize)*2)+1,40))
axes[0].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].plot(np.hstack(meanrew_norm))
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,40))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')


# ffmpeg -i \\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos\230621_E200.avi -c:v rawvideo \\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos\230621_E200.avi_conv.avi