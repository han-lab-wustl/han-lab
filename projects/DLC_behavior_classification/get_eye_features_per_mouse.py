import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign, perireward_binned_activity, consecutive_stretch, nan_helper
import statsmodels.api as sm
#%%
pdst = r"I:\pupil_pickles\E200_13_May_2023_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# remember than after interpolating, rewards > 1 are now cs
gainf, rewsize = 3/2, 20
# rewards = vralign['rewards']==1
# cs = vralign['rewards']==0.5

#removes repeated frames of reward delivery (to not double the number of trials)
areas_, circumferences, centroids_x,\
        centroids_y = get_area_circumference_from_vralign(pdst, gainf, rewsize)
############## GLM ##############
# peri reward
areas = scipy.signal.savgol_filter(areas_,5, 2)
# remove artifact licks
vralign['licks'] = np.hstack(vralign['licks'])
vralign['licks'][:-10]=0
licks = scipy.signal.savgol_filter(vralign['licks'],10, 2) 
velocity = np.hstack(vralign['forwardvel'])
nans, x = nan_helper(velocity)
velocity[nans]= np.interp(x(nans), x(~nans), velocity[~nans])
velocity = scipy.signal.savgol_filter(velocity,10, 2)
acc_ = np.diff(velocity)/np.diff(np.hstack(vralign['timedFF']))
# pad nans
acc=np.zeros_like(velocity)
acc[:-1]=acc_
speed = abs(velocity)
eyelid = vralign['EyeNorthEast_y']
eyelid = scipy.signal.savgol_filter(eyelid,10, 2)
# removes repeated frames of reward delivery (to not double the number of trials)
X = np.array([velocity, speed, acc, licks, eyelid]).T # Predictor(s)
X = sm.add_constant(X) # Adds a constant term to the predictor(s)
y = areas # Outcome
# Fit a regression model
model = sm.GLM(y, X, family=sm.families.Gaussian())
result = model.fit()
areas_res = result.resid_pearson
# areas_pred = result.predict(X)
############## GLM ##############
# run peri reward time & plot
range_val = 15 #s
binsize = 0.1 #s
input_peri = areas_res
csind = consecutive_stretch(np.where(vralign["rewards"]>0)[0])
csind = np.array([min(xx) for xx in csind])
cs = np.zeros_like(vralign["rewards"])
cs[csind]=1
# TEMP
vralign["rewards"]=np.hstack(vralign["rewards"])
cs = vralign["rewards"]==0.5 # for old pickles
normmeanrew_t, meanrew, normrewall_t, \
rewall = perireward_binned_activity(np.array(input_peri), \
                        cs.astype(int), 
                        np.hstack(vralign['timedFF']), range_val, binsize)

#%%
# plot peri reward
fig, axes=plt.subplots(2,1)
axes[0].imshow(normrewall_t)
axes[0].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[0].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].plot(normmeanrew_t)
axes[1].set_xticks(range(0, (int(range_val/binsize)*2)+1,20))
axes[1].set_xticklabels(range(-range_val, range_val+1, 2))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('Mean of Trials')
