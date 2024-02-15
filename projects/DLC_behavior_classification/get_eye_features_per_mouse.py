import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign, perireward_binned_activity
import statsmodels.api as sm
def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
                - y, 1d numpy array with possible NaNs
        Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices= index(logical_indices),
                to convert logical indices of NaNs to 'equivalent' indices
        Example:
                # linear interpolation of NaNs
                nans, x= nan_helper(y)
                y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

pdst = r"I:\vids_to_analyze\face_and_pupil\E200_09_May_2023_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# remember than after interpolating, rewards > 1 are now cs
gainf, rewsize = 3/2, 20
areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
        normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
        lickall, normmeanvel_t, meanvel, normvelall_t, \
        velall = get_area_circumference_from_vralign(pdst, gainf, rewsize)
############## GLM ##############
areas = scipy.ndimage.gaussian_filter(areas,1.5)
# Assuming `df` is a pandas DataFrame with your data,
# where "Behavior" is the binary outcome, and "Velocity" is your predictor
X = vralign['forwardvel'] # Predictor(s)
nans, x = nan_helper(X)
X[nans]= np.interp(x(nans), x(~nans), X[~nans])

X = sm.add_constant(X) # Adds a constant term to the predictor(s)
y = areas # Outcome
# Fit a regression model
model = sm.GLM(y, X, family=sm.families.Gaussian())
result = model.fit()
areas_res = result.resid_pearson
# areas_pred = result.predict(X)
############## GLM ##############
# run peri reward time & plot
range_val = 10 #s
binsize = 0.05 #s
input_peri = areas_res
normmeanrew_t, meanrew, normrewall_t, \
rewall = perireward_binned_activity(np.array(input_peri), \
                        (vralign['rewards']>1).astype(int), 
                        vralign['timedFF'], range_val, binsize)

normmeanlicks_t, meanlicks, normlickall_t, \
lickall = perireward_binned_activity(vralign['licks'], \
                (vralign['rewards']>1).astype(int), 
                vralign['timedFF'], range_val, binsize)
normmeanvel_t, meanvel, normvelall_t, \
velall = perireward_binned_activity(vralign['forwardvel'], \
                (vralign['rewards']>1).astype(int), 
                vralign['timedFF'], range_val, binsize)

plt.figure(); plt.imshow(normrewall_t)
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