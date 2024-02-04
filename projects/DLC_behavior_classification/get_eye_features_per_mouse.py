import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign
pdst = [r'D:\PupilTraining-Matt-2023-07-07\E228_20_Jan_2024_vr_dlc_align.p']
gainf, rewsize = 3/2, 20
areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = get_area_circumference_from_vralign(pdst, gainf, rewsize)
savedst = r"D:\PupilTraining-Matt-2023-07-07"
pdst = os.path.join(savedst, "E228_20_Jan_2024_vr_dlc_align.p")
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)

plt.figure()

areas = scipy.ndimage.gaussian_filter(areas,2)
r = np.random.randint(1000, len(areas))

plt.plot(areas[r:r+3000])
licks = np.hstack(vralign['licks']*500)
plt.plot(licks[r:r+3000])
plt.plot((vralign['rewards']==0.5)[r:r+3000]*600)
circumferences = scipy.ndimage.gaussian_filter(circumferences,2)
plt.plot(vralign['forwardvel'][r:r+3000])
#plt.plot(circumferences[r:r+1000])