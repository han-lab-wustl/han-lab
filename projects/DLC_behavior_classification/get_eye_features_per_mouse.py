import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign
pdst = [path to pickle]
gainf, rewsize = 3/2, 20
areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = get_area_circumference_from_vralign(pdst, gainf, rewsize)

open vralign pickle

plt.figure()
areas = scipy.ndimage.gaussian_filter(areas,2)
r = np.random.randint(1000, len(areas))

plt.plot(areas[r:r+1000])
plt.plot((vralign['rewards']==0.5)[r:r+1000])