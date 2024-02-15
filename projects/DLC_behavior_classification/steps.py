import pickle, matplotlib.pyplot as plt
import scipy, matplotlib.pyplot as plt, re
import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy, matplotlib.pyplot as plt, re
import h5py, pickle
import matplotlib

vralign = r"D:\Tail_E186\E200_14_Mar_2023_vr_dlc_align.p"
with open(vralign, "rb") as fp: #unpickle
    vralign = pickle.load(fp)
vralign.keys()
def get_dist(x, y):
    dist = ((y[0] - y[1])**2 + (x[0] - x[1])**2)**0.5
    return int(dist)
dist = []
for i in range(len(vralign['BackLeftHeel_x'])):
    backleftheelx,backleftheely, = vralign['BackLeftHeel_x'][i], vralign['BackLeftHeel_y'][i]
    backrightheelx,backrightheely, = vralign['BackRightHeel_x'][i], vralign['BackRightHeel_y'][i]
    x = [backleftheelx, backrightheelx]
    y = [backleftheely, backrightheely]
    dist.append(get_dist(x, y))
fig, axs = plt.subplots()
axs.plot(area[10000:11000])
axs.plot((vralign['rewards']==0.5)[10000:11000]*3000)
reward = np.hstack(vralign['rewards'])
axs.plot(vralign['rewards'])
plt.figure(); plt.plot(area[10000:12000])
axs.plot(reward[10000:120000])
diff = abs(np.diff(dist))
plt.figure()
plt.plot(diff)

r = np.random.randint(1000, len(dist))
plt.figure()

plt.plot((vralign['rewards']==0.5)[r:r+1000]*200)
plt.plot(vralign['forwardvel'][r:r+1000]*2)
plt.plot(diff[r:r+1000]*3)