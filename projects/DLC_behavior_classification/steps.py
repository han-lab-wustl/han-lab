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
#axs.plot(area[10000:11000])
axs.plot((vralign['rewards']==0.5)[10000:11000]*3000)
reward = np.hstack(vralign['rewards'])
axs.plot(vralign['rewards'])
#plt.figure(); plt.plot(area[10000:12000])
axs.plot(reward[10000:120000])
diff = abs(np.diff(dist))
plt.figure()
plt.plot(diff)

r = np.random.randint(1000, len(dist))
plt.figure()

plt.plot((vralign['rewards']==0.5)[r:r+1000]*200)
plt.plot(vralign['forwardvel'][r:r+1000]*2)
plt.plot(diff[r:r+1000]*3)

#slope of back

def get_average_slope(x1, y1, x2, y2, x3, y3):
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y3 - y2) / (x3 - x2)
    slope3 = (y3 - y1) / (x3 - x1)
    average_slope = (slope1 + slope2 + slope3) / 3
    return int(average_slope)
avg_slope = []
# Example usage
x1, y1 = 1, 2
x2, y2 = 3, 4
x3, y3 = 5, 6
avg_slope = average_slope(x1, y1, x2, y2, x3, y3)
print("Average slope:", avg_slope)
#slope of upper back
vralign['UpperBack_x'][vralign['UpperBack_likelihood'].astype('float32')<0.9]=np.nan
vralign['UpperBack_y'][vralign['UpperBack_likelihood'].astype('float32')<0.9]=np.nan
vralign['MidBack_x'][vralign['MidBack_likelihood'].astype('float32')<0.9]=np.nan
vralign['MidBack_y'][vralign['MidBack_likelihood'].astype('float32')<0.9]=np.nan
def get_slope(x, y):
    slope = (y[0] - y[1]) /(x[0] - x[1])
    return slope
slope = []
for i in range(len(vralign['UpperBack_x'])):
    upperbackx,upperbacky, = vralign['UpperBack_x'][i], vralign['UpperBack_y'][i]
    midbackx,midbacky, = vralign['MidBack_x'][i], vralign['MidBack_y'][i]
    x = [upperbackx, midbackx]
    y = [upperbacky, midbacky]
    slope.append(get_slope(x, y))
#slope of lower back
def get_slope_1(x1, y1):
    slope_1 = (y1[0] - y1[1]) /(x1[0] - x1[1])
    return int(slope_1)
slope_1 = []
for i in range(len(vralign['LowerBack_x'])):
    lowerbackx,lowerbacky, = vralign['LowerBack_x'][i], vralign['LowerBack_y'][i]
    midbackx,midbacky, = vralign['MidBack_x'][i], vralign['MidBack_y'][i]
    x1 = [midbackx,lowerbackx]
    y1 = [midbacky,lowerbacky]
    slope_1.append(get_slope_1(x1, y1))
plt.figure()
plt.plot(slope_1)

r = np.random.randint(1000, len(slope))
plt.figure()

plt.plot((vralign['rewards']==0.5)[r:r+1000]*10)
plt.plot(vralign['forwardvel'][r:r+1000]/4)
plt.plot((slope)[r:r+1000])


def get_avg_tail_x(x):
    avg_tail_x = (x[0] + x[1] + x[2]) / 3
    return avg_tail_x
def get_avg_tail_y(y):
    avg_tail_y = (y[0] + y[1] + y[2]) / 3
    return avg_tail_y
avg_tail_x = []
avg_tail_y =[]
slope_1 = []
for i in range(len(vralign['TailBase_x'])):
    tailbasex,tailbasey, = vralign['TailBase_x'][i], vralign['TailBase_y'][i]
    midtailx,midtaily, = vralign['MidTail_x'][i], vralign['MidTail_y'][i]
    tailtipx,tailtipy, = vralign['TailTip_x'][i], vralign['TailTip_y'][i]
    x = [midtailx,tailbasex, tailtipx]
    y = [midtaily,tailbasey, tailtipx]
    avg_tail_x.append(get_avg_tail_x(x))
    avg_tail_y.append(get_avg_tail_y(y))
angles = []
for i in range(len(avg_tail_x)):
    angle = np.arctan2(avg_tail_y[i], avg_tail_x[i])
    angles.append(angle)


plt.figure()
plt.plot(avg_tail_x)
plt.figure()
plt.plot(avg_tail_y)
plt.plot()
plt.plot(angles)