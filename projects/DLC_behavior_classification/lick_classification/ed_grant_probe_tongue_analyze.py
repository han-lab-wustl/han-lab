import pickle, matplotlib.pyplot as plt
import scipy, matplotlib.pyplot as plt, re
import os, sys, shutil, tifffile, numpy as np, pandas as pd
from datetime import datetime
import scipy, matplotlib.pyplot as plt, re
import h5py, pickle
import matplotlib
matplotlib.use('TkAgg')
        matplotlib.use('TkAgg')
        %matplotlib inline
vralign = r"D:\MixedMouse_trial_2\E201_02_May_2023_vr_dlc_align.p"
with open(vralign, "rb") as fp: #unpickle
    vralign = pickle.load(fp)
vralign.keys()
def get_area(x, y):
    #len1 = np.sqrt((x[3]-x[1])*(x[3]-x[1])+(y[3]-y[1])*(y[3]-y[1]))
    #len2 = np.sqrt((x[2]-x[1])*(x[2]-x[1])+(y[2]-y[1])*(y[2]-y[1]))
    #len3 = np.sqrt((x[2]-x[3])*(x[2]-x[3])+(y[2]-y[3])*(y[2]-y[3]))
    #s = ((len1+len2+len3)/2)
    #area = np.sqrt(s*(s-len1)*(s-len2)*(s-len3))
    area = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2]
                  * (y[0] - y[1]))
    return area
theshold = 1e-2
vralign['TongueTop_x'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
vralign['TongueTip_x'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
vralign['TongueBottom_x'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
vralign['TongueTop_y'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
vralign['TongueTip_y'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
vralign['TongueBottom_y'][vralign['TongueTip_likelihood'].astype('float32')<theshold]=np.nan
area = []
for i in range(len(vralign['TongueTop_x'])):
    tonguetopx,tonguetopy, = vralign['TongueTop_x'][i], vralign['TongueTop_y'][i]
    tonguebottomx, tonguebottomy = vralign['TongueBottom_x'][i], vralign['TongueBottom_y'][i]    
    tonguetipx,tonguetipy = vralign['TongueTip_x'][i], vralign['TongueTip_y'][i]
    x = [tonguetipx, tonguebottomx, tonguetopx]
    y = [tonguetipy, tonguebottomy, tonguetopy]
    area.append(get_area(x, y))
tongue_y = np.nanmean(np.array([vralign['TongueTip_y'],vralign['TongueTop_y'],
        vralign['TongueBottom_y']]).astype('float32'), axis=0)
fig, axs = plt.subplots()
#axs.plot(tongue_y[0:4000])
#axs.plot(area[2000:4000])
axs.plot(vralign['TongueTip_y'][0:4000], label = 'TongueTip_y')
#axs.plot(vralign['TongueTop_y'][0:4000], label = 'TongueTop_y')
#axs.plot(vralign['TongueBottom_y'][0:4000], label = 'TongueBottom_y')
axs.plot((vralign['rewards']==0.5)[0:4000]*200)
axs.legend()
reward = np.hstack(vralign['rewards'])
axs.plot(vralign['rewards'])
plt.figure(); plt.plot(area[10000:12000])
axs.plot(reward[10000:120000])
