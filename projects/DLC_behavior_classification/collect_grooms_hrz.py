# -*- coding: utf-8 -*-
"""
@author: zahra
"""
import os, sys, pickle, pandas as pd, numpy as np, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
#analyze videos and copy vr files before this step
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
from math import ceil 
import datetime
import quantify_grooms_hrz
from collections import Counter

vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
dlcfls = r'Y:\DLC\dlc_mixedmodel2'#\for_analysis'

with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp) 

hrz_summary = False
groom_binary_dct = {}; counts_dct = {}; yposgrs_dct_s = {};  yposgrs_dct_f = {}
for i,row in mouse_df.iterrows():
    groom, starts, stops, counts_s, counts_f, yposgrs_s, yposgrs_f = quantify_grooms_hrz.get_long_grooms_per_ep(dlcfls,row,hrz_summary = True)
    nm = row['VR']
    groom_binary_dct[nm] = groom
    counts_dct[nm] = counts_s
    yposgrs_dct_s[nm] = yposgrs_s
    yposgrs_dct_f[nm] = yposgrs_f
    print(nm)

# make figs of compiled data
ans_binary = np.array(list(groom_binary_dct.values()))
ans_binary_ = ans_binary[~np.isnan(ans_binary)].astype(bool)
ans = np.array(list(groom_binary_dct.keys()))[~np.isnan(ans_binary)]

# based on this, most animals groom within a session
ans_groom = ans[ans_binary_]
ans_no_groom = ans[~ans_binary_]

an_nms = [xx[:4].capitalize() for xx in ans_groom]
an_nms_unique = np.unique(np.array(an_nms))
an_session = Counter(an_nms)

# categories quant
# counts_groom = np.array(list(counts_dct.values()))[~np.isnan(ans_binary)][ans_binary_]
# cat = ['dark time', 'before rew', 'after rew', 'rew zone']
# counts_groom = np.array([np.array(xx) for xx in counts_groom])
# bar = np.sum(counts_groom,0)
# fig, ax = plt.subplots()                                        
# ax.bar(cat, bar)
# ax.set_title(f'n = {len(an_nms)} sessions, {len(an_nms_unique)} animals')
# ax.set_ylabel('number of grooming bouts')

# positive relative to reward quant
rel_ypos_groom_s = np.array(list(yposgrs_dct_s.values()))[~np.isnan(ans_binary)][ans_binary_]
rel_ypos_groom_s = np.hstack(rel_ypos_groom_s)
rel_ypos_groom_f = np.array(list(yposgrs_dct_f.values()))[~np.isnan(ans_binary)][ans_binary_]
rel_ypos_groom_f = np.hstack(rel_ypos_groom_f)
fig, ax = plt.subplots()                                        
ax.hist(rel_ypos_groom_s, bins = 20, label = 'successful trials')
ax.hist(rel_ypos_groom_f, bins = 20, label = 'failed trials')
# ax.set_xlim([-1, 1])
ax.set_title(f'n = {len(an_nms)} sessions, {len(an_nms_unique)} animals')
ax.set_ylabel('number of long grooming bouts')
ax.set_xlabel('distance relative to reward (cm)')
ax.legend()
# count number of sessions per animal
print(Counter(an_nms))

