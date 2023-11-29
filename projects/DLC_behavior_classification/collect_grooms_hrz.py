# -*- coding: utf-8 -*-
"""
@author: zahra
"""
import os, sys, pickle, pandas as pd, numpy as np, scipy
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
#analyze videos and copy vr files before this step
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from math import ceil 
import datetime
import quantify_grooms_hrz
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable 
vrdir =  r'Y:\DLC\VR_data\dlc' # copy of vr data, curated to remove badly labeled files
dlcfls = r'Y:\DLC\dlc_mixedmodel2'#\for_analysis'

with open(os.path.join(dlcfls,'mouse_df.p'),'rb') as fp: #unpickle
                mouse_df = pickle.load(fp) 

# TODO: comparison with hrz and random reward 
hrz_summary = False
groom_binary_dct = {}; trials_s = {}; yposgrs_dct_s = {};  yposgrs_dct_f = {}
yposgrs_dct_p = {}; len_grooms_dct = {}; starts_dct = {}; stops_dct = {}
for i,row in mouse_df.iterrows():
        try:
                groom, starts, stops, \
                yposgrs_s, yposgrs_f, yposgrs_p, tr_s, tr_f, \
                len_grooms = quantify_grooms_hrz.get_long_grooms_per_ep(dlcfls, \
                                row,hrz_summary = hrz_summary)
                nm = row['VR']
                starts_dct[nm] = starts
                stops_dct[nm] = stops
                groom_binary_dct[nm] = groom
                trials_s[nm] = (tr_s,tr_f)
                yposgrs_dct_s[nm] = yposgrs_s
                yposgrs_dct_f[nm] = yposgrs_f
                yposgrs_dct_p[nm] = yposgrs_p 
                len_grooms_dct[nm] = len_grooms
        except Exception as e:
                print(e)
        if i%20==0: print(f'{i}/{len(mouse_df)}')   # counter

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

# or import saved data
savepth = r"Y:\DLC\dlc_mixedmodel2\grooming\grooming_data.p"
with open(savepth, "rb") as fp: #unpickle
        datadct = pickle.load(fp)
starts_dct = datadct['starts']
ans_groom = datadct['animals_groom']
rel_ypos_groom_s = datadct['ypos_rel_reward_successfultrials']
rel_ypos_groom_f = datadct['ypos_rel_reward_failedtrials']
len_grooms_seconds = datadct['length_groom_seconds']
an_session = datadct['session_per_animal']
an_nms = [xx[:4].capitalize() for xx in ans_groom]
an_nms_unique = np.unique(np.array(an_nms))
an_session = Counter(an_nms)

################################## fig 1 ##################################
# position relative to reward quant
%matplotlib inline
import math
rel_ypos_groom_s = np.array(list(yposgrs_dct_s.values()))[~np.isnan(ans_binary)][ans_binary_]
rel_ypos_groom_s = np.hstack(rel_ypos_groom_s)
rel_ypos_groom_f = np.array(list(yposgrs_dct_f.values()))[~np.isnan(ans_binary)][ans_binary_]
rel_ypos_groom_f = np.hstack(rel_ypos_groom_f)
# rel_ypos_groom_p = np.array(list(yposgrs_dct_p.values()))[~np.isnan(ans_binary)][ans_binary_]
# rel_ypos_groom_p = np.hstack(rel_ypos_groom_p)
fig, ax = plt.subplots()                                        
ax.hist(rel_ypos_groom_s, bins = 20, label = 'successful trials',
        color = 'slategrey', edgecolor='black')
ax.hist(rel_ypos_groom_f, bins = 20, label = 'failed trials', color = 'lightgray', 
        alpha=0.5, edgecolor='black')
# ax.hist(rel_ypos_groom_p, bins = 20, label = 'probe trials (rel. to prev. rew zone)')
# ax.set_xlim([-1, 1])
ax.set_title(f'n = {len(an_nms)} sessions, {len(an_nms_unique)} animals')
ax.set_ylabel('Number of long grooming bouts')
ax.set_xlabel('Distance relative to reward (cm)')
ax.legend()
# count number of sessions per animal
print(Counter(an_nms))
plt.savefig(r'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\yppos_grooming_relative_to_rew.svg',
        bbox_inches = 'tight', transparent = True)
################################## fig 2 ##################################
# average time of long grooms
len_grooms_ = np.array(list(len_grooms_dct.values()))[~np.isnan(ans_binary)][ans_binary_]
len_grooms_ = np.hstack(len_grooms_)
len_grooms_seconds = np.round(len_grooms_/31.25,2)
fig, ax = plt.subplots()                                        
ax.hist(len_grooms_seconds, bins = 30, color = 'slategrey', edgecolor='black')
ax.set_ylabel('Frequency')
ax.set_xlabel('Duration of grooming bout (s)')
plt.savefig(r'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\grooming_duration.svg',
        bbox_inches = 'tight', transparent = True)

datadct = {}
datadct['animals_groom'] = ans_groom
datadct['ypos_rel_reward_successfultrials'] = rel_ypos_groom_s
datadct['ypos_rel_reward_failedtrials'] = rel_ypos_groom_f
datadct['length_groom_seconds'] = np.round(len_grooms_/31.25,2)
datadct['session_per_animal'] = an_session
datadct['starts'] = starts_dct
datadct['stops'] = stops_dct
savepth = r"Y:\DLC\dlc_mixedmodel2\grooming\grooming_data.p"
with open(savepth, "wb") as fp:   #Pickling
        pickle.dump(datadct, fp)
################################## fig 2 ##################################

# start trigger relative to tongue
src = r'Y:\DLC\dlc_mixedmodel2'
paws = []; tongues = []
for an in ans_groom:
        if 'T10_' in an or 'T11_' in an:
                pth = an[:-18]+'_vr_dlc_align.p'
        else:
                pth = an[:-18]+'vr_dlc_align.p'
        print(pth)
        with open(os.path.join(src,pth),'rb') as fp: #unpickle
                vralign = pickle.load(fp) 
        starts = starts_dct[an]
        #filter
        threshold = 0.99
        vralign['PawTop_x'][vralign['PawTop_likelihood'].astype('float32') < threshold] = 0
        vralign['PawTop_y'][vralign['PawTop_likelihood'].astype('float32') < threshold] = 0
        vralign['PawMiddle_x'][vralign['PawMiddle_likelihood'].astype('float32') < threshold] = 0
        vralign['PawMiddle_y'][vralign['PawMiddle_likelihood'].astype('float32') < threshold] = 0
        vralign['PawBottom_x'][vralign['PawBottom_likelihood'].astype('float32') < threshold] = 0
        vralign['PawBottom_y'][vralign['PawBottom_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueTop_x'][vralign['TongueTop_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueTop_y'][vralign['TongueTop_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueTip_x'][vralign['TongueTip_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueTip_y'][vralign['TongueTip_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueBottom_x'][vralign['TongueBottom_likelihood'].astype('float32') < threshold] = 0
        vralign['TongueBottom_y'][vralign['TongueBottom_likelihood'].astype('float32') < threshold] = 0

        paw_y = np.nanmean(np.array([vralign['PawTop_y'],
                vralign['PawBottom_y'],vralign['PawMiddle_y']]),axis=0)
        paw = paw_y>0
        tongue=(np.nanmean(np.array([vralign['TongueTip_y'],
                vralign['TongueTop_y'],
                vralign['TongueBottom_y']]),axis=0)).astype(int)
        range_val = 10; binsize=0.1
        normmeanstartpaw, meanstartpaw, normstartpaw, startpaw = perireward_binned_activity(paw_y, starts, vralign['timedFF'],
                range_val, binsize, rewind = True)
        normmeanstarttongue, meanstarttongue, normstarttongue, starttongue = perireward_binned_activity(tongue, starts, vralign['timedFF'],
                range_val, binsize, rewind = True)
        paws.append(startpaw)
        tongues.append(starttongue)

rng = (range_val/binsize)*2
med = np.median(np.arange(0,rng+1)).astype(int)+4
plotp = []; plott = []
for i,p in enumerate(paws):        
        trs = p.shape[1]
        for tr in range(trs):
                if sum(p[:med-1,tr])>0:
                        print("bad start")
                else:
                        plotp.append(p[:,tr])
                        plott.append(tongues[i][:,tr])

fig, axes = plt.subplots(2,1)
ax = axes[0]
im = ax.imshow(np.array(plott)>0, cmap = 'Reds')
ax.axvline(med, color='k', linestyle='--')
divider = make_axes_locatable(ax) 
cax = divider.append_axes('right', size='3%', pad=0.1) 
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_xticks([])
ax.set_ylabel('Trials')
ax.set_title('Tongue')

ax = axes[1]
im = ax.imshow(np.array(plotp)>0)
divider = make_axes_locatable(ax) 
cax = divider.append_axes('right', size='3%', pad=0.1) 
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_ylabel('Trials')
ax.axvline(med, color='white', linestyle='--')
ax.set_title('Paw')
ax.set_ylabel('Trials')
ax.set_xlabel('Time from grooming start (s)')
ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,20))
ax.set_xticklabels(np.arange(-range_val,range_val+1,2))
plt.savefig(r'C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\start_trig_grooms_with_tongue.svg',
        bbox_inches = 'tight', transparent = True)
