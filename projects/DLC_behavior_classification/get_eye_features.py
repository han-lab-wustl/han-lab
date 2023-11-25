# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas as pd, sys
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter 
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom to your clone
# sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
mpl.use('TkAgg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
if __name__ == "__main__":
    src = r"I:\pupil_pickles"
    sessions = os.listdir(src)    
    normall_s = [] # get multiple sessions
    normlickmean_s = []; lickmean_s = []
    normvelmean_s = []; velmean_s = []
    circumferences_s = []
    for session in sessions:
        print(session)
        pdst = os.path.join(src, session)
        areas, circumferences, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = eye.get_area_circumference_from_vralign(pdst, 3/2, 10)
        normall_s.append(normrewall_t)
        normlickmean_s.append(normmeanlicks_t)
        lickmean_s.append(meanlicks)
        normvelmean_s.append(normmeanlicks_t)
        velmean_s.append(meanvel)
        circumferences_s.append(circumferences)
    datadct= {}
    datadct['perirew_norm_all_sessions'] = normall_s
    datadct['perilickmean_norm_all_sessions'] = normlickmean_s
    datadct['perilickmean_all_sessions'] = lickmean_s
    datadct['perivelmean_norm_all_sessions'] = normvelmean_s
    datadct['perivelmean_all_sessions'] = velmean_s
    datadct['circumferences_all_sessions'] = circumferences
    with open(r"I:\pupil_circumference_data.p", "wb") as fp:   #Pickling
        pickle.dump(datadct, fp)
    with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
    # plot perireward pupil
    range_val = 10 #s
    binsize = 0.1 #s
        
    fig, axes = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax = axes[0]
    normall = normall_s[session]
    for normall in normall_s:
        # for rew in normall:
        # plot individual trials in grey
        ax.plot(np.nanmean(normall,axis=0), color='slategray', alpha=0.2)
    meanplot = []
    for mean in normall_s:
        meanplot.append(np.nanmean(mean,axis=0))
    meanplot = np.nanmean(np.array(meanplot),axis=0)
    ax.plot(meanplot,color='k')
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_ylabel('Normalized Pupil \n Circumference')
    ax.set_xticks([])
    ax = axes[1]
    for lick in lickmean_s:
        # lick = lickmean_s[session]
        ax.plot(lick, color='r', linewidth=0.5)
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_ylabel('Mean Lick \n Rate')
    ax.set_xticks([])
    ax = axes[2]
    for vel in velmean_s:
        # vel = velmean_s[session]
        ax.plot(vel, color='dimgrey', linewidth=0.5)
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_ylabel('Mean \n Velocity \n (cm/s)')
    ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
    ax.set_xticklabels(np.arange(-range_val,range_val+1))
    ax.set_xlabel('Time From Reward (s)')
    fig.suptitle('n = 10 sessions, 3 animals')
    plt.savefig(r"C:\Users\workstation2\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\perirew_pupil.svg", \
                bbox_inches='tight',transparent=True)
# #%%
        
# # run peri reward LOCATION????????
# range_val = 50 #cm
# binsize = 1 #cm
# normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = eye.perireward_binned_activity(np.array(circumferences), \
#     (vralign['rewards']==0.5).astype(int), 
#                         vralign['ybinned'], range_val, binsize)
# fig, axes = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 1, 1]})
# ax = axes[0]
# for normall in normrewdFF:
#     for rew in normall:
#         # plot individual trials in grey
#         ax.plot(rew, color='slategray', alpha=0.2)
#     ax.plot(np.nanmean(normall,axis=0),color='k')
# ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
# ax.set_ylabel('Normalized Pupil Circumference')
# ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
# ax.set_xticklabels(np.arange(-range_val,range_val+1))
# ax.set_xlabel('Distance From Reward (cm)')
    # # mpl.use('TkAgg')
    # %matplotlib inline
    # fig,axes = plt.subplots(3,1)    
    # rng = np.arange(0,vralign['ybinned'].shape[0]) # restrict number of frames
    # ypos =vralign['ybinned'][rng]*(3/2) 
    # cf = np.array(circumferences)[rng]
    # # filter out low vals for display
    # cf[cf<130]=np.nan
    # normcf = (cf-np.nanmin(cf))/(np.nanmax(cf)-np.nanmin(cf))
    # toplot = gaussian_filter(normcf,
    #                          sigma=5)
    # ax = axes[0]
    # ax.plot(toplot, color='k')
    # ax.set_ylabel('Normalized \n Pupil \n Circumference')
    # ax.set_xticks(np.arange(0,len(rng),2000),\
    #         [np.ceil(xx[0]/60).astype(int) for xx in \
    #          vralign['timedFF'][rng][::2000]], \
    #             fontsize=8)
    # ax = axes[1]
    # ax.plot(vralign['forwardvel'][rng], color='dimgrey')
    # ax.set_ylabel('Velocity (cm/s)')
    # ax.set_xticks(np.arange(0,len(rng),2000),\
    #         [np.ceil(xx[0]/60).astype(int) \
    #          for xx in vralign['timedFF'][rng][::2000]], \
    #             fontsize=8)
    # ax = axes[2]
    # ax.scatter(np.arange(0, len(ypos)), ypos, color='slategray', \
    #             s=0.5)
    
    # ax.scatter(np.argwhere(vralign['licks'][rng]>0).T[0], \
    #             ypos[vralign['licks'][rng]>0], color='r', marker='.',
    #             s=3, alpha=0.5)
    # ax.scatter(np.argwhere(vralign['rewards'][rng]==0.5).T[0], \
    #             ypos[vralign['rewards'][rng]==0.5], color='b', marker='.')
    # ax.set_ylim([0,270])
    # ax.set_yticks(np.arange(0,270+1,90),np.arange(0,270+1,90))
    # ax.set_xticks(np.arange(0,len(rng),2000),\
    #            [np.ceil(xx[0]/60).astype(int) for xx in vralign['timedFF'][rng][::2000]], \
    #             fontsize=8)
    # ax.set_xlabel('Time in Session (minutes)')
    # ax.set_ylabel('Position (cm)')
    # plt.savefig(r"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\pupil_in_session.svg", \
    #                 bbox_inches='tight', transparent=True)
