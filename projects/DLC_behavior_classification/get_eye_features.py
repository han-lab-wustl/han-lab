# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas as pd, sys, math
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter 
plt.rcParams["font.family"] = "Arial"
# sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom to your clone
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
mpl.use('TkAgg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 6
mpl.rcParams["ytick.major.size"] = 6
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__": # TODO; compare with diameter
    src = r"I:\pupil_pickles"
    add_to_dct = False # add to previous datadct
    if add_to_dct:
        with open(r"Z:\pupil_data.p", "rb") as fp: #unpickle
            datadct = pickle.load(fp)
        sessions_analyzed = datadct['sessions']
        sessions_to_analyze = [xx for xx in os.listdir(src) if xx not in sessions_analyzed]
        c_x = datadct['centroids_x_all_sessions']
        c_y = datadct['centroids_y_all_sessions']
        normall_s = datadct['perirew_norm_all_sessions']
        normlickmean_s = datadct['perilickmean_norm_all_sessions']
        lickmean_s = datadct['perilickmean_all_sessions']
        normvelmean_s= datadct['perivelmean_norm_all_sessions']
        velmean_s = datadct['perivelmean_all_sessions']
        circumferences_s = datadct['circumferences_all_sessions']
        sessions = [sessions_analyzed, sessions_to_analyze]
        lickall_s = datadct['perilick_all_sessions']
        velall_s = datadct['perivel_all_sessions']
    else:    
        normall_s = [] # get multiple sessions
        normlickmean_s = []; lickmean_s = []; lickall_s = []
        normvelmean_s = []; velmean_s = []; velall_s = []
        circumferences_s = []
        c_x = []; c_y = []
        sessions = os.listdir(src)   
        sessions_to_analyze = sessions 
    for session in sessions_to_analyze:
        print(session)
        pdst = os.path.join(src, session)
        areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = eye.get_area_circumference_from_vralign(pdst, 3/2, 10)
        c_x.append(centroids_x)
        c_y.append(centroids_y)
        normall_s.append(normrewall_t)
        normlickmean_s.append(normmeanlicks_t)
        lickmean_s.append(meanlicks)
        lickall_s.append(lickall)
        normvelmean_s.append(normmeanvel_t)
        velmean_s.append(meanvel)
        velall_s.append(velall)
        circumferences_s.append(circumferences)
    datadct= {}
    datadct['sessions'] = list(np.hstack(np.array(sessions)))
    datadct['centroids_x_all_sessions'] = c_x
    datadct['centroids_y_all_sessions'] = c_y
    datadct['perirew_norm_all_sessions'] = normall_s
    datadct['perilickmean_norm_all_sessions'] = normlickmean_s
    datadct['perilickmean_all_sessions'] = lickmean_s
    datadct['perivelmean_norm_all_sessions'] = normvelmean_s
    datadct['perivelmean_all_sessions'] = velmean_s
    datadct['circumferences_all_sessions'] = circumferences_s
    datadct['perilick_all_sessions'] = lickall_s
    datadct['perivel_all_sessions'] = velall_s
    with open(r"I:\pupil_data.p", "wb") as fp:   #Pickling
        pickle.dump(datadct, fp)
#%%
# plot perireward pupil
range_val = 10 #s
binsize = 0.1 #s
##################################### fig 1 #####################################
%matplotlib inline
fig, axes = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 1, 1]})
ax = axes[0]

for normall in normall_s:
    # for rew in normall:
    # plot average of individual trials in grey
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
ax.set_ylim([0,1])
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
ax.set_ylim([0,100])
fig.suptitle('n = 14 sessions, 4 animals')
plt.savefig(r"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\perirew_pupil.svg", \
            bbox_inches='tight',transparent=True)

##################################### fig 2 #####################################
# trial by trial subplot
for i,normall in enumerate(normall_s):
    # for rew in normall:
    # plot average of individual trials in grey
    fig, axes = plt.subplots(3,1)#, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax = axes[0]
    im = ax.imshow(normall)
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_xticks([])
    ax.set_ylabel('Trials')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')        
    ax = axes[1]
    im = ax.imshow(lickall[i].T, cmap='Reds')
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_xticks([])
    ax.set_ylabel('Lick \n Rate')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax = axes[2]
    im = ax.imshow(velall[i].T, cmap='gist_yarg')
    ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
    ax.axvline(np.median(np.arange(0,len(normall.T))+5), color='aqua', linestyle='--')
    ax.set_ylabel('Mean \n Velocity \n (cm/s)')
    ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
    ax.set_xticklabels(np.arange(-range_val,range_val+1))
    ax.set_xlabel('Time From Reward (s)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(sessions[i])
    plt.savefig(rf"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\perirew_pupil_per_trial_{sessions[i][:-15]}.svg", \
            bbox_inches='tight',transparent=True)

##################################### fig 3 #####################################
# failed trials
# get failed trials
rewsize = 10
circ = datadct['circumferences_all_sessions']
pdst = os.path.join(src, sessions_analyzed[6])
with open(pdst, "rb") as fp: #unpickle
    vralign = pickle.load(fp)

eps = list(np.where(vralign['changeRewLoc']>0)[0])
eps.append(len(vralign['changeRewLoc']))
rewlocs = vralign['changeRewLoc'][vralign['changeRewLoc']>0]
for i,ep in enumerate(eps):
    eprng = np.arange(ep,eps[i+1])
    c_x_ep = np.array(c_x_)[eprng]
    rew_ep = np.hstack(vralign['rewards'])[eprng]
    rewloc = rewlocs[i]
    ypos = vralign['ybinned'][eprng]
    successtrialtable = []
    failedtrialtable = []
    trials = vralign['trialnum'][eprng]
    for trial in np.unique(trials)[:-1]:
        c_x_tr = c_x_ep[np.hstack(trials==trial)]
        rew_tr = rew_ep[np.hstack(trials==trial)]
        ypos_tr = ypos[np.hstack(trials==trial)]
        rewloc_tr = (np.hstack(ypos_tr>=rewloc) & np.hstack(ypos_tr<=rewloc+2))
        ypos_ =np.hstack(ypos_tr).astype(int)
        df = pd.DataFrame(np.array([c_x_tr, rewloc_tr, ypos_]).T)
        # takes median and normalizes
        df_ = df.groupby(2).mean()
        c_x_tr_ = np.hstack(df_[0].values)
        rewloc_tr_ = np.hstack(df_[1].values)
        if sum(rew_tr)==1.5:
            successtrialtable.append([c_x_tr_, rewloc_tr_])
        elif sum(rew_tr)==0:
            failedtrialtable.append([c_x_tr_, rewloc_tr_])
    fig, ax = plt.subplots()
    for s in successtrialtable:
        ax.plot(s[0], color='slategray', alpha=0.2)
    y, error = tolerant_mean(np.array([xx[0] for xx in successtrialtable]))
    ax.plot(y, 'k')
    ymin,ymax = ax.get_ylim()
    rect = patches.Rectangle((rewloc-rewsize/2, ymin), rewsize, ymax, 
                linewidth=1, edgecolor='k', 
            facecolor='aqua', alpha=0.3)
    ax.add_patch(rect)
    ax.set_title(f'ep{i+1}, successful trials')
    fig, ax = plt.subplots()
    for s in failedtrialtable:
        ax.plot(s[0], color='slategray', alpha=0.2)        
    y, error = tolerant_mean(np.array([xx[0] for xx in failedtrialtable]))
    ax.plot(y, 'k')
    ymin,ymax = ax.get_ylim()
    rect = patches.Rectangle((rewloc-rewsize/2, ymin), rewsize, ymax, 
                linewidth=1, edgecolor='k', 
            facecolor='aqua', alpha=0.3)
    ax.add_patch(rect)
    ax.set_title(f'ep{i+1}, failed trials')


#     ##################################### fig 3 #####################################    
#     # angle of centroid relative to what?
#     # ax.plot(c_x[0])
#     # ax.plot(c_y[0])
#     # run peri reward time
#     range_val = 10 #s
#     binsize = 0.1 #s
#     pdst = os.path.join(src, sessions_analyzed[10])
#     with open(pdst, "rb") as fp: #unpickle
#         vralign = pickle.load(fp)
#     # angle between 2 points
#     xdist = vralign['EyeLidEast_x']-c_x[10]
#     ydist = vralign['EyeLidSouthWest_y']-c_y[10]
#     rad = [(math.atan(ydist[i]/xx)) for i,xx in enumerate(xdist)]
#     x = c_x[10]
#     # zscore = (c_x[7]-np.nanmean(c_x[7]))/np.std(c_x[7])
#     # zscore = c_y[7]
#     normmeanrew_t, meanrew, normrewall_t, \
#         rewall = eye.perireward_binned_activity(np.array(x), \
#                             (vralign['rewards']==0.5).astype(int), 
#                             vralign['timedFF'], range_val, binsize)
#     normmeanlicks_t, meanlicks, normlickall_t, \
#     lickall = eye.perireward_binned_activity(vralign['licks'], \
#                         (vralign['rewards']==0.5).astype(int), 
#                         vralign['timedFF'], range_val, binsize)
#     normmeanvel_t, meanvel, normvelall_t, \
#     velall = eye.perireward_binned_activity(vralign['forwardvel'], \
#                         (vralign['rewards']==0.5).astype(int), 
#                         vralign['timedFF'], range_val, binsize)
#     # normalize from -1 to 1
#     norm = []
#     for rew in rewall.T:
#         norm.append(2*((rew-np.min(rew)) / (np.max(rew)-np.min(rew)))-1)
#     norm = np.array(norm)   
#     fig, axes = plt.subplots()
#     axes.imshow(norm[:,:]) 
#     fig, axes = plt.subplots(3,1)
#     axes[0].imshow(norm[:,:])
#     axes[1].imshow(normlickall_t[:,:],cmap='Reds')
#     axes[2].imshow(velall.T[:,:],cmap='Greys')
#     x = 13
#     plt.plot(norm[x])
#     plt.plot(normlickall_t[x])
#     plt.plot(normvelall_t[x])
#     for n in norm:
#         ax.plot(norm)

# def tolerant_mean(arrs):
#     lens = [len(i) for i in arrs]
#     arr = np.ma.empty((np.max(lens),len(arrs)))
#     arr.mask = True
#     for idx, l in enumerate(arrs):
#         arr[:len(l),idx] = l
#     return arr.mean(axis = -1), arr.std(axis=-1)

# # get failed trials
# rewsize = 15
# c_x_ = c_x[10]
# pdst = os.path.join(src, sessions_analyzed[10])
# with open(pdst, "rb") as fp: #unpickle
#     vralign = pickle.load(fp)

# eps = list(np.where(vralign['changeRewLoc']>0)[0])
# eps.append(len(vralign['changeRewLoc']))
# rewlocs = vralign['changeRewLoc'][vralign['changeRewLoc']>0]
# for i,ep in enumerate(eps):
#     eprng = np.arange(ep,eps[i+1])
#     c_x_ep = np.array(c_x_)[eprng]
#     rew_ep = np.hstack(vralign['rewards'])[eprng]
#     rewloc = rewlocs[i]
#     ypos = vralign['ybinned'][eprng]
#     successtrialtable = []
#     failedtrialtable = []
#     trials = vralign['trialnum'][eprng]
#     for trial in np.unique(trials)[:-1]:
#         c_x_tr = c_x_ep[np.hstack(trials==trial)]
#         rew_tr = rew_ep[np.hstack(trials==trial)]
#         ypos_tr = ypos[np.hstack(trials==trial)]
#         rewloc_tr = (np.hstack(ypos_tr>=rewloc) & np.hstack(ypos_tr<=rewloc+2))
#         ypos_ =np.hstack(ypos_tr).astype(int)
#         df = pd.DataFrame(np.array([c_x_tr, rewloc_tr, ypos_]).T)
#         # takes median and normalizes
#         df_ = df.groupby(2).mean()
#         c_x_tr_ = np.hstack(df_[0].values)
#         rewloc_tr_ = np.hstack(df_[1].values)
#         if sum(rew_tr)==1.5:
#             successtrialtable.append([c_x_tr_, rewloc_tr_])
#         elif sum(rew_tr)==0:
#             failedtrialtable.append([c_x_tr_, rewloc_tr_])
#     fig, ax = plt.subplots()
#     for s in successtrialtable:
#         ax.plot(s[0], color='slategray', alpha=0.2)
#     y, error = tolerant_mean(np.array([xx[0] for xx in successtrialtable]))
#     ax.plot(y, 'k')
#     ymin,ymax = ax.get_ylim()
#     rect = patches.Rectangle((rewloc-rewsize/2, ymin), rewsize, ymax, 
#                 linewidth=1, edgecolor='k', 
#             facecolor='aqua', alpha=0.3)
#     ax.add_patch(rect)
#     ax.set_title(f'ep{i+1}, successful trials')
#     fig, ax = plt.subplots()
#     for s in failedtrialtable:
#         ax.plot(s[0], color='slategray', alpha=0.2)        
#     y, error = tolerant_mean(np.array([xx[0] for xx in failedtrialtable]))
#     ax.plot(y, 'k')
#     ymin,ymax = ax.get_ylim()
#     rect = patches.Rectangle((rewloc-rewsize/2, ymin), rewsize, ymax, 
#                 linewidth=1, edgecolor='k', 
#             facecolor='aqua', alpha=0.3)
#     ax.add_patch(rect)
#     ax.set_title(f'ep{i+1}, failed trials')
        
# # # run peri reward LOCATION????????
# # range_val = 50 #cm
# # binsize = 1 #cm
# # normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = eye.perireward_binned_activity(np.array(circumferences), \
# #     (vralign['rewards']==0.5).astype(int), 
# #                         vralign['ybinned'], range_val, binsize)
# # fig, axes = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 1, 1]})
# # ax = axes[0]
# # for normall in normrewdFF:
# #     for rew in normall:
# #         # plot individual trials in grey
# #         ax.plot(rew, color='slategray', alpha=0.2)
# #     ax.plot(np.nanmean(normall,axis=0),color='k')
# # ax.axvline(np.median(np.arange(0,len(normall.T))), color='b', linestyle='--')
# # ax.set_ylabel('Normalized Pupil Circumference')
# # ax.set_xticks(np.arange(0, ((range_val)/binsize*2)+1,10))
# # ax.set_xticklabels(np.arange(-range_val,range_val+1))
# # ax.set_xlabel('Distance From Reward (cm)')
#     # # mpl.use('TkAgg')
#     # %matplotlib inline
#     # fig,axes = plt.subplots(3,1)    
#     # rng = np.arange(0,vralign['ybinned'].shape[0]) # restrict number of frames
#     # ypos =vralign['ybinned'][rng]*(3/2) 
#     # cf = np.array(circumferences)[rng]
#     # # filter out low vals for display
#     # cf[cf<130]=np.nan
#     # normcf = (cf-np.nanmin(cf))/(np.nanmax(cf)-np.nanmin(cf))
#     # toplot = gaussian_filter(normcf,
#     #                          sigma=5)
#     # ax = axes[0]
#     # ax.plot(toplot, color='k')
#     # ax.set_ylabel('Normalized \n Pupil \n Circumference')
#     # ax.set_xticks(np.arange(0,len(rng),2000),\
#     #         [np.ceil(xx[0]/60).astype(int) for xx in \
#     #          vralign['timedFF'][rng][::2000]], \
#     #             fontsize=8)
#     # ax = axes[1]
#     # ax.plot(vralign['forwardvel'][rng], color='dimgrey')
#     # ax.set_ylabel('Velocity (cm/s)')
#     # ax.set_xticks(np.arange(0,len(rng),2000),\
#     #         [np.ceil(xx[0]/60).astype(int) \
#     #          for xx in vralign['timedFF'][rng][::2000]], \
#     #             fontsize=8)
#     # ax = axes[2]
#     # ax.scatter(np.arange(0, len(ypos)), ypos, color='slategray', \
#     #             s=0.5)
    
#     # ax.scatter(np.argwhere(vralign['licks'][rng]>0).T[0], \
#     #             ypos[vralign['licks'][rng]>0], color='r', marker='.',
#     #             s=3, alpha=0.5)
#     # ax.scatter(np.argwhere(vralign['rewards'][rng]==0.5).T[0], \
#     #             ypos[vralign['rewards'][rng]==0.5], color='b', marker='.')
#     # ax.set_ylim([0,270])
#     # ax.set_yticks(np.arange(0,270+1,90),np.arange(0,270+1,90))
#     # ax.set_xticks(np.arange(0,len(rng),2000),\
#     #            [np.ceil(xx[0]/60).astype(int) for xx in vralign['timedFF'][rng][::2000]], \
#     #             fontsize=8)
#     # ax.set_xlabel('Time in Session (minutes)')
#     # ax.set_ylabel('Position (cm)')
#     # plt.savefig(r"C:\Users\Han\Box\neuro_phd_stuff\han_2023\dlc\dlc_poster_2023\pupil_in_session.svg", \
#     #                 bbox_inches='tight', transparent=True)

# # %%

# %%
