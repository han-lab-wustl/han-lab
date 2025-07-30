#%%
from scipy.io import loadmat
import os, scipy, sys
import glob, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl, pandas as pd, seaborn as sns
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.DLC_behavior_classification import eye
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%

# from gerardo's data
src = r'Z:\vip_gcamp_data_reformatted'
dst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
from utils.utils import listdir
fls = listdir(src)
dys=[1,2,3,4,5]
for dy in dys:
    mats = [scipy.io.loadmat(fl) for fl in fls if f'day{dy}' in fl]
    changeRewLoc=mats[0]['changerewloc'].flatten()
    # Additional processing
    eps = np.where(changeRewLoc > 0)[0]
    eps = np.append(eps, len(changeRewLoc))
    gainf = 1
    rewlocs = np.hstack(changeRewLoc[changeRewLoc > 0] * gainf)
    rewsize = 20
    dff=mats[1]['plt']
    lick=mats[2]['lick'].flatten()    
    rew=mats[3]['rew'].flatten()
    trialnum=mats[5]['tr'].flatten()
    velocity=mats[6]['vel'].flatten()
    ypos=mats[7]['ybin'].flatten()
    time=mats[4]['time'].flatten()
    dt=np.nanmedian(np.diff(time))
    lick_rate=smooth_lick_rate(lick,dt)
    bin_size_dt=14
    bins_dt=38
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ypos,time,lick,
            dff.T,trialnum, rew,velocity,1,bin_size_dt,
            bins=bins_dt,lasttr=20)  
    bin_size=9
    bins=int(180/bin_size)
    tcs_correct_abs, coms_correct_abs, tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,
        ypos,dff.T,trialnum, rew,velocity,rewsize,bin_size,
        lasttr=20,velocity_filter=True,bins=bins,)  
    vel_tcs_correct, vel_coms_correct, vel_tcs_fail, vel_coms_fail = make_tuning_curves(eps,rewlocs,
        ypos,np.array([velocity]).T,trialnum, rew,velocity,rewsize,bin_size,
        lasttr=20,velocity_filter=True,bins=bins,) 
    lick_tcs_correct, lick_coms_correct, lick_tcs_fail, lick_coms_fail = make_tuning_curves(eps,rewlocs,
        ypos,np.array([lick_rate]).T,trialnum, rew,velocity,rewsize,bin_size,
        lasttr=20,velocity_filter=True,bins=bins,)  

    def normalize_rows(mat):
        mat = np.array(mat)
        row_min = np.nanmin(mat, axis=1, keepdims=True)
        row_max = np.nanmax(mat, axis=1, keepdims=True)
        return (mat - row_min) / (row_max - row_min + 1e-9)
    tcs_correct=np.array([normalize_rows(xx) for xx in tcs_correct])
    fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(8,7),height_ratios=[3,1,1],sharex=True,sharey='row')
    for ep in range(3):
        ax=axes[0,ep]
        im=ax.imshow(tcs_correct[ep][np.argsort(coms_correct[0])],aspect='auto',cmap='magma')
        ax.axvline(bins_dt/2,color='w',linestyle='--',linewidth=3)        
        ax.set_yticks([0, tcs_correct.shape[1]-1])
        ax.set_xticks([0, tcs_correct.shape[2]/2,tcs_correct.shape[2]-1])
        ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
        ax.set_title(f'Epoch {ep+1}')
        ax=axes[1,ep]
        ax.plot(lick_tcs_correct[ep][0],color='k')
        ax.axvline(bins_dt/2,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
        ax=axes[2,ep]
        ax.plot(vel_tcs_correct[ep][0],color='grey')
        ax.axvline(bins_dt/2,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
    # cbar = fig.colorbar(im, ax=axes[0, 2], orientation='vertical', label='VIP $\Delta F/F$', pad=0,fraction=0.5)
    axes[0,0].set_ylabel('VIP neurons')
    fig.suptitle(f'Day {dy}\nCorrect trials')
    axes[2,0].set_ylabel('Velocity (cm/s)')
    axes[1,0].set_ylabel('Lick rate(licks/s)')
    axes[2,0].set_xlabel('Reward-centric distance ($\Theta$)')
    plt.tight_layout()        
    # incorrect trials
    fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(8,7),height_ratios=[3,1,1],sharex=True,sharey='row')
    for ep in range(3):
        ax=axes[0,ep]
        im=ax.imshow(tcs_fail[ep][np.argsort(coms_fail[0])],aspect='auto',cmap='magma')
        ax.axvline(bins_dt/2,color='w',linestyle='--',linewidth=3)        
        ax.set_yticks([0, tcs_correct.shape[1]-1])
        ax.set_xticks([0, tcs_correct.shape[2]/2,tcs_correct.shape[2]-1])
        ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
        ax.set_title(f'Epoch {ep+1}')
        ax=axes[1,ep]
        ax.plot(lick_tcs_fail[ep][0],color='k')
        ax.axvline(bins_dt/2,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
        ax=axes[2,ep]
        ax.plot(vel_tcs_fail[ep][0],color='grey')
        ax.axvline(bins_dt/2,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
    # cbar = fig.colorbar(im, ax=axes[0, 2], orientation='vertical', label='VIP $\Delta F/F$', pad=0,fraction=0.5)
    axes[0,0].set_ylabel('VIP neurons')
    fig.suptitle(f'Day {dy}\nIncorrect trials')
    axes[2,0].set_ylabel('Velocity (cm/s)')
    axes[1,0].set_ylabel('Lick rate(licks/s)')
    axes[2,0].set_xlabel('Reward-centric distance ($\Theta$)')
    plt.tight_layout()  
    #%% 
    # peri reward
    range_val, binsize=8,.2
    rewall=[]
    for dffcll in dff:
        _, meanrewdFF, __, rewdFF = eye.perireward_binned_activity(dffcll, rew==1, time, range_val, binsize)
        rewall.append(meanrewdFF)
    _, meanlick, __, lickall = eye.perireward_binned_activity(lick_rate, rew==1, time, range_val, binsize)
    _, meanvel, __, velall = eye.perireward_binned_activity(velocity, rew==1, time, range_val, binsize)
    rewall=np.array(rewall) 
    #%%   
    def compute_com(tuning_curve):
        x = np.arange(tuning_curve.shape[-1])
        com = np.nansum(tuning_curve * x, axis=-1) / (np.nansum(tuning_curve, axis=-1) + 1e-9)
        return com
    data = normalize_rows(rewall)

    # Run k-means clustering to separate into 3 cell types
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(data)
    # Sort within each cluster by COM
    coms = compute_com(data)
    clustered_indices = []
    for i in range(3):
        cluster_inds = np.where(cluster_labels == i)[0]
        sorted_inds = cluster_inds[np.argsort(coms[cluster_inds])]
        clustered_indices.append(sorted_inds)
    clustered_indices[0], clustered_indices[1] = clustered_indices[1], clustered_indices[0]
    # Combine all sorted indices into a single order
    sorted_all_inds = np.concatenate(clustered_indices)
    # Reorder the data
    data_sorted = data[sorted_all_inds]
    # Plot
    fig, axes = plt.subplots(nrows=3, figsize=(5, 6), height_ratios=[3, 1, 1], sharex=True, sharey='row')
    ax = axes[0]
    im = ax.imshow(data_sorted, aspect='auto', cmap='cividis')
    # Optionally, draw horizontal lines to separate the clusters
    sep1 = len(clustered_indices[0])-.5
    sep2 = sep1 + len(clustered_indices[1])
    # Add cluster labels
    midpoints = [ 
        len(clustered_indices[0]) / 2,
        sep1 + len(clustered_indices[1]) / 2,
        sep2 + len(clustered_indices[2]) / 2
    ]
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    for y, label in zip(midpoints, labels):
        ax.text(-5, y, label, va='center', ha='right', fontsize=20, color='y')
    ax.hlines([sep1, sep2], *ax.get_xlim(), color='k',linewidth=5)
    ax.axvline(range_val/binsize, color='w',linestyle='--',linewidth=3)
    ax.set_title('VIP neuron clusters (K=3)')
    ax.set_yticks([0,int(sep1),int(sep2),data_sorted.shape[0]-1])
    ax.set_yticklabels([1,int(sep1+1),int(sep2+1),data_sorted.shape[0]])
    ax.set_ylabel('VIP neurons (sorted)')
    ax=axes[1]
    ax.plot(meanlick,color='k')
    ax.set_ylabel('Lick rate(licks/s)')
    ax.spines[['top','right']].set_visible(False)
    ax.axvline(range_val/binsize, color='k',linestyle='--',linewidth=3)
    ax=axes[2]
    ax.plot(meanvel,color='grey')
    ax.set_ylabel('Velocity (cm/s)')    
    ax.spines[['top','right']].set_visible(False)
    ax.axvline(range_val/binsize, color='k',linestyle='--',linewidth=3)
    plt.tight_layout()    
    # Move colorbar outside
    divider = make_axes_locatable(axes[0])
    # cax = divider.append_axes("right", size="7%",pad=0.1)  # size & pad can be tuned
    # cbar = fig.colorbar(im, cax=cax,fraction=0.5)
    cbar.set_label('Norm. $\Delta F/F$')    
    ax.set_xticks([0,range_val/binsize,range_val/binsize*2])
    ax.set_xticklabels([-8,0,8])
    ax.set_xlabel('Time from reward (s)')
    plt.savefig(os.path.join(dst, 'vip_clusters.svg'),bbox_inches='tight')
    #%%
    # save tc
    tcs_correct_abs=np.array([normalize_rows(xx) for xx in tcs_correct_abs])
    fig,axes=plt.subplots(nrows=3,ncols=2,figsize=(5.5,7),height_ratios=[4,1,1],sharex=True,sharey='row')
    eps=[1,2]
    for nm,ep in enumerate(eps):
        ax=axes[0,nm]
        im=ax.imshow(tcs_correct_abs[ep][np.argsort(coms_correct_abs[1])],aspect='auto',cmap='cividis')
        ax.axvline((rewlocs[ep]-rewsize/2)/bin_size,color='w',linestyle='--',linewidth=3)        
        ax.set_yticks([0, tcs_correct_abs.shape[1]-1])
        ax.set_yticklabels([1, tcs_correct_abs.shape[1]])
        ax.set_xticks([0, tcs_correct_abs.shape[2]/2,tcs_correct_abs.shape[2]-1])
        ax.set_xticklabels([0,90,180])
        ax.set_title(f'Epoch {nm+1}')
        ax=axes[1,nm]
        ax.plot(lick_tcs_correct[ep][0],color='k')
        ax.axvline((rewlocs[ep]-rewsize/2)/bin_size,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
        ax=axes[2,nm]
        ax.plot(vel_tcs_correct[ep][0],color='grey')
        ax.axvline((rewlocs[ep]-rewsize/2)/bin_size,color='k',linestyle='--',linewidth=3)                
        ax.spines[['top','right']].set_visible(False)
        axes[0,0].set_ylabel('VIP neurons (sorted)')
        fig.suptitle(f'mouse 137,Correct trials')
        axes[2,0].set_ylabel('Velocity (cm/s)')
        axes[1,0].set_ylabel('Lick rate(licks/s)')
        axes[2,0].set_xlabel('Track position (cm)')
    cbar_ax = fig.add_axes([.95, 0.465, 0.03, 0.35])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('VIP $\Delta F/F$')
    plt.tight_layout()  
    plt.savefig(os.path.join(dst, 'vip_tc.svg'),bbox_inches='tight')
      

# %%
