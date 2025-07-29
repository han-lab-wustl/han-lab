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
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime
from projects.DLC_behavior_classification import eye

#%%
# Definitions and setups
mice = ["z14",'z15','z17']
dys_s = [[17,28,29,33,34,36],[6,9,11],[18,19]]
# dys_s = [[43,44,45,48]]
opto_ep_s = [[2,2,2,2,2,2],[2,2,2],[2,2]]
# cells_to_plot_s = [[466,159,423,200,299]]#, [16,6,9], 
cells_to_plot_s = [[0,5,9,3,7,1],[[66,55],[136,165],[93]],[2,83]]

src = r"Y:\analysis\fmats"
dffs_cp_dys = []
mind = 0
# Define the number of bins and the size of each bin
nbins = 90
bin_size = 3                
binsize = 0.3 # s
range_val = 8
# Processing loop
for m, mouse_name in enumerate(mice):
    days = dys_s[m]
    cells_to_plot = cells_to_plot_s[m]
    opto_ep = opto_ep_s[m]
    dyind = 0
    for dy in days:
        daypath = os.path.join(src, mouse_name,'days',
        f"{mouse_name}_day{dy:03d}_plane0_Fall.mat")
        data = loadmat(daypath, variable_names=['dFF', 'forwardvel', 'ybinned', 'VR',
        'timedFF', 'changeRewLoc', 'rewards', 'licks', 'trialnum'])
        dFF = data['dFF']
        changeRewLoc = data['changeRewLoc'].flatten()
        VR = data['VR']
        ybinned = data['ybinned'].flatten()
        forwardvel = data['forwardvel'].flatten()
        time = data['timedFF'].flatten()
        rewards = np.hstack(data['rewards'])==.5
        rewards[:2000]=0 # remove first outliers?
        licks = data['licks'][0]
        trialnum = data['trialnum'].flatten()
        print(daypath)

        # Additional processing
        eps = np.where(changeRewLoc > 0)[0]
        eps = np.append(eps, len(changeRewLoc))
        gainf = 1 / VR['scalingFACTOR'].item()
        rewlocs = np.hstack(changeRewLoc[changeRewLoc > 0] * gainf)
        rewsize = VR['settings'][0][0][0][0][4] * gainf # reward zone is 5th element
        ypos = np.hstack(ybinned * gainf)
        velocity = forwardvel
        dffs_cp = []
        indtemp = 0

        rngopto = range(eps[opto_ep[dyind] - 1], eps[opto_ep[dyind]])
        rngpreopto = range(eps[opto_ep[dyind] - 2], eps[opto_ep[dyind] - 1])
        # normalize to rew loc?
        yposopto = ypos[rngopto]
        ypospreopto = ypos[rngpreopto]
        # mask for activity before reward loc
        # yposoptomask = np.hstack(yposopto < rewloc[opto_ep[dyind] - 1] - rewsize-10)
        # ypospreoptomask = np.hstack(ypospreopto < rewloc[opto_ep[dyind] - 2] - rewsize-10)
        # yposoptomask = np.hstack(yposopto > rewloc[opto_ep[dyind] - 1] - rewsize-10)
        # ypospreoptomask = np.hstack(ypospreopto > rewloc[opto_ep[dyind] - 2] - rewsize-10)
        # get entire tuning curve
        yposoptomask = (np.ones_like(np.hstack(yposopto))*True).astype(bool)
        ypospreoptomask = (np.ones_like(np.hstack(ypospreopto))*True).astype(bool)
        trialoptomask = trialnum[rngopto] > 10
        trialpreoptomask = trialnum[rngpreopto] > 10
        cp = cells_to_plot[dyind] # just get 1 cell        
        dffopto = dFF[rngopto, :]
        dffpreopto = dFF[rngpreopto, :]
        if isinstance(cp, list): 
            cp=cp[0]
        dffs_cp.append([np.nanmean(dffopto[:, [cp]],axis=1), 
                        np.nanmean(dffpreopto[:, [cp]],axis=1)])

        # Extract dFF arrays for the corresponding conditions
        optodff = np.nanmean(dffopto[:, [cp]],axis=1)
        prevepdff = np.nanmean(dffpreopto[:, [cp]],axis=1)
        # tuning curve
        bin_size_dt=3.5
        bins_dt=150
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
                rewsize[0],ybinned,time,licks,
                dFF[:,[cp]],trialnum, rewards,forwardvel,1/gainf[0][0],bin_size_dt,
                bins=bins_dt,lasttr=8)  

        # peri reward activity
        normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
        rewdFF_opto = eye.perireward_binned_activity(optodff, rewards[rngopto], 
                time[rngopto], range_val, binsize)
        normmeanrewdFF, meanrewdFF_ctrl, normrewdFF, \
        rewdFF_ctrl = eye.perireward_binned_activity(prevepdff, rewards[rngpreopto], 
                time[rngpreopto], range_val, binsize)
        #also get vel
        _, meanvel_ctrl, __, \
        velall_ctrl = eye.perireward_binned_activity(forwardvel, rewards[rngpreopto], 
                time[rngpreopto], range_val, binsize)

        dffs_cp_dys.append([meanrewdFF_opto, meanrewdFF_ctrl,rewdFF_opto,rewdFF_ctrl, velall_ctrl,tcs_correct, coms_correct, tcs_fail, coms_fail])
        
        indtemp += 1
        dyind += 1
#%%
# per trial activity during ctrl?
trial_ctrl = [xx[1].T for xx in dffs_cp_dys[:6]]
velall_ctrl = [np.nanmean(xx[4],axis=1).T for xx in dffs_cp_dys[:6]]
fig,axes=plt.subplots(nrows=2,figsize=(6,3),sharex=True)
ax=axes[0]
im=ax.imshow(trial_ctrl,aspect='auto',cmap='magma')
ax.axvline(range_val/binsize,color='w',linestyle='--',linewidth=3)
ax.set_ylabel('Days')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.set_xlabel('Time from reward (s)')
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
fig.colorbar(im, ax=ax, label='VIP $\Delta F/F$')
ax=axes[1]
im=ax.imshow(velall_ctrl,aspect='auto',cmap='Greys')
ax.axvline(range_val/binsize,color='k',linestyle='--',linewidth=3)
ax.set_ylabel('')
ax.set_xlabel('Time from CS (s)')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
ax.set_xticks([0, int(range_val/binsize), (int(range_val/binsize)*2)])
ax.set_xticklabels([-8, 0, 8])
# ax.set_xticklabels(np.arange(-range_val, range_val+1, 1))
fig.colorbar(im, ax=ax, label='Velocity (cm/s)')
axes[0].set_title('VIP+ neuron activity')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'

plt.savefig(os.path.join(savedst,'vip_ctrl_activity.svg'),bbox_inches='tight')
#%%
# tuning curve per day
# per trial activity during ctrl?
trial_ctrl = np.vstack([xx[5][0] for xx in dffs_cp_dys[:6]])
fig,ax=plt.subplots(figsize=(8,4))
im=ax.imshow(trial_ctrl,aspect='auto',cmap='magma')
ax.axvline(75,color='w',linestyle='--',linewidth=3)
ax.set_ylabel('Days')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.set_xlabel('Time from reward (s)')
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
fig.colorbar(im, ax=ax, label='VIP $\Delta F/F$')
ax=axes[1]
im=ax.imshow(velall_ctrl,aspect='auto',cmap='Greys')
ax.axvline(range_val/binsize,color='k',linestyle='--',linewidth=3)
ax.set_ylabel('')
ax.set_xlabel('Time from CS (s)')
ax.set_yticks([0, len(trial_ctrl)-1])
# ax.plot(np.nanmean(trial_ctrl,axis=0))
# axes[1].imshow(trial_ctrl[2].T,aspect='auto')
# axes[2].imshow(trial_ctrl[10].T,aspect='auto')
ax.set_xticks([0, int(range_val/binsize), (int(range_val/binsize)*2)])
ax.set_xticklabels([-8, 0, 8])
# ax.set_xticklabels(np.arange(-range_val, range_val+1, 1))
fig.colorbar(im, ax=ax, label='Velocity (cm/s)')
axes[0].set_title('VIP+ neuron activity')
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'

plt.savefig(os.path.join(savedst,'vip_ctrl_activity.svg'),bbox_inches='tight')
#%%
# from gerardo's data
src = r'Z:\vip_gcamp_data_reformatted'
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
    bin_size_dt=7
    bins_dt=75
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ypos,time,lick,
            dff.T,trialnum, rew,velocity,1,bin_size_dt,
            bins=bins_dt,lasttr=8)  
    vel_tcs_correct, vel_coms_correct, vel_tcs_fail, vel_coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ypos,time,lick,
            np.array([velocity]).T,trialnum, rew,velocity,1,bin_size_dt,
            bins=bins_dt,lasttr=8)  

    def normalize_rows(mat):
        mat = np.array(mat)
        row_min = np.nanmin(mat, axis=1, keepdims=True)
        row_max = np.nanmax(mat, axis=1, keepdims=True)
        return (mat - row_min) / (row_max - row_min + 1e-9)
    tcs_correct=np.array([normalize_rows(xx) for xx in tcs_correct])
    fig,axes=plt.subplots(nrows=2,figsize=(5,6),height_ratios=[3,1],sharex=True)
    ax=axes[0]
    im=ax.imshow(tcs_correct[0][np.argsort(coms_correct[0])],aspect='auto',cmap='magma')
    ax.axvline(bins_dt/2,color='w',linestyle='--',linewidth=3)
    ax.set_ylabel('VIP neurons')
    ax.set_yticks([0, tcs_correct.shape[1]-1])
    ax.set_xticks([0, tcs_correct.shape[2]/2,tcs_correct.shape[2]-1])
    ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
    # fig.colorbar(im, ax=ax, label='VIP $\Delta F/F$',pad=.1)
    ax.set_title(f'Day {dy}')
    ax=axes[1]
    ax.plot(vel_tcs_correct[0][0],color='grey')
    ax.axvline(bins_dt/2,color='k',linestyle='--',linewidth=3)
    ax.set_ylabel('Velocity (cm/s)')
    plt.tight_layout()
    ax.set_xlabel('Reward-centric distance ($\Theta$)')

