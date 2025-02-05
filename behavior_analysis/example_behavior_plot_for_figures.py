"""
quantify licks and velocity during consolidation task
aug 2024
TODO: get first lick during probes
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib
from projects.memory.behavior import consecutive_stretch, get_behavior_tuning_curve, get_success_failure_trials, get_lick_selectivity, \
    get_lick_selectivity_post_reward, calculate_lick_rate
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes

plt.close('all')
# save to pdf
src = r"Z:\chr2_grabda"
animals = ['e232']
# animals = ['e232']
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024"
# all days to quantify for stim @ reward memory analysis
days_all = [[35]]

for ii,animal in enumerate(animals):
    days = days_all[ii]
    for day in days: 
        path=list(Path(os.path.join(src, animal, str(day))).rglob('params.mat'))[0]
        params = scipy.io.loadmat(path, variable_names='VR')
        print(path)
        VR = params['VR'][0][0]
        # dtype=[('name_date_vr', 'O'), ('ROE', 'O'), ('lickThreshold', 'O'), ('reward', 'O'), 
        # ('time', 'O'), ('lick', 'O'), ('ypos', 'O'), 
        #          ('lickVoltage', 'O'), ('trialNum', 'O'), ('timeROE', 'O'), ('changeRewLoc', 'O'), ('pressedKeys', 'O'), ('world', 'O'), 
        #          ('imageSync', 'O'), ('scalingFACTOR', 'O'), ('wOff', 'O'),
        #          ('catchTrial', 'O'), ('optoTrigger', 'O'), ('settings', 'O')]) 
        velocity = VR[1][0]
        lick = VR[5][0]
        time = VR[4][0]
        gainf = VR[14][0][0]
        try:
            rewsize = VR[18][0][0][4][0][0]/gainf
        except:
            rewsize = 20
        velocity=-0.013*velocity[1:]/np.diff(time) # make same size
        velocity = np.append(velocity, np.interp(len(velocity)+1, np.arange(len(velocity)),velocity))
        velocitydf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(velocitydf.rolling(10).mean().values)
        rewards = VR[3][0]
        ypos = VR[6][0]/gainf
        trialnum = VR[8][0]
        changerewloc = VR[10][0]
        rewlocs = changerewloc[changerewloc>0]/gainf
        eps = np.where(changerewloc)[0]
        
        # example plot
        # # if before==True:
        #     rewloc = changerewloc[0]
        #     import matplotlib.patches as patches
        #     fig, ax = plt.subplots()
        #     ax.plot(ypos[probe])
        #     ax.scatter(np.where(lick[probe])[0], ypos[np.where(lick[probe])[0]], 
        #     color='k',s=80)
        #     ax.add_patch(
        #     patches.Rectangle(
        #         xy=(0,rewloc-10),  # point of origin.
        #         width=len(ypos[probe]), height=20, linewidth=1, # width is s
        #         color='slategray', alpha=0.3))
        #     ax.set_ylim([0,270])
        #     ax.spines[['top','right']].set_visible(False)
        #     ax.set_title(f'{day}')
        #     plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior_probes.svg'),bbox_inches='tight')

        
        # example plot during learning
        rew = (rewards==1).astype(int)
        # some trials
        
        import matplotlib.patches as patches
        fig, ax = plt.subplots(figsize=(8,4))
        eptoplot = [1,2]
        jitter1 = 10000
        jitter = 7000        

        mask = np.arange((eps[eptoplot[0]]+jitter1),(eps[eptoplot[1]]+jitter))
        ax.plot(ypos[mask],linewidth=2,zorder=1)
        ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k',
            s=20)
        ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan',
            s=20)
        eprng = np.arange(eps[eptoplot[0]]+jitter1,eps[eptoplot[0]+1])   
        # 1 ep     
        ep1width = np.arange((eps[eptoplot[0]]+jitter1),(eps[eptoplot[1]]))
        ax.add_patch(
        patches.Rectangle(
            xy=(0,rewlocs[eptoplot[0]]-rewsize/2),  # point of origin.
            width=len(ep1width), height=rewsize, linewidth=1, # width is s
            color='slategray', alpha=0.3))
        
        ep2width = np.arange((eps[eptoplot[1]]),(eps[eptoplot[1]]+jitter))
        ax.add_patch(
        patches.Rectangle(
            xy=(len(ep1width),rewlocs[eptoplot[1]]-rewsize/2),  # point of origin.
            width=len(ep2width), height=rewsize, linewidth=1, # width is s
            color='slategray', alpha=0.3))

        ax.set_ylim([0,270])
        ax.spines[['top','right']].set_visible(False)
        tic = time[mask][::1000]
        ax.set_xticks(np.arange(0,len(time[mask]),1000))
        ax.set_xticklabels(np.round(tic/60,2), rotation=45)

        ax.set_title(f'{day}')
        plt.savefig(os.path.join(dst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')
