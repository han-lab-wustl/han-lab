
"""
zahra
july 2024
quantify reward-relative cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_by_trialtype, intersect_arrays
from rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\retreat_2024'
savepth = os.path.join(savedst, 'true_pc.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
coms_all = []

# cm_window = [10,20,30,40,50,60,70,80] # cm
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if ((animal!='e217') & (animal!='e200')) & (conddf.optoep.values[ii]==-1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
        pcs = np.vstack(np.array(fall['putative_pcs'][0]))
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc))        
        lasttr=8 # last trials
        bins=90
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1 epoch
        pc_bool = np.sum(pcs,axis=0)>=len(eps)-2        
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        bin_size=3 # cm
        # get abs dist tuning 
        tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
            Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'steelblue']
        coms_all.append(coms_correct_abs)
        for gc in range(tcs_correct_abs.shape[1]):
            fig, ax = plt.subplots()
            for ep in range(len(coms_correct_abs)):
                ax.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
                ax.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,10))
                ax.set_xticklabels(np.arange(0,track_length+bin_size*10,bin_size*10).astype(int))
                ax.set_xlabel('Absolute position (cm)')
                ax.set_ylabel('Fc3')
                ax.spines[['top','right']].set_visible(False)
            ax.legend()
        #     plt.savefig(os.path.join(savedst, 'true_place_cell.png'), bbox_inches='tight', dpi=500)
            plt.close('all')
            pdf.savefig(fig)
        
pdf.close()
#%%
# #plot example tuning curve
plt.rc('font', size=14)  
fig,axes = plt.subplots(1,4,figsize=(10,10), sharey=True, sharex = True)
axes[0].imshow(tcs_correct_abs[0][np.argsort(coms_correct_abs[0]),:]**.6)
axes[0].set_title('Epoch 1')
im = axes[1].imshow(tcs_correct_abs[1][np.argsort(coms_correct_abs[0]),:]**.6)
axes[1].set_title('Epoch 2')
im = axes[2].imshow(tcs_correct_abs[2][np.argsort(coms_correct_abs[0]),:]**.6)
axes[2].set_title('Epoch 3')
im = axes[3].imshow(tcs_correct_abs[3][np.argsort(coms_correct_abs[0]),:]**.6)
axes[3].set_title('Epoch 4')
ax = axes[1]
axes[0].axvline((rewlocs[0]-rewsize/2)/bin_size, color='w', linestyle='--')
axes[1].axvline((rewlocs[1]-rewsize/2)/bin_size, color='w', linestyle='--')
axes[2].axvline((rewlocs[2]-rewsize/2)/bin_size, color='w', linestyle='--')
axes[3].axvline((rewlocs[3]-rewsize/2)/bin_size, color='w', linestyle='--')
ax = axes[3]
ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,15))
ax.set_xticklabels(np.arange(0,track_length+bin_size*15,bin_size*15).astype(int),rotation=45)

ax=axes[0]
ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,15))
ax.set_xticklabels(np.arange(0,track_length+bin_size*15,bin_size*15).astype(int),rotation=45)
ax=axes[1]
ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,15))
ax.set_xticklabels(np.arange(0,track_length+bin_size*15,bin_size*15).astype(int),rotation=45)
ax=axes[2]
ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,15))
ax.set_xticklabels(np.arange(0,track_length+bin_size*15,bin_size*15).astype(int),rotation=45)

fig.tight_layout()
axes[0].set_ylabel('All cells')
axes[3].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'place_cell_only_tuning_curves_4_ep.png'), bbox_inches='tight')