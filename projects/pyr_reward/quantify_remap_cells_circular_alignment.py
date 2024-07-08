
"""
zahra
june 2024
visualize reward-relative cells across days
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations
from itertools import chain
import matplotlib.backends.backend_pdf
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_relative_across_days.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
radian_alignment = {}
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if animal!='e217':
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
        VR = fall['VR'][0][0][()]
        stat = np.array([fall['stat'][0][ii][()] for ii in range(fall['stat'][0].shape[0])])
        # filter out skewed cellss
        skew = np.array([xx[0][0] for xx in np.squeeze(np.hstack(stat['skew']))])
        scalingf = VR['scalingFACTOR'][0][0]
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf
        ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = [] # get radian coordinates
        # same as giocomo preprint - worked with gerardo
        for i in range(len(eps)-1):
            y = ybinned[eps[i]:eps[i+1]]
            rew = rewlocs[i]-rewsize/2
            # convert to radians and align to rew
            rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
        rad = np.concatenate(rad)
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum, rewards)
        rates_all.append(success/total_trials)
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_late, coms = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # takes time
            fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3'])
            Fc3 = fall_fc3['Fc3']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            skew_mask = skew_filter>2
            Fc3 = Fc3[:, skew_mask] # only keep cells with skew greateer than 2
            rates, tcs_late, coms = make_tuning_curves_radians(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)
            tcs_late = np.array(tcs_late); coms = np.array(coms)            
        goal_window = 30*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms])
        perm = list(combinations(range(len(coms)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms[0])
        goal_cell_prop.append(goal_cell_p)
        num_epochs.append(len(coms))
        colors = ['navy', 'red', 'green', 'k','darkorange']
        for gc in goal_cells:
            fig, ax = plt.subplots()
            for ep in range(len(coms)):
                ax.plot(tcs_late[ep][gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins/2), color='k')
                ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                ax.set_xticks(np.arange(0,bins+1,10))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),3))
                ax.set_xlabel('Radian position (centered at start of rew loc)')
                ax.set_ylabel('Fc3')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iterations
        num_iterations = 1000
        shuffled_dist = np.zeros((num_iterations))
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms[ii].shape[0])) for ii in range(1, len(coms))]
            [random.shuffle(shuf) for shuf in shufs]
            com_shufs = np.zeros_like(coms)
            com_shufs[0,:] = coms[0]
            com_shufs[1:1+len(shufs),:] = [coms[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms)), 2))     
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf = intersect_arrays(*com_goal)
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms[0])
        
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value)
        print(p_value)
        total_cells.append(len(coms[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_late, coms]

pdf.close()

# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# goal cells across epochs
df = conddf.copy()
df = df[df.animals!='e217']
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df[df.opto==False],
        s=8)
ax.spines[['top','right']].set_visible(False)
#%%

df['recorded_neurons_per_session'] = total_cells
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df[(df.opto==False)&(df.p_value<0.05)],
        s=150, ax=ax)
ax.spines[['top','right']].set_visible(False)

#%%
# split into pre and post reward cells
pre_rew = [[cellind for cellind in range(xx[:,goal_cell_iind[kk]].shape[1]) if np.nanmedian(xx[:,goal_cell_iind[kk]][:,cellind])<0] for kk,xx in enumerate(dist_to_rew)]
post_rew = [[cellind for cellind in range(xx[:,goal_cell_iind[kk]].shape[1]) if np.nanmedian(xx[:,goal_cell_iind[kk]][:,cellind])>0] for kk,xx in enumerate(dist_to_rew)]
pre_rew_prop = [len(xx)/total_cells[ii] for ii,xx in enumerate(pre_rew)]
post_rew_prop = [len(xx)/total_cells[ii] for ii,xx in enumerate(post_rew)]

df['pre_rew_prop'] = pre_rew_prop
df['post_rew_prop'] = post_rew_prop
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='pre_rew_prop',
        hue='animals',data=df[(df.opto==False)&(df.p_value<0.05)],
        s=8)
ax.spines[['top','right']].set_visible(False)
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.stripplot(x='num_epochs', y='post_rew_prop',
        hue='animals',data=df[(df.opto==False)&(df.p_value<0.05)],
        s=8)
ax.spines[['top','right']].set_visible(False)

#%%
df['success_rate'] = rates_all

an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    sns.scatterplot(x='success_rate', y='goal_cell_prop',
            data=df[(df.animals==an)&(df.opto==False)&(df.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()
