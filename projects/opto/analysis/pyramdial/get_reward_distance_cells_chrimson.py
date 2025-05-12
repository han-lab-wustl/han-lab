"""get reward distance cells between opto and non opto conditions
feb 2025
vip chrimson excitation
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10 
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_behavior_chrimson_onlyz14.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_chrimson_rewardcells.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_vipexcitation.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
cm_window = 20
# iterate through all animals
# for ii in range(7):
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if True:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
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
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        # only test opto vs. ctrl
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: 
                eptest = random.randint(2,3)   
                if len(eps)<4: eptest = 2 # if no 3 epochs    

        tcs_early = []; tcs_late = []        
        ypos_rel = []; tcs_early = []; tcs_late = []; coms = []
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins        
        
        if f'{animal}_{day:03d}' in radian_alignment_saved.keys():
                k = [xx for xx in radian_alignment_saved.keys() if f'{animal}_{day:03d}' in xx][0]
                print(k)
                tcs_correct, coms_correct, tcs_fail, coms_fail,\
                        com_goal,goal_cell_shuf_ps_av = radian_alignment_saved[k]            
        else:# remake tuning curves relative to reward        
        # takes time
                fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
                Fc3 = fall_fc3['Fc3']
                dFF = fall_fc3['dFF']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
                dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
                skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
                # if animal!='z14' and animal!='e200' and animal!='e189':                
                Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
                if Fc3.shape[1]>0:                        
                        # 9/19/24
                        # find correct trials within each epoch!!!!
                        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                        rewards,forwardvel,rewsize,bin_size)         
                else: # if no skewed cells
                        Fc3 = fall_fc3['Fc3']                        
                        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
                        Fc3 = Fc3[:, skew>.5]
                        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                        rewards,forwardvel,rewsize,bin_size)         
        # only get opto vs. ctrl epoch comparisons
        tcs_correct = tcs_correct[[eptest-2, eptest-1]] 
        coms_correct = coms_correct[[eptest-2, eptest-1]] 
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = [(eptest-2, eptest-1)]    
        # print(eptest, perm)
        com_remap = np.array((coms_rewrel[1]-coms_rewrel[0]))        
        com_goal = [ii for ii,comr in enumerate(com_remap) if (comr<goal_window) & (comr>-goal_window) ]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = com_goal
        # get per comparison
        goal_cell_iind.append(goal_cells)
        goal_cell_p=len(goal_cells)/len(coms_correct[0])
        epoch_perm.append(perm)
        goal_cell_prop.append(goal_cell_p)
        # do not plot cell profiles
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        if len(goal_cells)>0:
            rows = int(np.ceil(np.sqrt(len(goal_cells))))
            cols = int(np.ceil(len(goal_cells) / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
            if len(goal_cells) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(goal_cells):            
                for ep in range(len(coms_correct)):
                    ax = axes[i]
                    ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    # if len(tcs_fail)>0:
                    #         ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                    ax.axvline((bins/2), color='k')
                    ax.set_title(f'cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,bins+1,20))
            ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
            ax.set_xlabel('Radian position (centered start rew loc)')
            ax.set_ylabel('Fc3')
            fig.suptitle(f'{animal}, day {day}')
            fig.tight_layout()
            # plt.show()
            pdf.savefig(fig)
            plt.close(fig)

        # get shuffled iterations
        num_iterations = 10000; shuffled_dist = np.zeros((num_iterations))
        # max of 5 epochs = 10 perms
        goal_cell_shuf_ps = []
        for i in range(num_iterations):
                # shuffle locations
                rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
                shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
                [random.shuffle(shuf) for shuf in shufs]
                # first com is as ep 1, others are shuffled cell identities
                com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
                com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
                # OR shuffle cell identities
                # relative to reward
                coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
                perm = [(eptest-2, eptest-1)]    
                com_remap = np.array((coms_rewrel[1]-coms_rewrel[0]))        
                com_goal_shuf = [ii for ii,comr in enumerate(com_remap) if (comr<goal_window) & (comr>-goal_window)]
                goal_cells_shuf = com_goal_shuf
                shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
                goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
                goal_cell_shuf_ps.append(goal_cell_shuf_p)
        # save median of goal cell shuffle
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
        goal_cell_null.append(goal_cell_shuf_ps_av)
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value); 
        print(f'\n\n {animal}, day {day}: significant goal cells proportion p-value: {p_value}\n\
                total cells: {len(coms_correct[0])}')
        total_cells.append(len(coms_correct[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                        com_goal,goal_cell_shuf_ps_av]
pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
df = conddf.copy()
df['goal_cell_prop'] = goal_cell_prop
df['goal_cell_prop']=df['goal_cell_prop']*100
df['opto'] = df.optoep.values>1
df['opto'] = ['stim' if xx==True else 'no_stim' for xx in df.opto.values]
df['condition'] = ['vip' if xx=='vip_ex' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = goal_cell_null
df['goal_cell_prop_shuffle']=df['goal_cell_prop_shuffle']*100
# remove 0 goal cell prop
df = df[df.goal_cell_prop>0]
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df, x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df['p_value'].values<0.05)/len(df)
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
df = df[(df.animals!='e200')&(df.animals!='e189')]
# exclude outliere?
df = df[(df.days!=7)]
df_plt = df
df_an = df_plt.groupby(['animals','condition','opto']).mean(numeric_only=True)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_plt,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        s=9, dodge=True,alpha=.5)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='opto',data=df_an,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        s=s, dodge=True)
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,
        palette={'no_stim': "slategray", 'stim': "darkgoldenrod"},
        fill=False,ax=ax, errorbar='se')
sns.barplot(x='condition', y='goal_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
# animal lines
df_an = df_an.reset_index()
ans = df_an.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=[-.2,0.2], y='goal_cell_prop', 
    data=df_an[df_an.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['VIP\nExcitation', 'Control'],rotation=45)
ax.set_ylabel('Reward cell %')
df_plt = df_plt.reset_index()
rewprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='stim')), 'goal_cell_prop']
shufprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='no_stim')), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# per animal stats
rewprop = df_an.loc[((df_an.condition=='vip')&(df_an.opto=='stim')), 'goal_cell_prop']
shufprop = df_an.loc[((df_an.condition=='vip')&(df_an.opto=='no_stim')), 'goal_cell_prop']
t,pval = scipy.stats.ttest_rel(rewprop, shufprop)

# statistical annotation    
fs=46
ii=0
y=50
pshift=10
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'nonopto vs. opto chrimson\np={pval:.3g}',fontsize=12,rotation=45)

ii=1
# control vs. chrimson
rewprop = df_plt.loc[((df_plt.condition=='vip')&(df_plt.opto=='stim')), 'goal_cell_prop']
shufprop = df_plt.loc[((df_plt.condition=='ctrl')&(df_plt.opto=='stim')), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# statistical annotation    
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'ctrl vs. chrimson\np={pval:.3g}',fontsize=12,rotation=45)
fig.suptitle('n=session')
# plt.savefig(os.path.join(savedst, 'reward_cell_prop_ctrlvopto_chrimson.svg'),bbox_inches='tight')

#%%
# subtract by led off sessions
