"""get reward distance cells between opto and non opto conditions
oct 2024
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
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper\vip_r21'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
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
# cm_window = [10,20,30,40,50,60,70,80] # cm

# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (conddf.optoep.values[ii]>1):
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
                tcs_correct, coms_correct, tcs_fail, coms_fail, \
                com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[k]            
        else:# remake tuning curves relative to reward        
        # takes time
                fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
                Fc3 = fall_fc3['Fc3']
                dFF = fall_fc3['dFF']
                if 'bordercells' in fall.keys():
                        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                else:
                        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
                        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
                skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
                # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                # skew_mask = skew_filter>2
                Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
                # 9/19/24
                # find correct trials within each epoch!!!!
                tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)         
        # only get opto vs. ctrl epoch comparisons
        tcs_correct = tcs_correct[[eptest-2, eptest-1]] 
        coms_correct = coms_correct[[eptest-2, eptest-1]] 
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = [(eptest-2, eptest-1)]    
        print(eptest, perm)
        com_remap = np.array((coms_rewrel[1]-coms_rewrel[0]))        
        com_goal = [ii for ii,comr in enumerate(com_remap) if (comr<goal_window) & (comr>-goal_window) ]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = com_goal
        # get per comparison
        goal_cell_iind.append(goal_cells);goal_cell_p=len(goal_cells)/len(coms_correct[0])
        epoch_perm.append(perm)
        goal_cell_prop.append(goal_cell_p)
        # do not plot cell profiles
        
        # get shuffled iterations
        num_iterations = 1000; shuffled_dist = np.zeros((num_iterations))
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
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
        total_cells.append(len(coms_correct[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                        com_goal,goal_cell_shuf_ps_av]
pdf.close()

# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=18)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[df.optoep>1]
df['goal_cell_prop'] = goal_cell_prop
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = goal_cell_null
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df, x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df['p_value'].values<0.05)/len(df)
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(2,5))
# av across mice
df = df[(df.animals!='e200')&(df.animals!='e189')]
df_plt = df
df_plt = df_plt.groupby(['animals','condition']).mean(numeric_only=True)
sns.stripplot(x='condition', y='goal_cell_prop',
        hue='condition',data=df_plt,
        palette={'ctrl': "slategray", 'vip': "red"},
        s=s)
sns.barplot(x='condition', y='goal_cell_prop',
        data=df_plt,
        palette={'ctrl': "slategray", 'vip': "red"},
        fill=False,ax=ax, color='k', errorbar='se')
sns.barplot(x='condition', y='goal_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['Control', 'VIP\nInhibition'],rotation=45)
ax.set_ylabel('Reward-centric cell proportion\n(LEDoff vs. LEDon)')
rewprop = df_plt.loc[(df_plt.index.get_level_values('condition')=='vip'), 'goal_cell_prop']
shufprop = df_plt.loc[(df_plt.index.get_level_values('condition')=='ctrl'), 'goal_cell_prop']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
# statistical annotation    
fs=46
ii=0.5; y=.37; pshift=.05
if pval < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=12)

plt.savefig(os.path.join(savedst, 'reward_cell_prop_ctrlvopto.svg'),bbox_inches='tight')

#%%    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perms['condition'] = ['vip' if (xx=='e218') or (xx=='e216') else 'ctrl' for xx in df_perms['animals'].values]
df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = df_permsav.index.get_level_values("epoch_comparison").unique()
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
    shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'condition','num_epochs']).mean(numeric_only=True)
#%%
# quantify reward cells vip vs. opto
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals','condition','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(5,5))
# av across mice
cmap = ['slategray','crimson']
sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='condition',
        data=df_plt2,palette=cmap,
        s=12, dodge=True, alpha=.8)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='condition',
        data=df_plt2,palette=cmap,
        fill=False,ax=ax, errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward cell proportion')
eps = [2,3,4]
y = 0.16
pshift = 0.02
fs=36
for con in ['vip', 'ctrl']:
    for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[((df_plt2.index.get_level_values('num_epochs')==ep) & \
            (df_plt2.index.get_level_values('condition')==con)), 'goal_cell_prop']
        shufprop = df_plt2.loc[((df_plt2.index.get_level_values('num_epochs')==ep)\
            & (df_plt2.index.get_level_values('condition')==con)), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop, shufprop)
        print(f'{con}, {ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'{con},p={pval:.3g}',fontsize=10)
    y+=.02

# plt.savefig(os.path.join(savedst, 'reward_cell_prop_per_an.png'), 
#         bbox_inches='tight')
#%%

df['recorded_neurons_per_session'] = total_cells
df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df_plt_,
        s=150, ax=ax)
sns.regplot(x='recorded_neurons_per_session', y='goal_cell_prop',
        data=df_plt_,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
        df_plt_['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Av. # of neurons per session')
ax.set_ylabel('Reward cell proportion')

plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
        bbox_inches='tight')

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

# #%%
# # #examples
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greateer than 2
bin_size=3 # cm
# get abs dist tuning 
tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)

# # #plot example tuning curve
plt.rc('font', size=30)  
fig,axes = plt.subplots(1,3,figsize=(20,20), sharex = True)
for ep in range(3):
        axes[ep].imshow(tcs_correct_abs[ep,com_goal[0]][np.argsort(coms_correct_abs[0,com_goal[0]])[:60],:]**.3)
        axes[ep].set_title(f'Epoch {ep+1}')
        axes[ep].axvline((rewlocs[ep]-rewsize/2)/bin_size, color='w', linestyle='--', linewidth=4)
        axes[ep].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
        axes[ep].set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
axes[0].set_ylabel('Reward-distance cells')
axes[2].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'abs_dist_tuning_curves_3_ep.svg'), bbox_inches='tight')

# fig,axes = plt.subplots(1,4,figsize=(15,20), sharey=True, sharex = True)
# axes[0].imshow(tcs_correct[0,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[0].set_title('Epoch 1')
# im = axes[1].imshow(tcs_correct[1,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[1].set_title('Epoch 2')
# im = axes[2].imshow(tcs_correct[2,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[2].set_title('Epoch 3')
# im = axes[3].imshow(tcs_correct[3,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[3].set_title('Epoch 4')
# ax = axes[1]
# ax.set_xticks(np.arange(0,bins+1,10))
# ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
# ax.axvline((bins/2), color='w', linestyle='--')
# axes[0].axvline((bins/2), color='w', linestyle='--')
# axes[2].axvline((bins/2), color='w', linestyle='--')
# axes[3].axvline((bins/2), color='w', linestyle='--')
# axes[0].set_ylabel('Reward distance cells')
# axes[3].set_xlabel('Reward-relative distance (rad)')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'tuning_curves_4_ep.png'), bbox_inches='tight')
# for gc in goal_cells:
#%%
gc = 51
plt.rc('font', size=24)  
fig2,ax2 = plt.subplots(figsize=(5,5))

for ep in range(3):        
        ax2.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
ax2.set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
ax2.set_xlabel('Absolute position (cm)')
ax2.set_ylabel('$\Delta$ F/F')
        
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_tuning_per_ep.svg'), bbox_inches='tight')

fig2,ax2 = plt.subplots(figsize=(5,5))
for ep in range(3):        
        ax2.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(bins/2, color="k", linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,bins+1,30))
ax2.set_xticklabels(np.round(np.arange(-np.pi, 
        np.pi+np.pi/1.5, np.pi/1.5),1))
ax2.set_xlabel('Radian position ($\Theta$)')
ax2.set_ylabel('$\Delta$ F/F')
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_aligned_tuning_per_ep.svg'), bbox_inches='tight')