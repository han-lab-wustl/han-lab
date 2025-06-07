"""get place cells between opto and non opto conditions
april 2025
"""

#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from statsmodels.formula.api import ols
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import itertools
from statsmodels.stats.anova import anova_lm  # <-- Correct import
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper\vip_r21'
savepth = os.path.join(savedst, 'vip_opto_place.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
# with open(saveddataset, "rb") as fp: #unpickle
#         radian_alignment_saved = pickle.load(fp)
# initialize var
datadct = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
place_cell_null=[]
place_window = 20
num_iterations=1000
bin_size=3 # cm
lasttr=8 # last trials
bins=90
# cm_window = [10,20,30,40,50,60,70,80] # cm
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if True:#(conddf.optoep.values[ii]>1):
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
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        # only test opto vs. ctrl
        eptest = conddf.optoep.values[ii]
        if conddf.optoep.values[ii]<2: 
                eptest = random.randint(2,3)   
                if len(eps)<4: eptest = 2 # if no 3 epochs    

        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1
        pc_bool = np.sum(pcs,axis=0)>=len(eps)-2
        # looser restrictions
        pc_bool = np.sum(pcs,axis=0)>=1
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        # if no cells pass these crit
        if Fc3.shape[1]==0:
                Fc3 = fall_fc3['Fc3']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                # to avoid issues with e217?
                # pc_bool = np.sum(pcs,axis=0)>=1
                Fc3 = Fc3[:,((skew>2))]
        if Fc3.shape[1]>0:
                # get abs dist tuning 
                tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
                        Fc3,trialnum,rewards,forwardvel,
                        rewsize,bin_size) # last 5 trials
                # get cells that maintain their coms b/wn previous and opto ep
                perm = [(eptest-2, eptest-1)]   
                if perm[0][1]<len(coms_correct_abs): # make sure tested epoch has enough trials
                        print(eptest, perm)            
                        com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
                        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
                        # get cells across OPTO VS. CONTROL EPOCHS
                        pcs = np.unique(np.concatenate(compc))
                        pcs_all = pcs
                        # get per comparison
                        pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
                        pc_ind.append(pcs_all);pc_p=len(pcs_all)/len(coms_correct_abs[0])
                        epoch_perm.append(perm)
                        pc_prop.append([pcs_p_per_comparison,pc_p])
                        num_epochs.append(len(coms_correct_abs))

                        # get shuffled iterations
                        shuffled_dist = np.zeros((num_iterations))
                        # max of 5 epochs = 10 perms
                        place_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
                        place_cell_shuf_ps = []
                        for i in range(num_iterations):
                                # shuffle locations
                                shufs = [list(range(coms_correct_abs[ii].shape[0])) for ii in range(1, len(coms_correct_abs))]
                                [random.shuffle(shuf) for shuf in shufs]
                                # first com is as ep 1, others are shuffled cell identities
                                com_shufs = np.zeros_like(coms_correct_abs); com_shufs[0,:] = coms_correct_abs[0]
                                com_shufs[1:1+len(shufs),:] = [coms_correct_abs[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
                                # get cells that maintain their coms across at least 2 epochs
                                perm = list(combinations(range(len(com_shufs)), 2))     
                                perm = [(eptest-2, eptest-1)]    
                                com_per_ep = np.array([(com_shufs[perm[jj][0]]-com_shufs[perm[jj][1]]) for jj in range(len(perm))])        
                                compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
                                # get cells across all epochs that meet crit
                                pcs = np.unique(np.concatenate(compc))
                                pcs_all = pcs#intersect_arrays(*compc)
                                # get per comparison
                                pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
                                shuffled_dist[i] = len(pcs_all)/len(coms_correct_abs[0])
                                place_cell_shuf_p=len(pcs_all)/len(com_shufs[0])
                                place_cell_shuf_ps.append(place_cell_shuf_p)
                                place_cell_shuf_ps_per_comp[i, :len(pcs_p_per_comparison)] = pcs_p_per_comparison
                        # save median of goal cell shuffle
                        place_cell_shuf_ps_per_comp_av = np.nanmedian(place_cell_shuf_ps_per_comp,axis=0)        
                        place_cell_shuf_ps_av = np.nanmedian(np.array(place_cell_shuf_ps)[1])
                        place_cell_null.append([place_cell_shuf_ps_per_comp_av,place_cell_shuf_ps_av])
                        p_value = sum(shuffled_dist>pc_p)/num_iterations
                        print(f'{animal}, day {day}: significant place cells proportion p-value: {p_value}')
                        pvals.append(p_value);     
                        total_cells.append(len(coms_correct_abs[0]))
                        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs, 
                                pcs_all, place_cell_shuf_ps_per_comp_av, place_cell_shuf_ps_av]
pdf.close()

#%%
plt.rc('font', size=18)          # controls default text sizes
# plot goal cells across epochs
# just opto days
s=12
df = conddf.copy()
inds = [int(xx[-3:]) for xx in datadct.keys()]
df = df[(df.index.isin(inds))]
df['place_cell_prop'] = [xx[1] for xx in pc_prop]
df['place_cell_prop']=df['place_cell_prop']*100
df['opto'] = df.optoep.values>1
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['place_cell_prop_shuffle'] =  [xx[1] for xx in place_cell_null]
df['place_cell_prop_shuffle']=df['place_cell_prop_shuffle']*100
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
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}

df = df[(df.animals!='e189')]
df_plt = df
df_plt = df_plt.groupby(['animals','condition','opto']).mean(numeric_only=True).reset_index()
sns.stripplot(x='opto', y='place_cell_prop',
        hue='condition',data=df_plt,
        palette=pl,dodge=True,
        s=s)
sns.barplot(x='opto', y='place_cell_prop',hue='condition',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se')
sns.barplot(x='opto', y='place_cell_prop_shuffle',
        data=df_plt,ax=ax, color='dimgrey',label='shuffle',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('')
# ax.set_xticks([0,1,2], labels=['Control', 'VIP\nInhibition', 'VIP\nExcitation'],rotation=20)
ax.set_ylabel('Place cell %\n(LEDon)')

model = ols('place_cell_prop ~ C(condition)', data=df_plt).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

plt.savefig(os.path.join(savedst, 'place_cell_prop_ctrlvopto.svg'),bbox_inches='tight')
# %%

#%%
# subtract by led off sessions
# ----------------------------------------
# Plotting Stim - No Stim per Animal
# ----------------------------------------
df_an = df_plt
fig2, ax2 = plt.subplots(figsize=(3, 5))
df_diff = (
    df_an[df_an.opto ==True]
    .set_index(['animals', 'condition'])[['place_cell_prop']]
    .rename(columns={'place_cell_prop': 'stim'})
)
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}

df_diff['no_stim'] = df_an[df_an.opto == False].set_index(['animals', 'condition'])['place_cell_prop']
df_diff['delta'] = df_diff['stim']-df_diff['no_stim']
df_diff = df_diff.reset_index()
df_diff = df_diff[(df_diff.animals!='e190')&(df_diff.animals!='e189')&(df_diff.animals!='e200')]
# Plot
a=0.7
sns.stripplot(data=df_diff, x='condition', y='delta', ax=ax2, 
             palette=pl, size=s,alpha=a)
sns.barplot(data=df_diff, x='condition', y='delta', ax=ax2, 
             palette=pl, fill=False)

# Aesthetics
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel('Δ Place cell % (LEDon-LEDoff)')
ax2.set_xlabel('')

ax2.set_xticklabels(['Control', 'VIP\nInhibition','VIP\nExcitation'], rotation=20)
ax2.set_title('Per-animal difference\n\n')
ax2.spines[['top', 'right']].set_visible(False)
rewprop = df_diff.loc[((df_diff.condition=='vip')), 'delta']
shufprop = df_diff.loc[((df_diff.condition=='ctrl')), 'delta']
t,pval = scipy.stats.ranksums(rewprop, shufprop)
rewprop = df_diff.loc[((df_diff.condition=='vip_ex')), 'delta']
shufprop = df_diff.loc[((df_diff.condition=='ctrl')), 'delta']
t,pval2 = scipy.stats.ranksums(rewprop, shufprop)

# statistical annotation    
y = 12
ii=0
if pval < 0.001:
        ax2.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        ax2.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        ax2.text(ii, y, "*", ha='center', fontsize=fs)
ax2.text(ii, y, f'{pval:.2g}', ha='center', fontsize=12)

plt.savefig(os.path.join(savedst, 'place_cell_prop_difference_all.svg'), bbox_inches='tight')

