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
saveddataset = r"Z:\saved_datasets\place_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        datadct = pickle.load(fp)
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

# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if ii!=179:
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
            # to avoid issues with e217 and z17?
            # pc_bool = np.sum(pcs,axis=0)>=1
            Fc3 = Fc3[:,((skew>1.5)&pc_bool)]
        if Fc3.shape[1]>0:
            # get abs dist tuning 
            if sum([f'{animal}_{day:03d}' in xx for xx in list(datadct.keys())])>0:
                k = [k for k,xx in datadct.items() if f'{animal}_{day:03d}' in k][0]
                tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs=datadct[k]
            else:
                print('#############making tcs#############\n')
                tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
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
                datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs]
pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(datadct, fp) 

#%%
# top down approach
# 1) com dist in opto vs. control
# 3) place v. reward
# tcs_correct, coms_correct, tcs_fail, coms_fail,
# tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early
# 1) get coms correct
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
coms_correct = [xx[1] for k,xx in datadct.items()]
optoep = [xx if xx>1 else 2 for xx in df.optoep.values]
# opto comparison
coms_correct = [xx[[optoep[ep]-2,optoep[ep]-1],:] if len(xx)>optoep[ep]-1 else xx[[optoep[ep]-3,optoep[ep]-2]]for ep,xx in enumerate(coms_correct)]
# tcs_correct = [xx[[optoep[ep]-2,optoep[ep]-1],:] for ep,xx in enumerate(tcs_correct)]
coms_correct_prev = [xx[0] for ep,xx in enumerate(coms_correct)]
coms_correct_opto = [xx[1] for ep,xx in enumerate(coms_correct)]
# tcs_correct_prev = [xx[0] for ep,xx in enumerate(tcs_correct)]
# tcs_correct_opto = [xx[1] for ep,xx in enumerate(tcs_correct)]

vip_in_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]>1))]
vip_in_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]>1))]
vip_in_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]==-1))]
vip_in_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip') and (df.optoep.values[kk]==-1))]
# excitation
vip_ex_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]>1))]
vip_ex_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]>1))]
vip_ex_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]<1))]
vip_ex_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if ((df.in_type.values[kk]=='vip_ex') and (df.optoep.values[kk]<1))]
#control
ctrl_com_prev = [xx for kk,xx in enumerate(coms_correct_prev) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]>1))]
ctrl_com_opto = [xx for kk,xx in enumerate(coms_correct_opto) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]>1))]
ctrl_com_ctrl_prev = [xx for kk,xx in enumerate(coms_correct_prev) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]<1))]
ctrl_com_ctrl_opto = [xx for kk,xx in enumerate(coms_correct_opto) if (('vip' not in df.in_type.values[kk]) and (df.optoep.values[kk]<1))]
#%%
plots = [[ctrl_com_prev,vip_in_com_prev,vip_ex_com_prev,ctrl_com_ctrl_prev,vip_in_com_ctrl_prev,vip_ex_com_ctrl_prev],
        [ctrl_com_opto,vip_in_com_opto,vip_ex_com_opto,ctrl_com_ctrl_opto,vip_in_com_ctrl_opto,vip_ex_com_ctrl_opto]]
lbls=['ctrl_ledon','vip_in_ledon','vip_ex_ledon','ctrl_ledoff','vip_in_ledoff','vip_ex_ledoff']
a=0.4
fig,axes=plt.subplots(ncols=3,nrows=2,figsize=(17,10))
axes=axes.flatten()
for pl in range(len(plots[0])):
    ax=axes[pl]
    # Concatenate and subtract pi
    data_prev = np.concatenate(plots[0][pl]) - np.pi
    data_opto = np.concatenate(plots[1][pl]) - np.pi
    ax.hist(data_prev,alpha=a,label='prev_ep',bins=100,density=True)
    ax.hist(data_opto,alpha=a,label='opto_ep',bins=100,density=True)
    # KDE plots
    sns.kdeplot(data_prev, ax=ax, label='prev_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    sns.kdeplot(data_opto, ax=ax, label='opto_ep', fill=True, alpha=.1, linewidth=1.5,legend=False)
    ax.set_title(lbls[pl])
#     ax.set_xlim([-np.pi/6,np.pi])
    ax.axvline(0, color='gray', linewidth=2,linestyle='--')
ax.legend()

#%%

plt.rc('font', size=20)          # controls default text sizes
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
df=df[df.place_cell_prop>0]
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
fig,ax = plt.subplots(figsize=(5,5))
# av across mice
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}

df=df[(df.animals!='e189')&(df.animals!='e190')]
# remove outlier days
df=df[~((df.animals=='z14')&(df.days<15))]
df=df[~((df.animals=='z15')&(df.days<8))]
df=df[~((df.animals=='e217')&(df.days<9))]
df=df[~((df.animals=='e216')&(df.days<32))]
# realdf=realdf[~((realdf.animal=='e218')&(realdf.day>44))]
df_plt = df
df_plt = df_plt.groupby(['animals','condition','opto']).mean(numeric_only=True).reset_index()
sns.stripplot(x='opto', y='place_cell_prop',
        hue='condition',data=df_plt,
        palette=pl,dodge=True,
        s=s,alpha=0.7)
sns.barplot(x='opto', y='place_cell_prop',hue='condition',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
sns.barplot(x='opto', y='place_cell_prop_shuffle',hue='condition',
        data=df_plt,ax=ax, color='dimgrey',alpha=0.3,
        err_kws={'color': 'grey'},errorbar=None,legend=False)
ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
handles, labels = ax.get_legend_handles_labels()
labels = [new_labels.get(label, label) for label in labels]
ax.legend(handles, labels, bbox_to_anchor=(.95, 1.0))
ax.set_xlabel('')
ax.set_xticks([0,1], labels=['LEDoff', 'LEDon'])
ax.set_ylabel('Place cell %')

# 2-way ANOVA
model = ols('place_cell_prop ~ C(condition) * C(opto)', data=df_plt).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Post-hoc Tukey HSD
df_plt['group'] = df_plt['condition'] + '_' + str(df_plt['opto'])
tukey = pairwise_tukeyhsd(endog=df_plt['place_cell_prop'],
                          groups=df_plt['group'],
                          alpha=0.05)
print(tukey.summary())
# Add annotations manually
from statannotations.Annotator import Annotator
# Define pairs to compare based on Tukey results
pairs = [
    (('True', 'ctrl'), ('False', 'ctrl')),
    (('True', 'vip'), ('False', 'vip')),
    (('True', 'vip_ex'), ('False', 'vip_ex')),
    (('True', 'ctrl'), ('True', 'vip')),
    (('True', 'ctrl'), ('True', 'vip_ex'))
]
# Format data for Annotator
df_plt['opto'] = df_plt['opto'].astype(str)
annot = Annotator(ax, pairs,data=df_plt, x='opto', y='place_cell_prop',
                  hue='condition', palette=pl, dodge=True)

annot.configure(test=None, text_format='star', loc='outside')
pvalues = []
for (o1, c1), (o2, c2) in pairs:
    group1 = df_plt[(df_plt['opto'] == o1) & (df_plt['condition'] == c1)]['place_cell_prop']
    group2 = df_plt[(df_plt['opto'] == o2) & (df_plt['condition'] == c2)]['place_cell_prop']
    stat, pval = scipy.stats.ttest_ind(group1, group2)
    pvalues.append(pval)

annot.set_pvalues_and_annotate(pvalues)
plt.savefig(os.path.join(savedst, 'place_cell_prop_ctrlvopto.svg'),bbox_inches='tight')
#%%
# shuffle subtracted
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import scipy.stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os

# Shuffle-subtracted value
df = df[(df.animals != 'e189') & (df.animals != 'e190')]
# df = df[~((df.animals == 'z14') & (df.days < 15))]
# df = df[~((df.animals == 'z15') & (df.days < 8))]
# df = df[~((df.animals == 'e217') & (df.days < 9))]
# df = df[~((df.animals == 'e216') & (df.days < 32))]

# Calculate delta = real - shuffle
df['delta_place_cell_prop'] = df['place_cell_prop'] - df['place_cell_prop_shuffle']
# Group by animal to average across days
df_plt = df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True).reset_index()

# Plot
fig, ax = plt.subplots(figsize=(7, 8))
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex': 'darkgoldenrod'}

sns.stripplot(x='opto', y='delta_place_cell_prop', hue='condition',
              data=df_plt, palette=pl, dodge=True, s=s, alpha=0.7)
sns.barplot(x='opto', y='delta_place_cell_prop', hue='condition',
            data=df_plt, palette=pl, fill=False, ax=ax,
            errorbar='se', legend=False)

# Aesthetics
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('')
ax.set_xticks([0, 1])
ax.set_xticklabels(['LED off', 'LED on'])
ax.set_ylabel('Place cell % (shuffle-subtracted)')

# Legend fix
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
handles, labels = ax.get_legend_handles_labels()
labels = [new_labels.get(label, label) for label in labels]
ax.legend(handles, labels, bbox_to_anchor=(.95, 1.0))

# Two-way ANOVA
model = ols('delta_place_cell_prop ~ C(condition) * C(opto)', data=df_plt).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# Statistical annotations
pairs = [
    ((True, 'ctrl'), (False, 'ctrl')),
    ((True, 'vip'), (False, 'vip')),
    ((True, 'vip_ex'), (False, 'vip_ex')),
    ((True, 'ctrl'), (True, 'vip')),
    ((True, 'ctrl'), (True, 'vip_ex'))
]

# Prepare data for annotation
df_plt['opto'] = df_plt['opto'].astype(bool)  # make sure it's boolean
annot = Annotator(ax, pairs, data=df_plt, x='opto', y='delta_place_cell_prop',
                  hue='condition', palette=pl, dodge=True)
annot.configure(test=None, text_format='star', loc='outside')

# Use ranksums
pvalues = []
for (o1, c1), (o2, c2) in pairs:
    g1 = df_plt[(df_plt['opto'] == o1) & (df_plt['condition'] == c1)]['delta_place_cell_prop']
    g2 = df_plt[(df_plt['opto'] == o2) & (df_plt['condition'] == c2)]['delta_place_cell_prop']
    stat, pval = scipy.stats.ranksums(g1, g2)
    pvalues.append(pval)

annot.set_pvalues_and_annotate(pvalues)

# Save
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'place_cell_prop_ctrlvopto_shufflesubtracted.svg'), bbox_inches='tight')

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
ax2.set_title('Place\n\n')
ax2.spines[['top', 'right']].set_visible(False)

model = ols('delta ~ C(condition)', data=df_diff).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
# Pairwise Mann-Whitney U tests (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
p_vals = []
for c1, c2 in comparisons:
    x1 = df_diff[df_diff['condition'] == c1]['delta'].dropna()
    x2 = df_diff[df_diff['condition'] == c2]['delta'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05,height=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + height, y_pos + height, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=40)
    plt.text(x_center, y_pos, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = df_diff['delta'].max() + .01
gap = 2
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i],height=0.7)
    y_start += gap
    
plt.savefig(os.path.join(savedst, 'place_cell_prop_difference_all.svg'), bbox_inches='tight')