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
from projects.pyr_reward.placecell import make_tuning_curves, make_tuning_curves_early, intersect_arrays, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_by_trialtype_w_darktime_early
from projects.pyr_reward.rewardcell import get_radian_position
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
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
other_sp_prop=[]
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
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF','licks',
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
        time = fall['timedFF'][0]
        lick = fall['licks'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            time=time[:-1]
            lick=lick[:-1]
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
        if animal=='e200' or animal=='e217' or animal=='z17':
            Fc3 = Fc3[:,((skew>1)&pc_bool)]
        else:
            Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greater than 2
        # if no cells pass these crit
        skew_thres_range=np.arange(0,1.6,0.1)[::-1]
        iii=0
        while Fc3.shape[1]==0:      
            iii+=1
            print('************************0 cells skew > 2************************')
            Fc3 = fall_fc3['Fc3']                        
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            Fc3 = Fc3[:, (skew>skew_thres_range[iii])&pc_bool]
        if Fc3.shape[1]>0:
            # get abs dist tuning 
            if sum([f'{animal}_{day:03d}' in xx for xx in list(datadct.keys())])>0:
                k = [k for k,xx in datadct.items() if f'{animal}_{day:03d}' in k][0]
                tcs_correct_abs, coms_correct_abs,tcs_fail_abs,coms_fail_abs, tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early,pcs_all=datadct[k]
            else:
                print('#############making tcs#############\n')
                tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
                Fc3,trialnum,rewards,forwardvel,
                rewsize,bin_size) # last 5 trials
                tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early = make_tuning_curves_early(eps,rewlocs,ybinned, Fc3,trialnum,rewards,forwardvel,
                rewsize,bin_size) # last 5 trials

            track_length_dt = 550 # cm estimate based on 99.9% of ypos
            track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
            bins_dt=150 
            bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
            tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
                bins=bins_dt,lasttr=8) 
            tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt,lasttr=8) 


            # get cells that maintain their coms b/wn previous and opto ep
            perm = [(eptest-2, eptest-1)]   
            if perm[0][1]<len(coms_correct_abs): # make sure tested epoch has enough trials
                print(eptest, perm)            
                goal_window = 20*(2*np.pi/track_length) # cm converted to rad
                coms_rewrel = np.array([com-np.pi for com in coms_correct])
                # account for cells that move to the end/front
                # Define a small window around pi (e.g., epsilon)
                epsilon = .7 # 20 cm
                # Find COMs near pi and shift to -pi
                com_loop_w_in_window = []
                for pi,p in enumerate(perm):
                    for cll in range(coms_rewrel.shape[1]):
                        com1_rel = coms_rewrel[p[0],cll]
                        com2_rel = coms_rewrel[p[1],cll]
                        # print(com1_rel,com2_rel,com_diff)
                        if ((abs(com1_rel - np.pi) < epsilon) and 
                        (abs(com2_rel + np.pi) < epsilon)):
                                com_loop_w_in_window.append(cll)
                # get abs value instead
                coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
                com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
                com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
                com_goal=[xx for xx in com_goal if len(xx)>0]
                if len(com_goal)>0:
                    goal_cells = intersect_arrays(*com_goal)
                else:
                    goal_cells=[]
                # early goal cells
                coms_rewrel = np.array([com-np.pi for com in coms_correct_early])
                # account for cells that move to the end/front
                # Define a small window around pi (e.g., epsilon)
                epsilon = .7 # 20 cm
                # Find COMs near pi and shift to -pi
                com_loop_w_in_window = []
                for pi,p in enumerate(perm):
                    for cll in range(coms_rewrel.shape[1]):
                        com1_rel = coms_rewrel[p[0],cll]
                        com2_rel = coms_rewrel[p[1],cll]
                        # print(com1_rel,com2_rel,com_diff)
                        if ((abs(com1_rel - np.pi) < epsilon) and 
                        (abs(com2_rel + np.pi) < epsilon)):
                                com_loop_w_in_window.append(cll)
                # get abs value instead
                coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
                com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
                com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
                com_goal=[xx for xx in com_goal if len(xx)>0]
                if len(com_goal)>0:
                    goal_cells_early = intersect_arrays(*com_goal)
                else:
                    goal_cells_early=[]

                # get cells that maintain their coms across at least 2 epochs
                place_window = 20 # cm converted to rad                
                com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
                compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
                # get cells across all epochs that meet crit
                pcs = np.unique(np.concatenate(compc))
                compc=[xx for xx in compc if len(xx)>0]
                if len(compc)>0:
                    pcs_all = intersect_arrays(*compc)
                    # exclude goal cells
                    pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
                else:
                    pcs_all=[]      
                pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
                pc_p=len(pcs_all)/len(coms_correct_abs[0])
                #early
                com_per_ep = np.array([(coms_correct_abs_early[perm[jj][0]]-coms_correct_abs_early[perm[jj][1]]) for jj in range(len(perm))])        
                compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
                # get cells across all epochs that meet crit
                pcs_early = np.unique(np.concatenate(compc))
                compc=[xx for xx in compc if len(xx)>0]
                if len(compc)>0:
                    pcs_all_early = intersect_arrays(*compc)
                    # exclude goal cells
                    pcs_all_early=[xx for xx in pcs_all if xx not in goal_cells_early]
                else:
                    pcs_all_early=[]      
                # get per comparison
                pcs_p_per_comparison_early = [len(xx)/len(coms_correct_abs_early[0]) for xx in compc]
                pc_p_early=len(pcs_all_early)/len(coms_correct_abs_early[0])
                # get other spatially tuned cells
                other_sp = [xx for xx in np.arange(Fc3.shape[1]) if xx not in pcs_all_early and xx not in pcs_all and xx not in goal_cells_early and xx not in goal_cells]
                other_sp_prop.append(len(other_sp)/len(coms_correct[0]))
                # print props
                print(len(other_sp)/len(coms_correct[0]), pc_p, len(goal_cells)/len(coms_correct[0]))
                epoch_perm.append(perm)
                pc_prop.append([pcs_p_per_comparison,pc_p,pcs_p_per_comparison_early,pc_p_early])
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
                print(eptest, perm)       
                
                datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs, tcs_correct_abs_early, coms_correct_abs_early,tcs_fail_abs_early, coms_fail_abs_early,pcs_all]
pdf.close()
# # save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
    pickle.dump(datadct, fp) 
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
df['place_cell_prop_early'] = [xx[3] for xx in pc_prop]
df['place_cell_prop_early']=df['place_cell_prop_early']*100
df['other_sp_prop'] = other_sp_prop
df['other_sp_prop'] = df['other_sp_prop']*100
df['opto'] = df.optoep.values>1
df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['place_cell_prop_shuffle'] =  [xx[1] for xx in place_cell_null]
df['place_cell_prop_shuffle']=df['place_cell_prop_shuffle']*100
df=df[df.place_cell_prop>0]

# number of epochs vs. reward cell prop    
fig,axes = plt.subplots(ncols=2,figsize=(10,5))
# av across mice
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}

df=df[(df.animals!='e189')&(df.animals!='e190')]
# remove outlier days
# df=df[~((df.animals=='z14')&(df.days<33))]
df=df[~((df.animals=='z15')&(df.days<8))]
df=df[~((df.animals=='e217')&(df.days<9)&(df.days==26))]
df=df[~((df.animals=='e216')&((df.days<32)|(df.days.isin([57]))))]
df=df[~((df.animals=='e200')&((df.days.isin([67,68,81]))))]

# df=df[~((df.animals=='e218')&(df.days>44))]

df_plt = df
df_plt = df_plt.groupby(['animals','condition','opto']).mean(numeric_only=True).reset_index()
ax=axes[0]
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
# subtract by led off sessions
# ----------------------------------------
# Plotting Stim - No Stim per Animal

# subtract by led off sessions for both
# ----------------------------------------

# Plotting Stim - No Stim per Animal
# ----------------------------------------

df_an = df_plt.copy()
df_an = df_an.sort_values(['animals', 'condition'])
df_an['opto'] = [True if xx=='True' else False for xx in df_an.opto]

# compute delta for each condition per animal
delta_vals = []
for (animal, condition), group in df_an.groupby(['animals', 'condition']):
    stim = group.loc[group.opto == True].set_index(['animals', 'condition'])[['place_cell_prop', 'place_cell_prop_early', 'other_sp_prop']]
    no_stim = group.loc[group.opto == False].set_index(['animals', 'condition'])[['place_cell_prop', 'place_cell_prop_early','other_sp_prop']]

    if not stim.empty and not no_stim.empty:
        delta_vals.append([animal, condition, 
                            stim.loc[(animal, condition),'place_cell_prop'] - no_stim.loc[(animal, condition), 'place_cell_prop'], 
                            stim.loc[(animal, condition), 'place_cell_prop_early'] - no_stim.loc[(animal, condition), 'place_cell_prop_early'], stim.loc[(animal, condition), 'other_sp_prop'] - no_stim.loc[(animal, condition), 'other_sp_prop']])

df_delta = pd.DataFrame(delta_vals, columns=['animals', 'condition', 'delta_late', 'delta_early', 'delta_other_sp'])

# Now we can plot side by side
fig, axs = plt.subplots(1, 3, figsize=(11,6),sharey=True)

pl ={'ctrl': "slategray", 'vip': 'red', 'vip_ex': 'darkgoldenrod'}
a = 0.7
s = 12

# Plotting late
ax = axs[1]
sns.stripplot(data=df_delta, x='condition', y='delta_late', hue='condition',ax=ax, palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_late',hue='condition', ax=ax, palette=pl, fill=False,errorbar='se')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Late Place')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_late'].quantile(.99)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_late']
    vals2 = data[data['condition'] == cond2]['delta_late']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)

# Plotting early
ax = axs[0]
sns.stripplot(data=df_delta, x='condition', y='delta_early', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_early', ax=ax, 
            palette=pl, fill=False,errorbar='se')
ax.set_ylabel('$\Delta$ Place cell % \n(LEDon-LEDoff)')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Early Place')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_early'].quantile(.85)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_early']
    vals2 = data[data['condition'] == cond2]['delta_early']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)

# other spatially tuned

# Plotting early
ax = axs[2]
sns.stripplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
            palette=pl, fill=False,errorbar='se')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Other spatially tuned')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_other_sp'].quantile(.99)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_other_sp']
    vals2 = data[data['condition'] == cond2]['delta_other_sp']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)
# Save the plot
plt.savefig(os.path.join(savedst, 'place_cell_prop_difference_all.svg'), bbox_inches='tight')

#%% 
# correlate with rates diff
beh = pd.read_csv(r'Z:\condition_df\vip_opto_behavior.csv')
beh=beh[(beh.animals.isin(df.animals.values))&(beh.days.isin(df.days.values))]
beh = beh.groupby(['animals', 'opto']).mean(numeric_only=True).reset_index()
beh=beh[beh.opto==True]
# take all trial cells
y = np.nanmean([df_an.loc[(df_an.opto==True), 'place_cell_prop_early'].values,df_an.loc[(df_an.opto==True), 'place_cell_prop'].values],axis=0)
# y=df_delta.delta_early
# Perform regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(beh.rates_diff.values, y)
print(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")

# Plot scatter plot with regression line
fig,ax=plt.subplots(figsize=(6,5))
sns.scatterplot(x=beh.rates_diff.values, y=y,hue=df_an.loc[(df_an.opto==True),'condition'].values,s=300,alpha=.7,palette=pl,ax=ax)
ax.plot(beh.rates_diff.values, intercept + slope * beh.rates_diff.values, color='steelblue', label='Regression Line',linewidth=3)
ax.legend()
ax.set_xlabel("% Correct trials (LEDon-LEDoff)")
ax.set_ylabel("Place cell %")
ax.set_title(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")
ax.spines[['top', 'right']].set_visible(False)
plt.savefig(os.path.join(savedst, 'placecell_v_performance.svg'), bbox_inches='tight')

#%%
# Change y='place_cell_prop' -> y='other_sp_prop' throughout

fig,ax=plt.subplots()
sns.stripplot(x='opto', y='other_sp_prop',
        hue='condition', data=df_plt,
        palette=pl, dodge=True,
        s=s, alpha=0.7)
sns.barplot(x='opto', y='other_sp_prop', hue='condition',
        data=df_plt,
        palette=pl,
        fill=False, ax=ax, color='k', errorbar='se', legend=False)
