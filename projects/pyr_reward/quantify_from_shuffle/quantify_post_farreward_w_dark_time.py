
"""
zahra
2025
far reward
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from statsmodels.stats.multitest import multipletests
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_post_farrew
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'post_far_rew_w_dt.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
radian_alignment_saved = {} # overwrite
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
goal_window_cm = 20
lasttr=8 # last trials
bins=90
dists = []
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_post_farreward.p"
# iterate through all animals
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                radian_alignment,rate,p_value,total_cells,goal_cell_iind,goal_cell_prop,num_epochs,\
                        goal_cell_null,epoch_perm,pvals=extract_data_post_farrew(ii,params_pth,\
                        animal,day,bins,radian_alignment,radian_alignment_saved,goal_window_cm,
                        pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
                        total_cells)
pdf.close()
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['day'] = df.days
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')

# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
df_plt=df_plt.reset_index()
df_plt=df_plt[df_plt.num_epochs<5]
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.num_epochs-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    
# include all comparisons 
df_perms = pd.DataFrame()
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_permsav2=df_permsav2.reset_index()

# compare to shuffle
# df_permsav2=df_permsav2.reset_index()
df_plt2 = pd.concat([df_permsav2,df_plt])
df_plt2 = df_plt2[(df_plt2.animals!='e200') & (df_plt2.animals!='e189')]
df_plt2 = df_plt2[df_plt2.num_epochs<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2['goal_cell_prop']=df_plt2['goal_cell_prop']*100
df_plt2['goal_cell_prop_shuffle']=df_plt2['goal_cell_prop_shuffle']*100
df_plt2=df_plt2.reset_index()
df_plt2 = df_plt2[df_plt2.animals!='e139']

# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(ncols=2,figsize=(7,5))
ax=axes[0]
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, 
#         y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
# bar plot of shuffle instead
sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_ylabel('Far reward cell %')
eps = [2,3,4]
y = 28
pshift = 1
fs=36
pvalues=[]
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        pvalues.append(pval)
        print(f'{ep} epochs, pval: {pval}')
# correct pvalues
reject, pvals_corrected, _, _ = multipletests(pvalues, method='bonferroni')

for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10,rotation=45)
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
ax.set_title('Far post-reward cells',pad=30)
ax.set_xlabel('')
ax.set_ylim([0,30])
ax=axes[1]
# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt2['goal_cell_prop_sub_shuffle'] = df_plt2['goal_cell_prop']-df_plt2['goal_cell_prop_shuffle']
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',color='cornflowerblue',
        data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
# make lines
df_plt2=df_plt2.reset_index()
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('')
ax.set_title('Far post-reward cell %-shuffle',pad=30)
ax.set_ylim([-1,8])

plt.savefig(os.path.join(savedst, 'post_farreward_dark_time_cell_prop-shuffle_per_an.svg'), 
        bbox_inches='tight')

#%% 
# find tau/decay

from scipy.optimize import curve_fit

# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)
tau_all = []; y_fit_all = []
# df_plt2=df_plt2.reset_index()
for an in df_plt2.animals.unique():
        # Initial guesses for the optimization
        initial_guess = [4, 2]  # Amplitude guess and tau guess
        y = df_plt2[df_plt2.animals==an]
        t=np.array([2,3,4])
        # Fit the model to the data using curve_fit
        params, params_covariance = curve_fit(exponential_decay, t, y.goal_cell_prop.values, p0=initial_guess)
        # Extract the fitted parameters
        A_fit, tau_fit = params
        tau_all.append(tau_fit)
        # Generate the fitted curve using the optimized parameters
        y_fit = exponential_decay(t, A_fit, tau_fit)
        y_fit_all.append(y_fit)
        
# plot fit
fig,ax = plt.subplots(figsize=(3,5))
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7)
plt.plot(np.array(y_fit_all).T,color='grey')

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward-centric cell proportion')
# plt.savefig(os.path.join(savedst, 'expo_fit_reward_centric.png'), 
#         bbox_inches='tight')

#%%
# compare to persistent cells
tau_all_postrew = [2.5081012042756297,
 1.3564559842969026,
 3.067144722017443,
 4.017594663159857,
 2.307820130958938,
 1.814554708027948,
 1.9844914154882163,
 1.7613987163758171,
 2.403541822072123]

tau_all_prerew =[2.349997695562105,
 3.3542296271312786,
 5.401831519039983,
 2.3131417522099587,
 2.177319788868507,
 1.4432255209360099,
 1.3448952235370013,
 1.3658867417424119,
 6.234399065992553]

tau_far_prerew = [1.902184232684948,
 1.1130375903752456,
 1.1891335993849583,
 0.42929035493938306,
 1.4182488607462183,
 1.0666197205416246,
 0.6676340435284304,
 0.7522738857806301,
 0.7907558229642044]

df = pd.DataFrame()
df['tau'] = np.concatenate([tau_far_prerew,tau_all,tau_all_postrew,tau_all_prerew])
df['cell_type'] =np.concatenate([['Far pre-reward']*len(tau_far_prerew),
                                ['Far post-reward']*len(tau_all),
                                ['Post-reward']*len(tau_all_postrew),
                                ['Pre-reward']*len(tau_all_prerew)])
order = ['Pre-reward', 'Post-reward', 'Far pre-reward','Far post-reward']
# number of epochs vs. reward cell prop incl combinations    
# make sure outlier numbers aren't there?
df=df[df.tau<10]
fig,ax = plt.subplots(figsize=(3.5,5))
# av across mice
sns.stripplot(x='cell_type', y='tau',color='k',
        data=df,s=10,alpha=0.7,
        order=order,
)
sns.barplot(x='cell_type', y='tau',
        data=df, fill=False,ax=ax, color='k', errorbar='se',
        order=order)
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)
ax.set_ylabel(f'Decay over epochs ($\\tau$)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)

import scipy.stats as stats
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests

df = df.reset_index()
# Get unique groups
groups = df['cell_type'].unique()

# Generate all pairwise combinations
comparisons = list(combinations(groups, 2))

# Perform t-tests
p_values = []
for group1, group2 in comparisons:
    data1 = df[df['cell_type'] == group1]['tau']
    data2 = df[df['cell_type'] == group2]['tau']
    stat, p = scipy.stats.ranksums(data1, data2)
    p_values.append(p)

# Apply Bonferroni correction
adjusted = multipletests(p_values, method='bonferroni')
adjusted_p_values = adjusted[1]
# Define y-position for annotations
y_max = df['tau'].max()
y_offset = y_max * 0.1  # adjust as needed
fs=40
pshift=1.5
# Add annotations
for i, (group1, group2) in enumerate(comparisons):
    x1 = list(groups).index(group1)
    x2 = list(groups).index(group2)
    y = y_max + y_offset * (i + 1)
    p_val = adjusted_p_values[i]
    if p_val < 0.001:
        significance = '***'
    elif p_val < 0.01:
        significance = '**'
    elif p_val < 0.05:
        significance = '*'
    else:
        significance = ''
    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-1, significance, ha='center', va='bottom', color='k',
            fontsize=fs)
    ax.text((x1 + x2) / 1.5, y-.5 + pshift, f'p={p_val:.2g}', ha='center', rotation=45, fontsize=12)

# ax.set_title('Ranksum and bonferroni')
plt.savefig(os.path.join(savedst, 'decay_rewardcell_dark_time.svg'), 
        bbox_inches='tight')

#%%
# as a function of session/day
df_plt = df.groupby(['animals','session_num','num_epochs']).mean(numeric_only=True)
df_permsav2 = df_perms.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')==2]
df_plt2 = df_plt2.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(7,5))
# av across mice
sns.stripplot(x='session_num', y='goal_cell_prop',hue='animals',
        data=df_plt2,s=10,alpha=0.7)
sns.barplot(x='session_num', y='goal_cell_prop',color='darkslateblue',
        data=df_plt2,fill=False,ax=ax, errorbar='se')
ax.set_xlim([-.5,9.5])
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('# of sessions')
ax.set_ylabel('Reward-distance cell proportion')
df_reset = df_plt2.reset_index()
sns.regplot(x='session_num', y='goal_cell_prop',
        data=df_reset, scatter=False, color='k')
r, p = scipy.stats.pearsonr(df_reset['session_num'], 
        df_reset['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.3f}, p={:.3g}'.format(r, p),
        transform=ax.transAxes)
ax.set_title('2 epoch combinations')

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

# all animals
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='success_rate', y='goal_cell_prop',hue='animals',
        data=df,
        s=150, ax=ax)
sns.regplot(x='success_rate', y='goal_cell_prop',
        data=df,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df['success_rate'], 
        df['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Success rate')
ax.set_ylabel('Reward cell proportion')
#%%
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