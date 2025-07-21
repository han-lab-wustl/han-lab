
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
only use spatially tuned!!!
july 2025
DOESNT WORK
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,\
    make_tuning_curves_by_trialtype_w_darktime,get_radian_position_first_lick_after_rew_w_dt
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle, intersect_arrays
from reward_shuffle import allrewardsubtypes
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_allcelltypes.p"
# with open(saveddataset, "rb") as fp: #unpickle
#         radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'dark_time_tuning.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
####################################### RUN CODE #######################################
# initialize var
radian_alignment_saved = {} # overwrite
goal_cell_prop=[]
bins = 90
goal_window_cm=20
num_iterations=1000
datadct = {}
goal_cell_null= []
perms = []
# goal_window_cm = np.arange(5,135,5) # cm
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      datadct=allrewardsubtypes(params_pth,animal,day,ii,conddf)

with open(saveddataset, "wb") as fp:   #Pickling
      pickle.dump(datadct, fp) 

####################################### RUN CODE #######################################
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
dfs=[]
celltypes=['Far pre-reward','Near pre-reward','Near post-reward','Far post-reward']
for cll,celltype in enumerate(celltypes):
   inds = [int(xx[-3:]) for xx in datadct.keys()]
   df = conddf.copy()
   df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
   df['num_epochs'] = [len(xx[1]) for k,xx in datadct.items()]
   df['goal_cell_prop'] =  [xx[4][cll][0] for k,xx in datadct.items()]
   df['opto'] = df.optoep.values>1   
   df['day'] = df.days
   df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
   df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
   df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
   # average shuffle
   df['goal_cell_prop_shuffle'] =  [xx[5][cll][1] for k,xx in datadct.items()]
   df['cell_type'] = [celltype]*len(df)
   # per comparison
   df_perms = pd.DataFrame()
   goal_cell_perm = [xx[4][cll][1] for k,xx in datadct.items()]
   goal_cell_perm_shuf = [xx[5][cll][0][~np.isnan(xx[5][cll][0])] for k,xx in datadct.items()]
   df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
   # HACK
   df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)[:len(df_perms)]
   df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
   df_perms['animals'] = np.concatenate(df_perm_animals)
   df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
   df_perms['session_num'] = np.concatenate(df_perm_days)
   # take a mean of all epoch comparisons
   df_perms['num_epochs'] = [2]*len(df_perms)
   df_perms['cell_type'] = [celltype]*len(df_perms)
   df=pd.concat([df, df_perms])
   dfs.append(df)

#%%
from statsmodels.stats.multitest import multipletests
plt.rc('font', size=20) 
df = pd.concat(dfs)
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(7,4))
df_plt = df[df.num_epochs<5]
order=celltypes

colors = [tuple([c * 0.4 for c in sns.color_palette("Dark2")[0]]),sns.color_palette("Dark2")[0],sns.color_palette("Dark2")[1],tuple([c * 0.4 for c in sns.color_palette("Dark2")[1]])]

# av across mice
df_plt = df_plt.groupby(['animals','num_epochs','cell_type']).mean(numeric_only=True)
df_plt=df_plt.reset_index()
df_plt['goal_cell_prop']=df_plt['goal_cell_prop']*100

df_plt['goal_cell_prop_shuffle']=df_plt['goal_cell_prop_shuffle']*100
exan=['e189','e139']
df_plt=df_plt[~df_plt.animals.isin(exan)]
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt, hue='cell_type',hue_order=order,palette=colors,
        s=7,ax=ax,dodge=True,alpha=.7)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,legend=False,palette=colors,
        fill=False,ax=ax, hue='cell_type', errorbar='se',hue_order=order)
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', hue='cell_type',
        alpha=0.3, err_kws={'color': 'grey'},errorbar=None,ax=ax,legend=False,hue_order=order)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
# --- Collect all tests ---
results = []
for ep in sorted(df_plt.num_epochs.unique()):
   for i, ct in enumerate(order):
      dsub = df_plt[(df_plt.cell_type == ct) & (df_plt.num_epochs == ep)]
      if len(dsub) >= 2:
         stat, pval = scipy.stats.wilcoxon(dsub['goal_cell_prop'], dsub['goal_cell_prop_shuffle'])
         results.append({'epoch': ep, 'cell_type': ct, 'pval': pval, 'xidx': i, 'ymax': dsub[['goal_cell_prop', 'goal_cell_prop_shuffle']].values.max()})
# --- FDR correction ---
raw_pvals = [r['pval'] for r in results]
rej, pvals_corr, _, _ = multipletests(raw_pvals, method='fdr_bh')
# --- Annotate plot ---
for r, pcorr, rj in zip(results, pvals_corr, rej):
    ep = r['epoch']
    ct = r['cell_type']
    xpos = ep - 2.3 + (.2 * r['xidx'])
    ymax = r['ymax']-2
    
    # Significance stars
    if pcorr < 0.001:
        star = '***'
    elif pcorr < 0.01:
        star = '**'
    elif pcorr < 0.05:
        star = '*'
    else:
        star = ''
    
    if star:
        ax.text(xpos, ymax + 2.2, star, ha='center', fontsize=25)
    else:
        ax.text(xpos, ymax + 2.2, f'{pcorr:.2f}', ha='center', fontsize=10)

# --- Final cleanup ---
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Reward cell %')
ax.set_xlabel('# of epochs')
ax.legend(title='Cell type',fontsize=14,title_fontsize=14)

# fig.suptitle('Including delay period')
plt.savefig(os.path.join(savedst, 'allrewtype_cell_prop.svg'), 
        bbox_inches='tight')
#%%
from scipy.optimize import curve_fit

def exp_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C
fit_results = {}
for ct in order:
   tau_all = []
   for an in df_plt.animals.unique():
      dct = df_plt[(df_plt.cell_type == ct) & (df_plt.animals==an)]
      initial_guess = [4, 2]
      x = dct['num_epochs'].values
      y = dct['goal_cell_prop'].values
      popt, _ = curve_fit(exp_decay, x, y, p0=(y.max(), 1.0, y.min()))
      tau_fit = popt[1]  # tau
      fit_results[ct] = tau_fit
      tau_all.append({'cell_type': ct, 'tau': popt[1]})
   fit_results[ct]=tau_all
#%%      
tau_df = pd.DataFrame(taus)

far = tau_df[tau_df['cell_type'].str.contains('far|pre', case=False)]['tau']
near = tau_df[tau_df['cell_type'].str.contains('near|post', case=False)]['tau']

# Welch's t-test (or Mann-Whitney if not normal)
tstat, pval = scipy.stats.wilcoxon(far, near)
print(f"Tau comparison (far vs near): t={tstat:.2f}, p={pval:.4f}")

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
