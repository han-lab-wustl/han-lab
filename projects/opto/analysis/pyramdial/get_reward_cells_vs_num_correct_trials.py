"""
get reward distance cells between opto and non opto conditions
oct 2024
mods in june 2025
control vs. opto epoch only
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
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto_w_trial_num
import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# initialize var
# radian_alignment_saved = {} # overwrite
results_all=[]
radian_alignment = {}
cm_window = 20

#%%
# iterate through all animals 
for ii in range(len(conddf)):
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   # skip e217 day
   if ii!=202:#(conddf.optoep.values[ii]>1):
      if animal=='e145': pln=2  
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      radian_alignment, results_pre, results_post, results_pre_early, results_post_early,pre_strials, opto_strials,pre_total_trials,opto_total_trials = get_rew_cells_opto_w_trial_num(
         params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
         radian_alignment, cm_window=cm_window)
      results_all.append([results_pre, results_post, results_pre_early, results_post_early,pre_strials, opto_strials,pre_total_trials,opto_total_trials])

pdf.close()
# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 


# %%
# rew cell %
# separate out variables
df = conddf.copy()
df = df.drop([202]) # skipped e217 day
# df=df.iloc[:120]
pre_late = [xx[0] for xx in results_all]
post_late = [xx[1] for xx in results_all]
pre_early = [xx[2] for xx in results_all]
post_early = [xx[3] for xx in results_all]
prev_trialnum = [len(xx[4]) for xx in results_all]
opto_trialnum = [len(xx[5]) for xx in results_all]
prev_trialnum_total = [xx[6] for xx in results_all]
opto_trialnum_total = [xx[7] for xx in results_all]

plt.rc('font', size=20)
# concat all cell type goal cell prop
all_cells = [pre_late, post_late, pre_early, post_early]
goal_cell_prop = np.concatenate([[xx['goal_cell_prop'] for xx in cll] for cll in all_cells])
pl = {'ctrl':'slategray','vip':'red','vip_ex':'darkgoldenrod'}
realdf= pd.DataFrame()
realdf['goal_cell_prop']=goal_cell_prop
lbl = ['pre_late', 'post_late', 'pre_early', 'post_early']
realdf['cell_type']=np.concatenate([[lbl[kk]]*len(cll) for kk,cll in enumerate(all_cells)])
realdf['animal']=np.concatenate([df.animals]*len(all_cells))
realdf['prev_trialnum']=np.concatenate([prev_trialnum]*len(all_cells))
realdf['opto_trialnum']=np.concatenate([opto_trialnum]*len(all_cells))

realdf['opto_trialnum_total']=np.concatenate([opto_trialnum_total]*len(all_cells))
realdf['prev_trialnum_total']=np.concatenate([prev_trialnum_total]*len(all_cells))
realdf['optoep']=np.concatenate([df.optoep]*len(all_cells))
realdf['opto']=[True if xx>1 else False for xx in realdf['optoep']]
realdf['condition']=np.concatenate([df.in_type]*len(all_cells))
realdf['condition']=[xx if 'vip' in xx else 'ctrl' for xx in realdf.condition.values]
realdf['day']=np.concatenate([df.days]*len(all_cells))
# realdf['goal_cell_prop'] = realdf['goal_cell_prop'] - realdf['goal_cell_prop_shuf']
realdf=realdf[realdf['goal_cell_prop']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e190')]
# # remove outlier days
realdf=realdf[~((realdf.animal=='e201')&((realdf.day>62)))]
realdf=realdf[~((realdf.animal=='z14')&((realdf.day<33)))]
# realdf=realdf[~((realdf.animal=='z16')&((realdf.day>13)))]
realdf=realdf[~((realdf.animal=='z15')&((realdf.day<8)|(realdf.day.isin([15]))))]
# realdf=realdf[~((realdf.animal=='e217')&((realdf.day<9)|(realdf.day.isin([21,29,30,26]))))]
# realdf=realdf[~((realdf.animal=='e216')&((realdf.day<32)|(realdf.day.isin([47,55,57]))))]
realdf=realdf[~((realdf.animal=='e200')&((realdf.day.isin([67,68,81]))))]
# realdf=realdf[~((realdf.animal=='e218')&(realdf.day.isin([41,55])))]
# reward cell v. trialnum
realdf['goal_cell_prop']=realdf['goal_cell_prop']*100
df = realdf[realdf.opto==True]

fig,ax=plt.subplots()
sns.swarmplot(y='goal_cell_prop',x='opto_trialnum',hue='condition',data=df,palette=pl)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Prepare figure
fig, axes = plt.subplots(ncols = 2,figsize=(8,5),sharey=True)
ax=axes[0]

# Swarmplot
sns.swarmplot(
    y='goal_cell_prop',
    x='prev_trialnum',
    hue='condition',
    data=df,
    palette=pl,
    ax=ax,
    alpha=0.5,
    s=8
)

# Fit regression for each condition with confidence interval
# conditions = df['condition'].unique()
# for cond in conditions:
#     sub = df[df['condition'] == cond]
#     X = sub['prev_trialnum'].values
#     y_vals = sub['goal_cell_prop'].values

#     # Fit linear model
#     X_const = sm.add_constant(X)
#     model = sm.OLS(y_vals, X_const).fit()

#     # Prediction for plotting
#     x_pred = np.linspace(X.min(), X.max()-15, 100)
#     X_pred_const = sm.add_constant(x_pred)
#     y_pred = model.predict(X_pred_const)
    
#     # Get 95% confidence interval
#     pred_summary = model.get_prediction(X_pred_const).summary_frame(alpha=0.05)
#     y_lower = pred_summary['obs_ci_lower']
#     y_upper = pred_summary['obs_ci_upper']
    
#     # Plot regression line
#     ax.plot(x_pred, y_pred, color=pl[cond], linewidth=2)
#     # Plot shaded confidence interval
#    #  ax.fill_between(x_pred, y_lower, y_upper, color=pl[cond], alpha=0.2)
#     # Annotate r and p
#     r = np.corrcoef(X, y_vals)[0,1]
#     p = model.pvalues[1]  # p-value for slope
#     ax.text(X.max()*0.2, y_pred.max(), f'{cond}: r={r:.2g}, p={p:.2g}', color=pl[cond])
ax.legend_.remove()  # remove existing legend
ax.set_xlabel('# of correct trials')
ax.set_ylabel('Reward cell %')
# ax.set_xticks([0,2,4])
# ax.set_xticklabels([1, 11, 22])

ax.set_title('LED off epoch')
sns.despine()

ax=axes[1]
# Swarmplot
sns.swarmplot(
    y='goal_cell_prop',
    x='opto_trialnum',
    hue='condition',
    data=df,
    palette=pl,
    ax=ax,
    alpha=0.5,
    s=8
)

# Fit regression for each condition with confidence interval
conditions = df['condition'].unique()
for cond in conditions:
    sub = df[df['condition'] == cond]
    X = sub['opto_trialnum'].values
    y_vals = sub['goal_cell_prop'].values

    # Fit linear model
    X_const = sm.add_constant(X)
    model = sm.OLS(y_vals, X_const).fit()

    # Prediction for plotting
    x_pred = np.linspace(X.min(), X.max()-3, 100)
    X_pred_const = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred_const)
    
    # Get 95% confidence interval
    pred_summary = model.get_prediction(X_pred_const).summary_frame(alpha=0.05)
    y_lower = pred_summary['obs_ci_lower']
    y_upper = pred_summary['obs_ci_upper']
    
    # Plot regression line
    ax.plot(x_pred, y_pred, color=pl[cond], linewidth=2)
    # Plot shaded confidence interval
   #  ax.fill_between(x_pred, y_lower, y_upper, color=pl[cond], alpha=0.2)
    # Annotate r and p
    r = np.corrcoef(X, y_vals)[0,1]
    p = model.pvalues[1]  # p-value for slope
    ax.text(X.max()*0.2, y_pred.max(), f'{cond}: r={r:.2g}, p={p:.2g}', color=pl[cond])
    
# After plotting everything
handles, labels = ax.get_legend_handles_labels()

# Map old labels to new ones
label_mapping = {
    'vip': 'VIP Inhibition',
    'vip_ex': 'VIP Excitation',
    'ctrl': 'Control',
    # Add any other labels if needed
}
new_labels = [label_mapping.get(lbl, lbl) for lbl in labels]
ax.legend(handles, new_labels, title="Condition", loc='best')
ax.set_xlabel('# of correct trials')
ax.set_ylabel('Reward cell %')
ax.set_xticks([0, 9, 18])
ax.set_xticklabels([1, 11, 22])
ax.set_title('LED on epoch')
sns.despine()
#%%
# just led on
# Prepare figure
fig, ax = plt.subplots(figsize=(6,5))

# Swarmplot
sns.swarmplot(
    y='goal_cell_prop',
    x='opto_trialnum',
    hue='condition',
    data=df,
    palette=pl,
    ax=ax,
    alpha=0.5,
    s=8
)

# Fit regression for each condition with confidence interval
conditions = df['condition'].unique()
for cond in conditions:
    sub = df[df['condition'] == cond]
    X = sub['opto_trialnum'].values
    y_vals = sub['goal_cell_prop'].values

    # Fit linear model
    X_const = sm.add_constant(X)
    model = sm.OLS(y_vals, X_const).fit()

    # Prediction for plotting
    x_pred = np.linspace(X.min(), X.max()-3, 100)
    X_pred_const = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred_const)
    
    # Get 95% confidence interval
    pred_summary = model.get_prediction(X_pred_const).summary_frame(alpha=0.05)
    y_lower = pred_summary['obs_ci_lower']
    y_upper = pred_summary['obs_ci_upper']
    
    # Plot regression line
    ax.plot(x_pred, y_pred, color=pl[cond], linewidth=2)
    # Plot shaded confidence interval
   #  ax.fill_between(x_pred, y_lower, y_upper, color=pl[cond], alpha=0.2)
    # Annotate r and p
    r = np.corrcoef(X, y_vals)[0,1]
    p = model.pvalues[1]  # p-value for slope
    ax.text(X.max()*0.2, y_pred.max(), f'{cond}: r={r:.2g}, p={p:.2g}', color=pl[cond])
    
# After plotting everything
handles, labels = ax.get_legend_handles_labels()

# Map old labels to new ones
label_mapping = {
    'vip': 'VIP Inhibition',
    'vip_ex': 'VIP Excitation',
    'ctrl': 'Control',
    # Add any other labels if needed
}
new_labels = [label_mapping.get(lbl, lbl) for lbl in labels]
ax.legend(handles, new_labels, title="Condition", loc='best')
ax.set_xlabel('# of correct trials')
ax.set_ylabel('Reward cell %')
ax.set_xticks([0, 9, 18])
ax.set_xticklabels([1, 11, 22])
ax.set_title('LED on epoch')
sns.despine()
plt.savefig(os.path.join(savedst,'trial_v_reward_cell.svg'),bbox_inches='tight')

#%%

# just led on
# Prepare figure
fig, ax = plt.subplots(figsize=(6,5))

# Swarmplot
sns.swarmplot(
    y='goal_cell_prop',
    x='opto_trialnum_total',
    hue='condition',
    data=df,
    palette=pl,
    ax=ax,
    alpha=0.5,
    s=8
)

# Fit regression for each condition with confidence interval
conditions = df['condition'].unique()
for cond in conditions:
    sub = df[df['condition'] == cond]
    X = sub['opto_trialnum_total'].values
    y_vals = sub['goal_cell_prop'].values

    # Fit linear model
    X_const = sm.add_constant(X)
    model = sm.OLS(y_vals, X_const).fit()

    # Prediction for plotting
    x_pred = np.linspace(X.min(), X.max()-20, 100)
    X_pred_const = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred_const)
    
    # Get 95% confidence interval
    pred_summary = model.get_prediction(X_pred_const).summary_frame(alpha=0.05)
    y_lower = pred_summary['obs_ci_lower']
    y_upper = pred_summary['obs_ci_upper']
    
    # Plot regression line
    ax.plot(x_pred, y_pred, color=pl[cond], linewidth=2)
    # Plot shaded confidence interval
   #  ax.fill_between(x_pred, y_lower, y_upper, color=pl[cond], alpha=0.2)
    # Annotate r and p
    r = np.corrcoef(X, y_vals)[0,1]
    p = model.pvalues[1]  # p-value for slope
    ax.text(X.max()*0.2, y_pred.max(), f'{cond}: r={r:.2g}, p={p:.2g}', color=pl[cond])
    
# After plotting everything
handles, labels = ax.get_legend_handles_labels()

# Map old labels to new ones
label_mapping = {
    'vip': 'VIP Inhibition',
    'vip_ex': 'VIP Excitation',
    'ctrl': 'Control',
    # Add any other labels if needed
}
new_labels = [label_mapping.get(lbl, lbl) for lbl in labels]
ax.legend(handles, new_labels, title="Condition", loc='best')
ax.set_xlabel('# of total trials')
ax.set_ylabel('Reward cell %')
ax.set_xticks([0, 16, 33])
ax.set_xticklabels([11, 27, df.opto_trialnum_total.max()])
ax.set_title('LED on epoch')
sns.despine()
plt.savefig(os.path.join(savedst,'all_trial_v_reward_cell.svg'),bbox_inches='tight')