
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
only use spatially tuned!!!
july 2025
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
#%%
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
      'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
      'stat', 'licks'])
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
      licks=fall['licks'][0]
      time=fall['timedFF'][0]
      if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            licks=licks[:-1]
            time=time[:-1]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      diff =np.insert(np.diff(eps), 0, 1e15)
      eps=eps[diff>2000]
      track_length_rad = track_length*(2*np.pi/track_length)
      bin_size=track_length_rad/bins 
      rz = get_rewzones(rewlocs,1/scalingf)       
      # get average success rate
      rates = []
      for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
      rate=np.nanmean(np.array(rates))
      rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates
      # added to get anatomical info
      # takes time
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      #if pc in all but 1
      # looser restrictions
      pc_bool = np.sum(pcs,axis=0)>0
      Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
      # tc w/ dark time
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
      goal_window = 20*(2*np.pi/track_length) # cm converted to rad
      # change to relative value 
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      perm = list(combinations(range(len(coms_correct)), 2)) 
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
      #only get perms with non zero cells
      com_goal=[com for com in com_goal if len(com)>0]
      ######################### SPLIT INTO NEAR VS. FAR #########################
      bounds = [[-np.pi, -np.pi/4], [-np.pi/4,0], [0,np.pi/4], [np.pi/4, np.pi]]
      celltypes = ['far_pre', 'near_pre', 'near_post', 'far_post']
      ######################### PRE v POST #########################
      goal_cell_null_per_celltype=[]; goal_cell_prop_per_celltype=[]
      for kk,celltype in enumerate(celltypes):
         print(celltype)
         com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=bounds[kk][0]) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)<bounds[kk][1]))] if len(com)>0 else [] for com in com_goal]
         #only get perms with non zero cells
         com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]         
         if len(com_goal_postrew)>0:
            goal_cells = intersect_arrays(*com_goal_postrew); 
         else:
            goal_cells=[]
         # get shuffled iteration
         shuffled_dist = np.zeros((num_iterations))
         # max of 5 epochs = 10 perms
         goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
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
            perm = list(combinations(range(len(coms_correct)), 2)) 
            # account for cells that move to the end/front
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
            # cont.
            coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            # in addition, com near but after goal
            com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
               xx], axis=0)>=bounds[kk][0]) & (np.nanmedian(coms_rewrel[:,
               xx], axis=0)<bounds[kk][1]))] if len(com)>0 else [] for com in com_goal_shuf]
            # check to make sure just a subset
            # otherwise reshuffle
            while not sum([len(xx) for xx in com_goal_shuf])>=sum([len(xx) for xx in com_goal_postrew_shuf]):
               print('redo')
               shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
               [random.shuffle(shuf) for shuf in shufs]
               # first com is as ep 1, others are shuffled cell identities
               com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
               com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
               # OR shuffle cell identities
               # relative to reward
               coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
               perm = list(combinations(range(len(coms_correct)), 2)) 
               # account for cells that move to the end/front
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
               # cont.
               coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
               com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
               # get goal cells across all epochs
               com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
               # in addition, com near but after goal
               com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                  xx], axis=0)>=bounds[i][0]) & (np.nanmedian(coms_rewrel[:,
                  xx], axis=0)<bounds[i][1]))] if len(com)>0 else [] for com in com_goal_shuf]

            com_goal_postrew_shuf=[com for com in com_goal_postrew_shuf if len(com)>0]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_shuf]
            if len(com_goal_postrew_shuf)>0:
               goal_cells_shuf = intersect_arrays(*com_goal_postrew_shuf); 
            else:
               goal_cells_shuf=[]
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison

         goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
         goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps))
         goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
         goal_cell_p_per_comp = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
         goal_cell_null_per_celltype.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
         goal_cell_prop_per_celltype.append([goal_cell_p,goal_cell_p_per_comp])
      goal_cell_prop.append(goal_cell_prop_per_celltype)
      goal_cell_null.append(goal_cell_null_per_celltype)
      datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,  goal_cell_prop_per_celltype, goal_cell_null_per_celltype]

pdf.close()
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
