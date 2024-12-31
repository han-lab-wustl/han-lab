"""zahra's dopamine hrz analysis
feb 2024
for chr2 experiments
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from projects.memory.behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from projects.memory.dopamine import get_rewzones, extract_vars

# plt.rc('font', size=12)          # controls default text sizes

plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"halo_opto_peri_analysis.pdf"))

condrewloc = pd.read_csv(r"C:\Users\Han\Downloads\data_organization - halo_grab.csv", index_col = None)
# convert rewlcos to float
# Drop rows with non-numeric values
condrewloc = condrewloc[pd.to_numeric(condrewloc['rewloc'], errors='coerce').notnull()]
condrewloc = condrewloc[pd.to_numeric(condrewloc['prevrewloc'], errors='coerce').notnull()]
# Convert to float
condrewloc[['rewloc', 'prevrewloc']] = condrewloc[['rewloc', 'prevrewloc']].astype(float)
condrewloc['Day'] = condrewloc['Day'].astype(int)
condrewloc['Opto'] = [1 if xx=='TRUE' else 0 for xx in condrewloc['Opto'].values]

src = r"Y:\halo_grabda"
# animals = ['e241','e243']#,'e242','e243']
animals = ['e243']
# days_all = [[34,35,36,37,38,39,40],[35,36,37,38,39,40,41]]#,[29,30],[36,37]]
days_all = [[35,36,38,39,40,41,42,43]]
opto_cond = 'Opto' # experiment condition
rolling_win = 3 # 3 for significance in 10 trial on/ 1 off
# optodays = [18, 19, 22, 23, 24]
#%%
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        # extract variables and makes plots into pdf
        plndff = extract_vars(src, animal, day, condrewloc, opto_cond, dst,
        pdf, rolling_win=5, planes=4)
        day_date_dff[f'{animal}_{day}'] = plndff
pdf.close()
#%%
plt.rc('font', size=20)          # controls default text sizes
# plot mean and sem of opto days vs. control days
# on same plane
# 1 - set conditions
days_all=[[int(yy) for yy in xx] for xx in days_all]
animals_days=[[f'{animal[ii]}_{yy}' for yy in xx] for ii,xx in enumerate(days_all)]

planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
opto_condition = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & \
    (condrewloc.Animal==animal)), opto_cond].values.astype(float) for ii,animal in enumerate(animals)])
animal = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
            'Animal'].values for ii,animal in enumerate(animals)])
opto_condition = np.array([True if xx==1 else False for xx in opto_condition])
opto_days = np.concatenate(animals_days)[opto_condition]
# only mean values
day_date_dff_arr = np.array([[[p[0],p[1]] for p in v] for k,v in day_date_dff.items()])
# all trials
day_date_dff_arr_all_tr = [[[p[2],p[3]] for p in v] for k,v in day_date_dff.items()]
day_date_dff_arr_opto_all_tr = [[[p[2],p[3]] for p in v] for k,v in day_date_dff.items() \
    if k in opto_days]
day_date_dff_arr_nonopto_all_tr = [[[p[2],p[3]] for p in v] for k,v in day_date_dff.items() \
    if k not in opto_days]
day_date_dff_arr_opto = day_date_dff_arr[opto_condition]
animal_opto = animal[opto_condition]
animal_nonopto = animal[~opto_condition]
day_date_dff_arr_nonopto = day_date_dff_arr[~opto_condition]
learning_day = np.concatenate([condrewloc.loc[((condrewloc.Animal==an)&(condrewloc.Day.isin(days_all[ii]))), 'learning_day'].values-1 for ii,an in enumerate(animals)])
# rewzone_learning = np.concatenate([get_rewzones(condrewloc.loc[((condrewloc.Animal==an)&(condrewloc.Day.isin(days_all[ii]))), 'RewLoc'].values[0], 1/gainf) for ii,an in enumerate(animals)])
learning_day_opto = learning_day[opto_condition].astype(int)
learning_day_nonopto = learning_day[~opto_condition].astype(int)
# rewzone_learning_opto = rewzone_learning[opto_condition]
# rewzone_learning_nonopto = rewzone_learning[~opto_condition]
height = 1.035 # ylim
#%%
# 2 -quantify so transients
# get time period around stim
range_val=5;binsize=0.2
time_rng = range(int(range_val/binsize-0/binsize),
            int(range_val/binsize+(1/binsize))) # during and after stim
before_time_rng = range(int(range_val/binsize-2/binsize),
            int(range_val/binsize-0/binsize)) # during and after stim
# normalize pre-window to 1
# remember than here we only take led off trials bc of artifact
trialtype=1
so_traces = day_date_dff_arr_opto[:,3,trialtype,:]#[learning_day_opto==1]
so_transients_opto = [so_traces[ii,:]-np.nanmean(so_traces[ii,before_time_rng]) for ii,xx in enumerate(range(so_traces.shape[0]))]
so_transients_opto = [np.nanmax(xx[time_rng]) for xx in so_transients_opto]
trialtype=0
so_traces = day_date_dff_arr_nonopto[:,3,trialtype,:]
so_transients_nonopto = [so_traces[ii,:]-np.nanmean(so_traces[ii,before_time_rng]) for ii,xx in enumerate(range(so_traces.shape[0]))]
so_transients_nonopto = [np.nanmax(xx[time_rng]) for xx in so_transients_nonopto]

fig, ax = plt.subplots(figsize=(2.2,5))
df = pd.DataFrame(np.concatenate([so_transients_opto,so_transients_nonopto]),columns=['so_transient_dff_difference'])
df['condition'] = np.concatenate([['LED on']*len(so_transients_opto), ['LED off']*len(so_transients_nonopto)])
df['animal'] = np.concatenate([animal_opto, animal_nonopto])
df = df.sort_values('condition')
df_plt = df.groupby(['animal', 'condition']).mean(numeric_only=True)
ax = sns.barplot(x='condition', y='so_transient_dff_difference',hue='condition', data=df_plt, fill=False,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"},)
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df_plt,s=15,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df,s=12,
    alpha=0.5,palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
for i in range(len(df.animal.unique())):
    ax = sns.lineplot(x='condition', y='so_transient_dff_difference', data=df[df.animal==df.animal.unique()[i]],
            errorbar=None, color='dimgray', linewidth=2)

# ax.set_ylim(0, 1.04)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\Delta$ F/F (stratum oriens)')
ledon, ledoff = df.loc[(df.condition=='LED on'), 'so_transient_dff_difference'].values, df.loc[(df.condition=='LED off'), 'so_transient_dff_difference'].values
t,pval = scipy.stats.ranksums(ledon[~np.isnan(ledon)], ledoff)
ledon, ledoff = df_plt.loc[(df_plt.index.get_level_values('condition')=='LED on'), 
                'so_transient_dff_difference'].values, df_plt.loc[(df_plt.index.get_level_values('condition')=='LED off'), 'so_transient_dff_difference'].values
t,pval_an = scipy.stats.ttest_rel(ledon[~np.isnan(ledon)], ledoff)

ax.set_title(f'Stim at reward\n\
    per session p={pval:.4f}\n per animal p={pval_an:.4f}',fontsize=12)

# plt.savefig(os.path.join(dst, 'so_transient_quant.svg'), bbox_inches='tight')
#%%
# 2.1 quantify superficial layer slopes
plt.rc('font', size=22)
# get slope time
time_rng = range(0,int(range_val/binsize)) # during and after stim

pln=[0,1,2] # average of sup
# remember than here we only take led off trials bc of artifact
# though that doesn't matter here for slope
trialtype=0
slopes_opto = []
for ii,xx in enumerate(range(day_date_dff_arr_opto.shape[0])):
    slope, _,__,___,___ = scipy.stats.linregress(time_rng,
                np.nanmean(day_date_dff_arr_opto[ii,:3,trialtype,time_rng],axis=1))
    slopes_opto.append(slope)
#trailtype=0
trialtype=0
slopes_nonopto = []
for ii,xx in enumerate(range(day_date_dff_arr_nonopto.shape[0])):
    slope, _,__,___,___ = scipy.stats.linregress(time_rng,
                np.nanmean(day_date_dff_arr_nonopto[ii,:3,trialtype,time_rng],axis=1))
    slopes_nonopto.append(slope)

fig, ax = plt.subplots(figsize=(2.2,5))
df = pd.DataFrame(np.concatenate([slopes_opto,slopes_nonopto]),columns=['slopes'])
df['condition'] = np.concatenate([['LED on']*len(slopes_opto), ['LED off']*len(slopes_nonopto)])
df['animal'] = np.concatenate([animal_opto, animal_nonopto])
df = df.sort_values('condition')
df_plt = df.groupby(['animal', 'condition']).mean(numeric_only=True)
ax = sns.barplot(x='condition', y='slopes',hue='condition', data=df_plt, fill=False,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"},)
ax = sns.stripplot(x='condition', y='slopes', hue='condition', data=df_plt,s=15,
            palette={'LED off': "slategray", 'LED on': "mediumturquoise"})

for i in range(len(df.animal.unique())):
    ax = sns.lineplot(x='condition', y='slopes', data=df[df.animal==df.animal.unique()[i]],
            errorbar=None, color='dimgray', linewidth=2)
ax = sns.stripplot(x='condition', y='slopes', hue='condition', data=df,s=12,
    alpha=0.5,palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
# ax.set_ylim(0, 1.04)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\Delta$ F/F Slope')
ledon, ledoff = df.loc[(df.condition=='LED on'), 'slopes'].values, df.loc[(df.condition=='LED off'), 'slopes'].values
t,pval = scipy.stats.ranksums(ledon[~np.isnan(ledon)], ledoff)
ledon, ledoff = df_plt.loc[(df_plt.index.get_level_values('condition')=='LED on'), 
                'slopes'].values, df_plt.loc[(df_plt.index.get_level_values('condition')=='LED off'), 'slopes'].values
t,pval_an = scipy.stats.ttest_rel(ledon[~np.isnan(ledon)], ledoff)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax.set_title(f'Slope, before stim\n\
    per session p={pval:.3f}\n paired, per animal p={pval_an:.3f}', fontsize=12)
# plt.savefig(os.path.join(dst, 'slopes_chr2.svg'), bbox_inches='tight')

#%%
# 3 - transient trace of so vs. superficial
height=1.04
fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex=True,
                        figsize=(17,7))
supledon = []
deepledon = [] 
supledoff = []
deepledoff = []

for pln in range(4):
    if pln<3:
        trialtype = 1 # even
        supledon.append([np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_opto_all_tr) if learning_day_opto[ii]==0]).T,
                np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_opto_all_tr) if learning_day_opto[ii]==1]).T])
        trialtype=0
        supledoff.append([np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_nonopto_all_tr) if learning_day_nonopto[ii]==0]).T,
                np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_nonopto_all_tr) if learning_day_nonopto[ii]==1]).T])
    else:
        trialtype=1 # even
        deepledon.append([np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_opto_all_tr) if learning_day_opto[ii]==0]).T,
                np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_opto_all_tr) if learning_day_opto[ii]==1]).T])
        trialtype=0
        deepledoff.append([np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_nonopto_all_tr) if learning_day_nonopto[ii]==0]).T,
                np.hstack([xx[pln][trialtype] for ii,xx in \
        enumerate(day_date_dff_arr_nonopto_all_tr) if learning_day_nonopto[ii]==1]).T])

# day 1 vs. 2
ledon = [[np.vstack([xx[0] for xx in deepledon]),
        np.vstack([xx[1] for xx in deepledon])],
        [np.vstack([xx[0] for xx in supledon]),
        np.vstack([xx[1] for xx in supledon])]]
ledoff = [[np.vstack([xx[0] for xx in deepledoff]),
        np.vstack([xx[1] for xx in deepledoff])], 
        [np.vstack([xx[0] for xx in supledoff]),
        np.vstack([xx[1] for xx in supledoff])]]

colors_on = ['mediumturquoise', 'darkcyan']
colors_off = ['slategray', 'dimgray']
domain_nm = ['Deep', 'Superficial']
for domain in range(2):            
    for ld in range(2): # per learning day     
        data = ledon[domain][ld]       
        ax = axes[ld]
        ax.plot(np.nanmean(data,axis=0), 
                color=colors_on[domain],label=f'{domain_nm[domain]},LED on')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(data,axis=0)-scipy.stats.sem(data,axis=0,nan_policy='omit'),
                np.nanmean(data,axis=0)+scipy.stats.sem(data,axis=0,nan_policy='omit'), 
                alpha=0.5, color=colors_on[domain])
        ax.add_patch(
        patches.Rectangle(
            xy=(range_val/binsize,0),  # point of origin.
            width=2/binsize, height=height, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.15))

        ax.set_ylim(.97, height) 
        ax.set_xlabel('Time from CS (s)')
        trialtype = 0 # odd
        ax = axes[ld]
        data = ledoff[domain][ld]       
        ax.plot(np.nanmean(data,axis=0), 
                color=colors_off[domain],label=f'{domain_nm[domain]},LED off')

        ax.fill_between(range(0,int(range_val/binsize)*2), 
                np.nanmean(data,axis=0)-scipy.stats.sem(data,axis=0,nan_policy='omit'),
                np.nanmean(data,axis=0)+scipy.stats.sem(data,axis=0,nan_policy='omit'), 
                alpha=0.5, color=colors_off[domain])
        # trialtype = 1 # even
        # ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
        #         color='peru', label='even, 0mA')
        # ax.fill_between(range(0,int(range_val/binsize)*2), 
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
        #             alpha=0.5, color='peru')

        if ld==1: ax.legend(bbox_to_anchor=(1, 1.05))
        # else: ax.get_legend().set_visible(False)
        ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
        ax.set_xticklabels(np.arange(-range_val, range_val+1, 4))
        ax.set_title(f'Day {ld+1}')
        ax.set_ylim(.97, height)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.tight_layout()

# plt.savefig(os.path.join(dst, 'chr2_every10trials_peri_cs_summary.svg'), bbox_inches='tight')

#%%
# 4-combine days and split by trials
# transient trace of so
# per trial
height=.02
ymin=-.01
fig, ax = plt.subplots(figsize=(9,5))
pln=3
trialtype = 0# odd bc red laser
stimsec=4
opto_all_trial = np.hstack([xx[pln][trialtype] for ii,xx in \
    enumerate(day_date_dff_arr_opto_all_tr)]).T
# subset of trials
# color part of trace turqoise, the other part grey
# normalize to before 1 s
pre_window = 0 #s
opto_all_trial = np.array([xx-np.nanmean(xx[int((range_val/binsize)-(pre_window/binsize)):int((range_val/binsize)+(1/binsize))]) for xx in opto_all_trial])
opto_all_trial_ledon = np.ones_like(opto_all_trial)*np.nan
opto_all_trial_ledoff = opto_all_trial
rng1, rng2 = int(range_val/binsize), int(range_val/binsize+stimsec/binsize)+1
opto_all_trial_ledon[:,rng1:rng2]=opto_all_trial[:,rng1:rng2]
opto_all_trial_ledoff[:,rng1+1:rng2-1]=np.nan
ax.plot(np.nanmean(opto_all_trial_ledon,axis=0), 
        color='crimson',label='LED on', linewidth=4)
ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(opto_all_trial_ledon,axis=0)-scipy.stats.sem(opto_all_trial_ledon,
                                    axis=0,nan_policy='omit'),
            np.nanmean(opto_all_trial_ledon,axis=0)+scipy.stats.sem(opto_all_trial_ledon,
                                            axis=0,nan_policy='omit'), 
            alpha=0.3, color='crimson')
# off period
ax.plot(np.nanmean(opto_all_trial_ledoff,axis=0), 
        color='slategray',label='LED on', linewidth=4)
ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(opto_all_trial_ledoff,axis=0)-scipy.stats.sem(opto_all_trial_ledoff,axis=0,nan_policy='omit'),
            np.nanmean(opto_all_trial_ledoff,axis=0)+scipy.stats.sem(opto_all_trial_ledoff,axis=0,nan_policy='omit'), 
            alpha=0.3, color='slategray')

ax.add_patch(
patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height-ymin, linewidth=1, # width is s
    color='lightcoral', alpha=0.15))
ax.set_ylim(ymin, height) 
ax.set_xlabel('Time from Conditioned Stimulus (s)')
ax.set_ylabel('$\Delta$ F/F')
trialtype = 0 # odd
nonopto_all_trial = np.hstack([xx[pln][trialtype] for ii,xx in enumerate(day_date_dff_arr_nonopto_all_tr)]).T
# subset of trials
# normalize to before 1 s
pre_window = 0 #s
nonopto_all_trial = np.array([xx-np.nanmean(xx[int((range_val/binsize)-(pre_window/binsize)):int((range_val/binsize)+(1/binsize))]) for xx in nonopto_all_trial])
ax.plot(np.nanmean(nonopto_all_trial,axis=0), 
        color='slategray',label='LED off', linewidth=4)
ax.fill_between(range(0,int(range_val/binsize)*2), 
            np.nanmean(nonopto_all_trial,axis=0)-scipy.stats.sem(nonopto_all_trial,axis=0,nan_policy='omit'),
            np.nanmean(nonopto_all_trial,axis=0)+scipy.stats.sem(nonopto_all_trial,axis=0,nan_policy='omit'), 
            alpha=0.3, color='slategray')
# trialtype = 1 # even
# ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
#         color='peru', label='even, 0mA')
# ax.fill_between(range(0,int(range_val/binsize)*2), 
#             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
#             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
#             alpha=0.5, color='peru')

ax.legend(bbox_to_anchor=(1.1, 1.05))
# ax.get_legend().set_visible(False)
ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,5))
ax.set_xticklabels(np.arange(-range_val, range_val+1, 1))
ax.set_ylim(ymin, height)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# fig.suptitle('Basal dendrite layer (stratum oriens)')
fig.tight_layout()
plt.savefig(os.path.join(dst, 'halo_every10trials_peri_cs_summary.svg'), bbox_inches='tight')

#%%
# all planes
fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex=True,
                        figsize=(12,10))
for pln in range(4):
    for ld in range(2): # per learning day
        trialtype = 0 # opto trials
        ax = axes[pln,ld]
        ax.plot(np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0), 
                color='mediumturquoise', label='LEDon')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='mediumturquoise')
        trialtype = 1 # even
        ax.plot(np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0), 
                color='k',label='LEDoff')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='k')
        ax.legend()
        ax.add_patch(
        patches.Rectangle(
            xy=(range_val/binsize,0),  # point of origin.
            width=2/binsize, height=height, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))
        ax.set_ylim(.97, height)
        if pln==3: ax.set_xlabel('Time from CS (s)')
        trialtype = 0 # odd
        ax = axes[pln,ld]
        ax.plot(np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0), 
                color='peru', label='odd, 0mA')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='peru')
        # trialtype = 1 # even
        # ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
        #         color='peru', label='even, 0mA')
        # ax.fill_between(range(0,int(range_val/binsize)*2), 
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
        #             alpha=0.5, color='peru')

        if pln==0: ax.legend(bbox_to_anchor=(1.1, 1.05))
        else: ax.get_legend().set_visible(False)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        if pln==0: ax.set_title(f'Learning day {ld+1}\nPlane {planelut[pln]}')
        else: ax.set_title(f'Plane {planelut[pln]}')
        ax.set_ylim(.97, height)
        if pln==3: ax.set_xlabel('Time from CS (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('ChR2 per day per + mouse averages')
pdf.savefig(fig)
pdf.close()
# plt.savefig(os.path.join(dst, 'chr2_every10trials_peri_cs_summary.svg'), bbox_inches='tight')