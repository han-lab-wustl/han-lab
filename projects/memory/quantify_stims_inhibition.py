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
from behavior import get_success_failure_trials, consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from dopamine import get_rewzones

# plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"halo_opto"))

src = r"Z:\halo_grabda"
animals = ['e243']
# first batch
# days_all = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
#         [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]]
days_all = [[1]]
# days_all = [[40,41,42,43,44,45,46,47,48,49,51,52,53],[82,83,84,85,86,87,88,89,90,91,93,94,95]]

range_val = 8; binsize=0.2 #s
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
#%%
# optodays = [18, 19, 22, 23, 24]
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*ZD_000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
            params = scipy.io.loadmat(path)
            VR = params['VR'][0][0]; gainf = VR[14][0][0]             
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 6 - roibasemean3/average
            # raw in row 7
            row = 7
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])
            # row = 8
            # dff = np.hstack(params['params'][0][0][row])/np.nanmean(np.hstack(params['params'][0][0][row]))#/np.hstack(params['params'][0][0][9])
            
            # ax.plot(dff)
            timedFF = np.hstack(params['timedFF'])
            # nan out stims
            # dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')
            # plt.legend()
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)
            offpln=pln+1 if pln<3 else pln-1
            startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
            min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            startofstims = np.zeros_like(dff)
            startofstims[min_iind]=1
            fig,ax=plt.subplots()
            ax.plot(dff,label=f'plane: {pln}')
            ax.plot(startofstims)
            ax.legend()

            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, timedFF, 
                                        range_val, binsize)
            # Find the rows that contain NaNs
            fig, ax = plt.subplots()
            ax.plot(meanrewdFF, color = 'slategray')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            color='slategray',alpha=0.4)
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))

            ax.set_title(f'Peri-stim, halo, 280mA, plane {pln}')
    
        # day_date_dff[str(day)] = plndff

#%%
plt.rc('font', size=20)          # controls default text sizes
# plot mean and sem of opto days vs. control days
# on same plane
# 1 - set conditions
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
opto_condition = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
            opto_cond].values for ii,animal in enumerate(animals)])
animal = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
            'Animal'].values for ii,animal in enumerate(animals)])
opto_condition = np.array([True if xx==1 else False for xx in opto_condition])
day_date_dff_arr = np.array([v for k,v in day_date_dff.items()])
day_date_dff_arr_opto = day_date_dff_arr[opto_condition]
animal_opto = animal[opto_condition]
animal_nonopto = animal[~opto_condition]
day_date_dff_arr_nonopto = day_date_dff_arr[~opto_condition]
learning_day = np.concatenate([condrewloc.loc[((condrewloc.Animal==an)&(condrewloc.Day.isin(days_all[ii]))), 'learning_date'].values-1 for ii,an in enumerate(animals)])
rewzone_learning = np.concatenate([get_rewzones(condrewloc.loc[((condrewloc.Animal==an)&(condrewloc.Day.isin(days_all[ii]))), 'RewLoc'].values, 1/gainf) for ii,an in enumerate(animals)])
learning_day_opto = learning_day[opto_condition]
learning_day_nonopto = learning_day[~opto_condition]
rewzone_learning_opto = rewzone_learning[opto_condition]
rewzone_learning_nonopto = rewzone_learning[~opto_condition]
height = 1.035 # ylim
#%%
# 2 -quantify so transients
# get time period around stim
time_rng = range(int(range_val/binsize-0/binsize),
            int(range_val/binsize+(2/binsize))) # during and after stim
before_time_rng = range(int(range_val/binsize-1/binsize),
            int(range_val/binsize-0/binsize)) # during and after stim

# normalize pre-window to 1
# remember than here we only take led off trials bc of artifact
so_transients_opto = [day_date_dff_arr_opto[ii,3,1,:]/np.nanmean(day_date_dff_arr_opto[ii,3,1,:int(range_val/binsize)]) for ii,xx in enumerate(range(day_date_dff_arr_opto.shape[0]))]
so_transients_opto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_opto]
so_transients_nonopto = [day_date_dff_arr_nonopto[ii,3,0,:]/np.nanmean(day_date_dff_arr_nonopto[ii,
                    3,0,:int(range_val/binsize)]) for ii,xx in enumerate(range(day_date_dff_arr_nonopto.shape[0]))]
so_transients_nonopto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_nonopto]
fig, ax = plt.subplots(figsize=(2.2,5))
df = pd.DataFrame(np.concatenate([so_transients_opto,so_transients_nonopto])-1,columns=['so_transient_dff_difference'])
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
# ax.set_ylim(0, 1.04)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('$\Delta$ F/F-baseline (stratum oriens)')
ledon, ledoff = df.loc[(df.condition=='LED on'), 'so_transient_dff_difference'].values, df.loc[(df.condition=='LED off'), 'so_transient_dff_difference'].values
t,pval = scipy.stats.ranksums(ledon[~np.isnan(ledon)], ledoff)
ledon, ledoff = df_plt.loc[(df_plt.index.get_level_values('condition')=='LED on'), 
                'so_transient_dff_difference'].values, df_plt.loc[(df_plt.index.get_level_values('condition')=='LED off'), 'so_transient_dff_difference'].values
t,pval_an = scipy.stats.ttest_rel(ledon[~np.isnan(ledon)], ledoff)

ax.set_title(f'Stim at reward\n\
    per session p={pval:.4f}\n per animal p={pval_an:.4f}')

plt.savefig(os.path.join(dst, 'so_transient_quant.svg'), bbox_inches='tight')

#%%
# transient trace of so
height=1.04
fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex=True,
                        figsize=(13,5))
pln=3
for ld in range(2): # per learning day
        trialtype = 0 # opto trials
        ax = axes[ld]
       
        trialtype = 1 # even
        ax.plot(np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0), 
                color='mediumturquoise',label='LED on')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='mediumturquoise')
        ax.add_patch(
        patches.Rectangle(
            xy=(range_val/binsize,0),  # point of origin.
            width=2/binsize, height=height, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.15))
        ax.set_ylim(.97, height) 
        ax.set_xlabel('Time from CS (s)')
        trialtype = 0 # odd
        ax = axes[ld]
        ax.plot(np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0), 
                color='k', label='LED off')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.3, color='k')
        # trialtype = 1 # even
        # ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
        #         color='peru', label='even, 0mA')
        # ax.fill_between(range(0,int(range_val/binsize)*2), 
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
        #             alpha=0.5, color='peru')

        if ld==1: ax.legend(bbox_to_anchor=(1.1, 1.05))
        # else: ax.get_legend().set_visible(False)
        ax.set_xticks(np.arange(0, (int(range_val/binsize)*2)+1,20))
        ax.set_xticklabels(np.arange(-range_val, range_val+1, 4))
        ax.set_title(f'Day {ld+1}')
        ax.set_ylim(.97, height)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('Basal dendrite layer (stratum oriens)')
fig.tight_layout()
plt.savefig(os.path.join(dst, 'chr2_every10trials_peri_cs_summary.svg'), bbox_inches='tight')

#%%
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