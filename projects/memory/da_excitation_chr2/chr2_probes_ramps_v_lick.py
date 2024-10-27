"""zahra's dopamine hrz analysis
june 2024
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
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches

# plt.rc('font', size=12)          # controls default text sizes
plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"chr2_opto_probe_analysis.pdf"))

condrewloc = pd.read_csv(r"Z:\condition_df\chr2_grabda.csv", index_col = None)
src = r"Z:\chr2_grabda"
animals = ['e231', 'e232']
# first batch
# days_all = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
#         [44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]]
days_all = [[28,29,30,31,32,33,34,35,36],
    [70,71,72,73,74,75,76,77,78]]
numtrialsstim=10
range_val = 5; binsize=0.2
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}

day_date_dff = {}
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        newrewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
        rewloc = condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0]
        plndff = []
        optoday = (condrewloc.loc[((condrewloc.Day==day)&(condrewloc.Animal==animal)), 'Opto'].values[0])
        optoday = optoday==1
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
            # dff is in row 7 - roibasemean3/average
            dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
            # fig, ax = plt.subplots()
            # ax.plot(dff)
            # nan out stims
            dff[stims[pln::4].astype(bool)] = np.nan
            # ax.plot(dff)
            # # temp remove artifacts
            # mean = np.mean(dff)
            # std = np.std(dff)
            # z_scores = np.abs((dff - mean) / std)
            # if pln==1:
            #     artifact_threshold = np.std(z_scores)*2
            # else:
            #     artifact_threshold = np.std(z_scores)*3    
            # if optoday:
            #     mean = np.mean(dff)
            #     std = np.std(dff)
            #     z_scores = np.abs((dff - mean) / std)
            #     artifact_mask = z_scores > artifact_threshold
            #     # Remove artifacts by setting them to NaN
            #     clean_data = dff.copy()
            #     clean_data[artifact_mask] = np.nan
            #     ax.plot(clean_data)
            #     ax.set_ylim(0.9,1.1)
            #     dff = clean_data
            # plt.close(fig)
            # get z score
            # dff = (dff-np.nanmean(dff))/np.nanstd(dff)
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)            
            rewards = np.hstack(params['solenoid2'])
            trialnum = np.hstack(params['trialnum'])
            ybinned = np.hstack(params['ybinned'])/gainf
            licks = np.hstack(params['licks'])
            timedFF = np.hstack(params['timedFF'])
            # mask out dark time
            dff = dff[ybinned>3]
            rewards = rewards[ybinned>3]
            trialnum = trialnum[ybinned>3]
            licks = licks[ybinned>3]
            timedFF = timedFF[ybinned>3]
            ybinned = ybinned[ybinned>3]
            # plot pre-first reward dop activity    
            firstrew = np.where(rewards==1)[0][0]
            # or only probes
            # firstrew = np.where(trialnum==1)[0][0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            # center by prev rew loc
            rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[min_iind]=1
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, 
                    timedFF[:firstrew], range_val, binsize)
            # peri reward initial probes        
            # Find the rows that contain NaNs
            # rows_with_nans = np.any(np.isnan(rewdFF.T), axis=1)
            # Select rows that do not contain any NaNs
            clean_arr = rewdFF.T#[~rows_with_nans]    
            fig, axes = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes[0]
            ax.imshow(clean_arr)
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Probe Trials (Centered by prev. rewloc)')
            fig2, axes2 = plt.subplots(nrows=3,ncols=1,sharex=True)#,gridspec_kw={'width_ratios':[4,1]})
            ax = axes2[0]
            ax.plot(meanrewdFF)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
                    meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'), alpha=0.5)                
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Probe Trials (Centered by prev. rewloc)')
            # # center by old rew zone
            # rews_centered = np.zeros_like(ybinned)
            # rews_centered[(ybinned > rewloc-2) & (ybinned < rewloc+2)]=1
            # rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            # min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            # rews_centered = np.zeros_like(ybinned)
            # rews_centered[min_iind]=1
            
            plt.close('all')
            plndff.append([meanrewdFF])
        day_date_dff[str(day)] = plndff


#%%
plt.rc('font', size=20)# controls default text sizes
# plot mean and sem of opto days vs. control days
# on same plane
opto_condition = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
            'Opto_memory_day'].values for ii,animal in enumerate(animals)])
opto_condition = np.array([True if xx==1 else False for xx in opto_condition])
day_date_dff_arr = np.array([v for k,v in day_date_dff.items()])
day_date_dff_arr_opto = day_date_dff_arr[opto_condition]
day_date_dff_arr_nonopto = day_date_dff_arr[~opto_condition]
learning_day = np.concatenate([condrewloc.loc[((condrewloc.Animal==an)&(condrewloc.Day.isin(days_all[ii]))), 'learning_date'].values-1 for ii,an in enumerate(animals)])
learning_day_opto = learning_day[opto_condition]
learning_day_nonopto = learning_day[~opto_condition]
# slope calc: pre-reward window
# slope that ramps to the previous reward loc
from scipy.stats import linregress
pln=2
window_extra = 10
slopes_dff_opto = [linregress(np.arange(0,(range_val/binsize)+window_extra),day_date_dff_arr_opto[ii,pln,:,
                        :int(range_val/binsize)+window_extra]) for ii in range(day_date_dff_arr_opto.shape[0])]
slopes_dff_opto = [xx[0] for xx in slopes_dff_opto]
slopes_dff_nonopto = [linregress(np.arange(0,(range_val/binsize)+window_extra),day_date_dff_arr_nonopto[ii,pln,:,
                        :int(range_val/binsize)+window_extra]) for ii in range(day_date_dff_arr_nonopto.shape[0])]
slopes_dff_nonopto = [xx[0] for xx in slopes_dff_nonopto]

sldf = pd.DataFrame()
fig, ax= plt.subplots(figsize=(2.5,6))
sldf['slope'] = np.concatenate([slopes_dff_opto,slopes_dff_nonopto])*100
sldf['condition'] = np.concatenate([['LEDon']*len(slopes_dff_opto),['LEDoff']*len(slopes_dff_nonopto)])
# sldf['animal'] = np.concatenate([condrewloc.loc[((condrewloc.Day.isin(days_all[ii])) & (condrewloc.Animal==animal)), 
#             'Animal'].values for ii,animal in enumerate(animals)])

# sldf = sldf.groupby(['animal','condition']).mean(numeric_only=True)

ax = sns.stripplot(x='condition', y='slope', hue='condition',data=sldf,s=12,)
ax = sns.barplot(x='condition', y='slope', hue='condition',data=sldf,fill=False)

s1 = sldf.loc[(sldf.condition=='LEDon'), 'slope'].values
s2 = sldf.loc[(sldf.condition=='LEDoff'), 'slope'].values
t,pval = scipy.stats.ranksums(s1,s2)
ax.set_title(f'p-value = {pval:.2f}')
#%%
y1 = .98
height = 1.02 # ylim
fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex=True,
                        figsize=(12,10))
for pln in range(4):
    for ld in range(2): # per learning day
        trialtype = 0 # opto trials
        ax = axes[pln,ld]
        ax.plot(np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0), 
                color='mediumturquoise', label='LEDon_previous_day')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_opto[(learning_day_opto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='mediumturquoise')
        ax.plot(np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0), 
                color='k',label='LEDoff_previous_day')
        ax.fill_between(range(0,int(range_val/binsize)*2), 
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],
                    axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'),
                    np.nanmean(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],
                    axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[(learning_day_nonopto==ld),pln,trialtype,:],axis=0,nan_policy='omit'), 
                    alpha=0.5, color='k')
        ax.legend()
        
        if pln==0: ax.legend(bbox_to_anchor=(1.1, 1.05))
        else: ax.get_legend().set_visible(False)
        ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
        ax.set_xticklabels(range(-range_val, range_val+1, 1))
        if pln==0: ax.set_title(f'Learning day {ld+1}\nPlane {planelut[pln]}')
        else: ax.set_title(f'Plane {planelut[pln]}')
        ax.set_ylim(y1, height)
        if pln==3: ax.set_xlabel('Time from center of reward loc. (s)')
        ax.spines[['top','right']].set_visible(False)
    

fig.suptitle('ChR2 per day per + mouse averages')
pdf.savefig(fig)
pdf.close()
