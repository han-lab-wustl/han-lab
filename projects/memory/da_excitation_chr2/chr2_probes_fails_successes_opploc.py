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
# formatting for figs
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from projects.memory.dopamine import extract_vars, get_rewzones
# plt.rc('font', size=12)          # controls default text sizes
plt.close('all')
# save to pdf
dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
    f"chr2_opto_peri_analysis_opploc.pdf"))
# read condition
condrewloc = pd.read_csv(r"Z:\condition_df\chr2_grabda.csv", index_col = None)
src = r"Z:\chr2_grabda"
animals = ['e231', 'e232']
# first batch
days_all = [[40,41,42,43,44,45,46,47,48,49,50,51,52,53],
            [82,83,84,85,86,87,88,89,90,91,92,93,94,95]]
# days_all = [[40,41,43,44,46,47],
#             [82,83,86,87,88,89]]
numtrialsstim=10 # every 10 trials stim w 1 trial off
range_val = 8; binsize=0.2 # peri stimulus before/after in s
opto_cond = 'Opto_opp_loc' # experiment condition
day_date_dff = {}
day_date_dff_stim = {}
planelut = {0: 'SLM', 1: 'SR', 2: 'SP', 3: 'SO'}
rolling_win=5
reward_var='cs'
for ii,animal in enumerate(animals):
    days = days_all[ii]    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # set conditions
        newrewloc = condrewloc.loc[((condrewloc.Day==str(day))&(condrewloc.Animal==animal)), 'RewLoc'].values[0]
        rewloc = float(condrewloc.loc[((condrewloc.Day==str(day))&(condrewloc.Animal==animal)), 'PrevRewLoc'].values[0])
        plndff = []; stimdff = []
        optoday = (condrewloc.loc[((condrewloc.Day==str(day))&(condrewloc.Animal==animal)), opto_cond].values[0])
        optoday = optoday==1
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*ZD_000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
            # get vars
            params = scipy.io.loadmat(path)
            stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
            stims = scipy.io.loadmat(stimspth)        
            if len(stims['stims']>0): stims = np.hstack(stims['stims']) # nan out stims
            else: stims = np.zeros_like(params['forwardvelALL'][0])
            VR = params['VR'][0][0]; gainf = VR[14][0][0]      
            # adjust for gain
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 7 - roibasemean3/average
            dff = np.hstack(params['params'][0][0][6][0][0])/np.nanmean(np.hstack(params['params'][0][0][6][0][0]))#/np.hstack(params['params'][0][0][9])
            # nan out stims
            dff[stims[pln::4].astype(bool)] = np.nan
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(rolling_win).mean().values)
            if reward_var=='cs':
                rewards = np.hstack(params['solenoid2'])
            else:
                rewards = np.hstack(params['rewards'])
            if dff.shape[0]<rewards.shape[0]:
                rewards = rewards[:-1]
                trialnum = np.hstack(params['trialnum'])[:-1]
                ybinned = np.hstack(params['ybinned'])[:-1]/gainf
                licks = np.hstack(params['licks'])[:-1]
                timedFF = np.hstack(params['timedFF'])[:-1]
                forwardvel = np.hstack(params['forwardvel'])[:-1]
            else:
                trialnum = np.hstack(params['trialnum'])
                ybinned = np.hstack(params['ybinned'])/gainf
                licks = np.hstack(params['licks'])
                timedFF = np.hstack(params['timedFF'])
                forwardvel = np.hstack(params['forwardvel'])

            ###################### plot behavior ######################
            fig, ax = plt.subplots()
            ax.plot(ybinned)
            ax.scatter(np.where(rewards>0)[0], ybinned[np.where(rewards>0)[0]], color = 'cyan', s=30)
            ax.scatter(np.where(licks>0)[0], ybinned[np.where(licks>0)[0]], color = 'k', 
                marker = '.', s=30)
            ax.axhline(rewloc, color = 'slategray', linestyle = '--')
            ax.axhline(newrewloc, color = 'k', linestyle = '--')
            ax.set_title(f'Animal {animal}, Day {day}, {layer}')
            fig.tight_layout()
            pdf.savefig(fig)
            ###################### peri memory rew loc,  initial probes #####################
            firstrew = np.where(rewards==1)[0][0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[(ybinned[:firstrew] >= rewloc-3) & (ybinned[:firstrew] <= rewloc+3)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(ybinned[:firstrew])
            rews_centered[min_iind]=1

            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF = eye.perireward_binned_activity(dff[:firstrew], rews_centered, timedFF[:firstrew], range_val, binsize)                                
            clean_arr = rewdFF.T
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
            
            #TODO: peri reward catch trials
            ###################### peri learning rew loc, failed trials ######################             
            # split into opto vs. non opto
            # opto
            trialnumvr = VR[8][0]
            catchtrialsnum = trialnumvr[VR[16][0].astype(bool)]
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum, rewards)

            failtr_bool = np.array([(xx in ftr_trials) and 
                    (xx not in catchtrialsnum) and (xx%numtrialsstim==0) for xx in trialnum])
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_opto, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(dff[failtr_bool],
                rews_centered, timedFF[failtr_bool], range_val, binsize)
            
            # nonopto  
            failtr_bool = np.array([(xx in ftr_trials) and 
                    (xx not in catchtrialsnum) and (xx%numtrialsstim==1) for xx in trialnum])        
            failed_trialnum = trialnum[failtr_bool]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[(ybinned[failtr_bool] >= newrewloc-5) & (ybinned[failtr_bool] <= newrewloc+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(failed_trialnum)
            rews_centered[min_iind]=1
            normmeanrewdFF_nonopto, meanrewdFF_nonopto, normrewdFF, \
                rewdFF_nonopto = eye.perireward_binned_activity(dff[failtr_bool],
                rews_centered, timedFF[failtr_bool], range_val, binsize)
            # plot
            ax = axes[1]
            ax.imshow(np.concatenate([rewdFF_opto.T, rewdFF_nonopto.T]))
            ax.axhline(rewdFF_opto.T.shape[0], color='yellow')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Failed Trials (Centered by rewloc)')
            ax = axes2[1]
            if optoday:
                colorl = 'mediumturquoise'
            else: colorl = 'slategray'
            ax.plot(meanrewdFF_opto, color = colorl)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                    meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color=colorl)
            
            ax.plot(meanrewdFF_nonopto, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color='k')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Failed Trials (Centered by rewloc)')
            
            # for opto days, plot opto trials only // vs. non opto
            if numtrialsstim==10: # if every 10th trial is non opto stim
                mask = ~(trialnum%10==0)
            else:
                mask = (trialnum%2==1)
            ###################### all subsequent rews ######################
            normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(dff[mask], rewards[mask], timedFF[mask], 
                                        range_val, binsize)            
            clean_arr_opto = rewdFF_opto.T#[~rows_with_nans]  normmeanrewdFF, meanrewdFF, normrewdFF, \
            normmeanrewdFF, meanrewdFF_nonopto, normrewdFF, rewdFF_nonopto = eye.perireward_binned_activity(dff[~mask], rewards[~mask], timedFF[~mask], 
                                    range_val, binsize)  
            clean_arr_nonopto = rewdFF_nonopto.T
            ax = axes[2]
            ax.imshow(np.concatenate([clean_arr_opto,clean_arr_nonopto]))
            ax.axhline(clean_arr_opto.shape[0], color='yellow')
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title('Successful Trials (Centered by CS)')
            ax = axes2[2]
            ax.plot(meanrewdFF_nonopto, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), alpha=0.5, color='k')        
            if optoday:
                colorl = 'mediumturquoise'
            else: colorl = 'slategray'
            ax.plot(meanrewdFF_opto, color = colorl)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
                    meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                    meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color=colorl)        
            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            fig.suptitle(f'Peri CS/Rew Loc, Animal {animal}, Day {day}, {layer}')
            fig2.suptitle(f'Mean of Trials, Peri CS/Rew Loc, Animal {animal}, Day {day}, {layer}, Opto = {optoday}')                    
            fig.tight_layout()
            fig2.tight_layout() 
            pdf.savefig(fig)
            pdf.savefig(fig2)              
            plndff.append([meanrewdFF_opto, meanrewdFF_nonopto])
            ###################### peri stim before vs. after 2s ######################
            rewsize=20 # checked in vr
            stimzone = ((newrewloc*gainf-((rewsize*gainf)/2)+90)%180)/gainf
            rews_centered = np.zeros_like(ybinned[trialnum>2])
            rews_centered[(ybinned[trialnum>2] >= stimzone-5) & (ybinned[trialnum>2] <= stimzone+5)]=1
            rews_iind = consecutive_stretch(np.where(rews_centered)[0])
            min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
            rews_centered = np.zeros_like(ybinned[trialnum>2])
            rews_centered[min_iind]=1
            range_val_stim = 5 # just for stim instances
            binsize_stim=0.2 # s
            if numtrialsstim==10: # if every 10th trial is non opto stim
                mask = ~(trialnum[trialnum>2]%10==0)
            else:
                mask = (trialnum[trialnum>2]%2==1)

            normmeanrewdFF, meanrewdFF_opto, normrewdFF, \
                rewdFF_opto = eye.perireward_binned_activity(dff[trialnum>2][mask], rews_centered[mask], 
                    timedFF[trialnum>2][mask], range_val_stim, binsize_stim)            
            clean_arr_opto = rewdFF_opto.T#[~rows_with_nans]  normmeanrewdFF, meanrewdFF, normrewdFF, \
            normmeanrewdFF, meanrewdFF_nonopto, normrewdFF, rewdFF_nonopto = eye.perireward_binned_activity(dff[trialnum>2][~mask], 
                    rews_centered[~mask], timedFF[trialnum>2][~mask], 
                    range_val_stim, binsize)  
            clean_arr_nonopto = rewdFF_nonopto.T
            fig3, axes3 = plt.subplots(nrows=2,ncols=1,sharex=True)
            ax = axes3[0]
            ax.imshow(np.concatenate([clean_arr_opto,clean_arr_nonopto]))
            ax.axhline(clean_arr_opto.shape[0], color='yellow')
            ax.set_xticks(np.arange(0, (int(range_val_stim/binsize_stim)*2)+1,5))
            ax.set_xticklabels(np.arange(-range_val_stim, range_val_stim+1, 1))
            ax = axes3[1]
            ax.plot(meanrewdFF_nonopto, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val_stim/binsize_stim)*2), 
                    meanrewdFF_nonopto-scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'),
                    meanrewdFF_nonopto+scipy.stats.sem(rewdFF_nonopto,axis=1,nan_policy='omit'), alpha=0.5, color='k')        
            if optoday:
                colorl = 'mediumturquoise'
            else: colorl = 'slategray'
            ax.plot(meanrewdFF_opto, color = colorl)   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(np.arange(0,int(range_val_stim/binsize_stim)*2), 
                    meanrewdFF_opto-scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'),
                    meanrewdFF_opto+scipy.stats.sem(rewdFF_opto,axis=1,nan_policy='omit'), 
                    alpha=0.5, color=colorl)        
            ax.set_xticks(np.arange(0, (int(range_val_stim/binsize_stim)*2)+1,5))
            ax.set_xticklabels(np.arange(-range_val_stim, range_val_stim+1, 1))
            fig3.suptitle(f'Centered at 2s stim start, Opto = {optoday}')        
            fig3.tight_layout()   
            pdf.savefig(fig3)     
            plt.close('all')
            stimdff.append([meanrewdFF_opto, meanrewdFF_nonopto])
            
        day_date_dff[str(day)] = plndff
        day_date_dff_stim[str(day)] = stimdff

#%%
plt.rc('font', size=20)          # controls default text sizes
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
day_date_dff_stim_arr = np.array([v for k,v in day_date_dff_stim.items()])
day_date_dff_stim_opto = day_date_dff_stim_arr[opto_condition]
day_date_dff_stim_nonopto = day_date_dff_stim_arr[~opto_condition]
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
time_rng = range(int(range_val/binsize),
            int(range_val/binsize+2/binsize_stim)) # during and after stim
before_time_rng = range(int(range_val/binsize-1/binsize_stim),
            int(range_val/binsize)) # during and after stim

# normalize pre-window to 1
so_transients_opto = [day_date_dff_arr_opto[ii,3,
            0,:]/np.nanmean(day_date_dff_arr_opto[ii,3,0,:int(range_val/binsize)]) for ii,xx in enumerate(range(day_date_dff_arr_opto.shape[0]))]
so_transients_opto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_opto]
so_transients_nonopto = [day_date_dff_arr_nonopto[ii,3,0,:]/np.nanmean(day_date_dff_arr_nonopto[ii,
                    3,0,:int(range_val/binsize)]) for ii,xx in enumerate(range(day_date_dff_arr_nonopto.shape[0]))]
so_transients_nonopto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_nonopto]
fig, ax = plt.subplots(figsize=(2.5,5))
df = pd.DataFrame(np.concatenate([so_transients_opto,so_transients_nonopto])-1,columns=['so_transient_dff_difference'])
df['condition'] = np.concatenate([['LED on']*len(so_transients_opto), ['LED off']*len(so_transients_nonopto)])
df['animal'] = np.concatenate([animal_opto, animal_nonopto])
df = df.sort_values('condition')
df_plt = df.groupby(['animal', 'condition']).mean(numeric_only=True)
ax = sns.barplot(x='condition', y='so_transient_dff_difference',hue='condition', data=df_plt, fill=False,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"},)
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df_plt,s=10,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df,s=8,
    alpha=0.5,palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
# ax.set_ylim(0.985, 1.04)
ax.spines[['top','right']].set_visible(False)

ledon, ledoff = df.loc[(df.condition=='LED on'), 'so_transient_dff_difference'].values, df.loc[(df.condition=='LED off'), 'so_transient_dff_difference'].values
t,pval = scipy.stats.ranksums(ledon[~np.isnan(ledon)], ledoff)
ledon, ledoff = df_plt.loc[(df_plt.index.get_level_values('condition')=='LED on'), 
                'so_transient_dff_difference'].values, df_plt.loc[(df_plt.index.get_level_values('condition')=='LED off'), 'so_transient_dff_difference'].values
t,pval_an = scipy.stats.ttest_rel(ledon[~np.isnan(ledon)], ledoff)

ax.set_title(f'Opposite rew. loc. stim at reward\n per session p={pval:.3f}\n per animal p={pval_an:.3f}')
#%%# during stim
# get time period around stim
time_rng = range(int(range_val_stim/binsize_stim-1/binsize_stim),
        int(range_val_stim/binsize_stim+.5/binsize_stim)) # during and after stim
before_time_rng = range(int(range_val_stim/binsize_stim-2/binsize_stim),
            int(range_val_stim/binsize_stim-1/binsize_stim)) # during and after stim

# normalize pre-window to 1
so_transients_opto = [day_date_dff_stim_opto[ii,3,
            0,:]/np.nanmean(day_date_dff_stim_opto[ii,3,0,:int(range_val_stim/binsize_stim)]) for ii,xx in enumerate(range(day_date_dff_stim_opto.shape[0]))]
so_transients_opto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_opto]
so_transients_nonopto = [day_date_dff_stim_nonopto[ii,3,0,:]/np.nanmean(day_date_dff_stim_nonopto[ii,
                    3,0,:int(range_val_stim/binsize_stim)]) for ii,xx in enumerate(range(day_date_dff_stim_nonopto.shape[0]))]
so_transients_nonopto = [np.nanmax(xx[time_rng])/np.nanmean(xx[before_time_rng]) for xx in so_transients_nonopto]

fig, ax = plt.subplots(figsize=(2.5,5))
df = pd.DataFrame(np.concatenate([so_transients_opto,so_transients_nonopto])-1,columns=['so_transient_dff_difference'])
df['condition'] = np.concatenate([['LED on']*len(so_transients_opto), ['LED off']*len(so_transients_nonopto)])
df['animal'] = np.concatenate([animal_opto, animal_nonopto])

df = df.sort_values('condition')
df_plt = df.groupby(['animal', 'condition']).mean(numeric_only=True)
ax = sns.barplot(x='condition', y='so_transient_dff_difference',hue='condition', data=df_plt, fill=False,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df_plt,s=10,
    palette={'LED off': "slategray", 'LED on': "mediumturquoise"})
ax = sns.stripplot(x='condition', y='so_transient_dff_difference', hue='condition', data=df,s=8,
    alpha=0.5,palette={'LED off': "slategray", 'LED on': "mediumturquoise"})

ax.spines[['top','right']].set_visible(False)
ledon, ledoff = df.loc[(df.condition=='LED on'), 'so_transient_dff_difference'].values, df.loc[(df.condition=='LED off'), 
                        'so_transient_dff_difference'].values
t,pval = scipy.stats.ranksums(ledon[~np.isnan(ledon)], ledoff)
ledon, ledoff = df_plt.loc[(df_plt.index.get_level_values('condition')=='LED on'), 
                'so_transient_dff_difference'].values, df_plt.loc[(df_plt.index.get_level_values('condition')=='LED off'), 'so_transient_dff_difference'].values
t,pval_an = scipy.stats.ttest_rel(ledon[~np.isnan(ledon)], ledoff)

ax.set_title(f'Opposite rew. loc. stim\n per session p={pval:.3f}\n per animal p={pval_an:.3f}')
#%%
# plot peri rew mean and sem of opto days vs. control days
# learning 1 vs. 2

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
#%%
# plot peri stim mean and sem of opto days vs. control days
# learning 1 vs. 2
rz = 3
fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex=True,
                        figsize=(12,10))
ymin, ymax= .97, 1.04
for pln in range(4):
    for ld in range(2): # per learning day
        trialtype = 0 # opto trials
        ax = axes[pln,ld]
        dff_plot=day_date_dff_stim_opto[((learning_day_opto==ld)&(rewzone_learning_opto==rz)),pln,trialtype,:]
        ax.plot(np.nanmean(dff_plot,axis=0), 
                color='mediumturquoise', label='LEDon')
        ax.fill_between(range(0,int(range_val_stim/binsize_stim)*2), 
            np.nanmean(dff_plot,axis=0)-scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'),
            np.nanmean(dff_plot,axis=0)+scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'), 
            alpha=0.5, color='mediumturquoise')
        trialtype = 1 # even
        dff_plot = day_date_dff_stim_nonopto[((learning_day_nonopto==ld)&(rewzone_learning_nonopto==rz)),pln,trialtype,:]
        ax.plot(np.nanmean(dff_plot,axis=0), 
                color='k',label='LEDoff')
        ax.fill_between(range(0,int(range_val_stim/binsize_stim)*2), 
            np.nanmean(dff_plot,axis=0)-scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'),
            np.nanmean(dff_plot,axis=0)+scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'), 
            alpha=0.5, color='k')            
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.add_patch(
        patches.Rectangle(
            xy=(range_val_stim/binsize_stim,0),  # point of origin.
            width=2/binsize_stim, height=height, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))
        ax.set_ylim(ymin, ymax)
        if pln==3: ax.set_xlabel('Time from stim (s)')
        trialtype = 0 # odd
        dff_plot = day_date_dff_stim_nonopto[((learning_day_nonopto==ld)&(rewzone_learning_nonopto==rz)),pln,trialtype,:]
        ax = axes[pln,ld]
        ax.plot(np.nanmean(dff_plot,axis=0), 
                color='peru', label='odd, 0mA')
        ax.fill_between(range(0,int(range_val_stim/binsize_stim)*2), 
            np.nanmean(dff_plot,axis=0)-scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'),
            np.nanmean(dff_plot,axis=0)+scipy.stats.sem(dff_plot,axis=0,nan_policy='omit'), 
            alpha=0.5, color='peru')
        # trialtype = 1 # even
        # ax.plot(np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0), 
        #         color='peru', label='even, 0mA')
        # ax.fill_between(range(0,int(range_val/binsize)*2), 
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)-scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'),
        #             np.nanmean(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0)+scipy.stats.sem(day_date_dff_arr_nonopto[:,pln,trialtype,:],axis=0,nan_policy='omit'), 
        #             alpha=0.5, color='peru')

        if pln==0 and ld==1: ax.legend(bbox_to_anchor=(1.1, 1.05))
        else: ax.get_legend().set_visible(False)
        ax.set_xticks(np.arange(0, (int(range_val_stim/binsize_stim)*2)+1,5))
        ax.set_xticklabels(np.arange(-range_val_stim, range_val_stim+1, 1))
        if pln==0: ax.set_title(f'Learning day {ld+1}\nPlane {planelut[pln]}')
        else: ax.set_title(f'Plane {planelut[pln]}')
        ax.set_ylim(ymin, ymax)
        if pln==3: ax.set_xlabel('Time from stim (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
fig.suptitle(f'ChR2 per day per + mouse averages of opposite location stim, rewzone {rz}')
pdf.savefig(fig)
pdf.close()
# plt.savefig(os.path.join(dst, 'chr2_every10trials_peri_cs_summary.svg'), bbox_inches='tight')
