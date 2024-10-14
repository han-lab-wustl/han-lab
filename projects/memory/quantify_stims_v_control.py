"""zahra
sept 2024
opn3/halo power tests
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from behavior import consecutive_stretch
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
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'Z:\opn3_grabda'
range_val = 10; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_excel(r'Z:\opn3_grabda\opn3_key.xlsx',sheet_name='opn3')
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = conddf.loc[((conddf.animal==animal) & (conddf.artifact!=True)), 'day'].values.astype(int)    
    if animal=='e219': src = r'Y:\opto_control_grabda_2m'
    else: src = r'Z:\opn3_grabda'
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        plndff = []
        for path in Path(os.path.join(src, animal, str(day))).rglob('params.mat'):
            params = scipy.io.loadmat(path)
            VR = params['VR'][0][0]; gainf = VR[14][0][0]             
            timedFF = np.hstack(params['timedFF'])
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 6 - roibasemean3/average
            # raw in row 7
            row = 6
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmedian(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])            
            # nan out stims
            # dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')
            # plt.legend()
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)
            # get off plane stim
            offpln=pln+1 if pln<3 else pln-1
            startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
            min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            # remove rewarded stims
            cs=params['solenoid2'][0]
            # cs within 50 frames of start of stim - remove
            framelim=20
            unrewstimidx = [idx for idx in min_iind if sum(cs[idx-framelim:idx+framelim])==0]            
            startofstims = np.zeros_like(dff)
            startofstims[unrewstimidx]=1

            fig,ax=plt.subplots()
            ax.plot(dff,label=f'plane: {pln}')
            ax.plot(startofstims)
            ax.set_ylim([.9,1.1])
            ax.legend()
            # peri stim binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            fig, ax = plt.subplots()
            ax.plot(meanrewdFF, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            color='k',alpha=0.4)
            ymin=min(meanrewdFF)-.02
            ymax=max(meanrewdFF)+.02-ymin
            ax.add_patch(
                patches.Rectangle(
            xy=(range_val/binsize,ymin),  # point of origin.
            width=5/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim, {animal}, day {day}, plane {pln}')
            plndff.append(rewdFF)
            plt.show()
    
        day_date_dff[f'{animal}_{day}'] = plndff

#%%

# power tests quantification
condition_org = [[conddf.loc[((conddf.animal==an) & (conddf.artifact!=True) & \
    (conddf.day==dy)), 'optopower'].values.astype(int)[0] for dy in \
        conddf.loc[((conddf.animal==an) & (conddf.artifact!=True)), 
    'day'].values.astype(int)] for an in animals]
condition_org=list(np.concatenate(condition_org)[:-3])
condition_col = {280:'k', 200:'darkcyan',80:'slategray'}
# settings
stimsec = 5 # stim duration (s)
ymin=-0.04
ymax=0.03
height=ymax-(ymin)
planes=4
# subtract ctrl
ctrl_mean_trace_per_pln=[] # 200 ma
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    for dy,v in day_date_dff.items():
        if 'e219' in dy:
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[15:22]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[15:22]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin])
            else: idx_to_catch.append(ii)
    meanrewdFF = np.nanmean(np.vstack([x[0] for x in condition_dff]),axis=0)
    ctrl_mean_trace_per_pln.append(meanrewdFF)

# assumes 4 planes
fig, axes = plt.subplots(nrows=4, figsize=(4,6), sharex=True)
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    condition = condition_org.copy() # custom condition
    for dy,v in day_date_dff.items():
        if 'e219' not in dy:
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[15:22]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[15:22]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin])
            else: idx_to_catch.append(ii)
            ii+=1
    print(len(condition))
    condition = [xx for ii,xx in enumerate(condition) if ii not in idx_to_catch]

    ax = axes[pln]
    meanrewdFF = np.vstack([x[0] for x in condition_dff])
    rewdFF = [x[1] for x in condition_dff]
    # plot per condition
    for cond in np.array([200,280]):
        meancond = np.nanmean(meanrewdFF[condition==cond],axis=0)-ctrl_mean_trace_per_pln[pln]
        trialcond = np.concatenate([[condition[ii]]*xx.shape[1] for ii,xx in enumerate(rewdFF)])
        rewcond = np.hstack(rewdFF).T[trialcond==cond].T
        rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewcond.T]).T
        ax.plot(meancond, label=f'{cond}_trials{rewcond.shape[1]}', color=condition_col[cond])   
        xmin,xmax = ax.get_xlim()         
        ax.fill_between(range(0,int(range_val/binsize)*2), 
        meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.4,color=condition_col[cond])        
    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    ii+=1
    ax.set_title(f'\nPlane {planelut[pln]}')
    ax.set_ylim([ymin,ymax])
    ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
fig.suptitle(f'All animals (n=3), per condition \n\
    Subtracted from GRABDA 2m (200mA)')
fig.tight_layout()
#%%
# test
plndff =[]
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    condition = condition_org.copy() # custom condition
    for dy,v in day_date_dff.items():
        if 'e219' not in dy:
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[15:22]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[15:22]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin])
            else: idx_to_catch.append(ii)
            ii+=1    
    condition = [xx for ii,xx in enumerate(condition) if ii not in idx_to_catch]

    meanrewdFF = np.vstack([x[0] for x in condition_dff])
    rewdFF = [x[1] for x in condition_dff]
    # plot per condition
    save = []
    for cond in np.array([200,280]):
        meancond = np.nanmean(meanrewdFF[condition==cond],axis=0)-ctrl_mean_trace_per_pln[pln]
        trialcond = np.concatenate([[condition[ii]]*xx.shape[1] for ii,xx in enumerate(rewdFF)])
        rewcond = np.hstack(rewdFF).T[trialcond==cond].T
        rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewcond.T]).T
        t,pval = scipy.stats.ttest_1samp(np.nanmean(rewcond[50:75],axis=0),popmean=0)
        print(f'Plane {pln}, condition {cond}mA, P-value={pval:.5f}')
        save.append((np.nanmean(rewcond[50:75],axis=0),cond,pval))
    plndff.append(save)
#%%
plt.rc('font', size=25)
dfs = []
conds = [200,280]
for pln in range(planes):
    df = pd.DataFrame()
    df['mean_dff_during_stim'] = np.concatenate([xx[0] for xx in plndff[pln]])
    pval=[xx[2] for xx in plndff[pln]]
    df['pval']=np.concatenate([[pval[ii]]*len(xx[0]) for ii,xx in enumerate(plndff[pln])])
    df['condition'] = np.concatenate([[conds[ii]]*len(xx[0]) for ii,xx in enumerate(plndff[pln])])
    df['plane'] = np.concatenate([[planelut[pln]]*len(df)])
    df['plane_subgroup'] = np.concatenate([['superficial' if pln<3 else 'deep']*len(df)])
    dfs.append(df)
bigdf = pd.concat(dfs)

import seaborn as sns

fig,ax = plt.subplots(figsize=(6,5))
cmap = ['darkcyan', 'k']
g=sns.barplot(x='plane',y='mean_dff_during_stim',hue='condition',data=bigdf,fill=False,
            errorbar='se',palette=cmap,ax=ax,linewidth=3,err_kws={'linewidth': 3})
sns.stripplot(x='plane',y='mean_dff_during_stim',hue='condition',data=bigdf,s=11,
        alpha=0.7,palette=cmap,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

pval200 = bigdf.loc[bigdf.condition==200,'pval'].unique()
pval280 = bigdf.loc[bigdf.condition==280,'pval'].unique()
y=0.060
fs=14
for i in range(len(pval200)):
    ax.text(i, y, f'200mA, \np={pval200[i]:.4f}', ha='center', fontsize=fs)
y=0.075
for i in range(len(pval280)):
    ax.text(i, y, f'280mA, \np={pval280[i]:.6f}', ha='center', fontsize=fs)
    
ax.set_title('Per trial quantification n=3 animals',pad=90)
#%%
# by plane subgroup

fig,ax = plt.subplots(figsize=(3,5))
cmap = ['darkcyan', 'k']
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='condition',data=bigdf,fill=False,
            errorbar='se',palette=cmap,ax=ax,linewidth=3,err_kws={'linewidth': 3})
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='condition',data=bigdf,s=11,
        alpha=0.5,palette=cmap,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

x1=bigdf.loc[((bigdf.condition==200) & (bigdf.plane_subgroup=='superficial')), 
        'mean_dff_during_stim'].values
t,pval200_sup=scipy.stats.ttest_1samp(x1,popmean=0)
x1=bigdf.loc[((bigdf.condition==280) & (bigdf.plane_subgroup=='superficial')), 
            'mean_dff_during_stim'].values
t,pval280_sup=scipy.stats.ttest_1samp(x1,popmean=0)
x1=bigdf.loc[((bigdf.condition==200) & (bigdf.plane_subgroup=='deep')), 
        'mean_dff_during_stim'].values
t,pval200_deep=scipy.stats.ttest_1samp(x1,popmean=0)
x1=bigdf.loc[((bigdf.condition==280) & (bigdf.plane_subgroup=='deep')), 
        'mean_dff_during_stim'].values
t,pval280_deep=scipy.stats.ttest_1samp(x1,popmean=0)

pval200=[pval200_sup, pval200_deep]
pval280=[pval280_sup, pval280_deep]
y=0.060
fs=14
for i in range(len(pval200)):
    ax.text(i, y, f'200mA, \np={pval200[i]:.4f}', ha='center', fontsize=fs)
y=0.075
for i in range(len(pval280)):
    ax.text(i, y, f'280mA, \np={pval280[i]:.6f}', ha='center', fontsize=fs)
    
ax.set_title('Per trial quantification n=3 animals',pad=90)
# fig.tight_layout()