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

src = r'Y:\halo_grabda'
range_val = 8; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_excel(r'Y:\halo_grabda\halo_key.xlsx',sheet_name='halo')
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
# animals = ['e241', 'e242', 'e243']

day_date_dff = {}
for ii,animal in enumerate(animals):
    days = conddf.loc[((conddf.animal==animal)), 'day'].values.astype(int)    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        plndff = []
        fig,axes=plt.subplots(nrows=3, ncols=4, figsize=(12,6))
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

            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(8).mean().values)
            startofstims=params['optoEvent'][0]

            # plot mean img
            ax=axes[0,pln]
            ax.imshow(params['params'][0][0][0],cmap='Greys_r')
            ax.axis('off')
            ax.set_title(f'{animal}, day {day}, {planelut[pln]}')
            ax=axes[1,pln]
            ax.plot(dff-1,label=f'plane: {pln}')
            ax.plot(startofstims-1)
            ax.set_ylim([-.1,.1])
            ax.set_title(f'Stim events')

            # peri stim binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            ax=axes[2,pln]
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
            width=3/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim, {animal}, day {day}, plane {pln}')
            plndff.append(rewdFF)
        condition = conddf.loc[((conddf.animal==animal) & (conddf.day==day)), 'drug'].values[0]    

        day_date_dff[f'{animal}_{day}_{condition}'] = plndff

#%%

# quantification
# get control traces
plt.rc('font', size=8)
# settings
stimsec = 3 # stim duration (s)
ymin=-0.012
ymax=0.012
height=ymax-(ymin)
planes=4
norm_window = 3 #s

# subtract ctrl
ctrl_mean_trace_per_pln=[] 
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'condition'].values[0]=='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin])
            else: idx_to_catch.append(ii)
    meanrewdFF = np.nanmean(np.vstack([x[0] for x in condition_dff]),axis=0) # mean across days
    ctrl_mean_trace_per_pln.append(meanrewdFF)
#%%
plt.rc('font', size=11)

# assumes 4 planes
fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(6,9), sharex=True)
for pln in range(planes):
    ii=0; 
    saline_dff = []
    drug_dff = []
    idx_to_catch = []
    
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'condition'].values[0]!='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                if 'drug' in dy:
                    drug_dff.append([meanrewdFF, rewdFF_prewin, dy[:4]])
                else:
                    saline_dff.append([meanrewdFF, rewdFF_prewin, dy[:4]])
            else: idx_to_catch.append(ii)
            ii+=1

    ax = axes[pln,0]
    meanrewdFF_s = np.vstack([x[0] for x in saline_dff])
    rewdFF_s = np.hstack([x[1] for x in saline_dff])
    meanrewdFF_d = np.vstack([x[0] for x in drug_dff])
    rewdFF_d = np.hstack([x[1] for x in drug_dff])

    # plot
    meancond = np.nanmean(meanrewdFF_s,axis=0)-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewdFF_s.T]).T
    ax.plot(meancond,linewidth=1.5,color='k',label='none')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='k')  
    # also plot drug
    meancond = np.nanmean(meanrewdFF_d,axis=0)-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewdFF_d.T]).T
    ax.plot(meancond,linewidth=1.5,color='royalblue',label='drug')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='royalblue')        
      
    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.3))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.13))

    ii+=1
    ax.set_title(f'\nPlane {planelut[pln]}')
    ax.set_ylim([ymin,ymax])
    if pln==3: ax.legend()

# subtract from drug
# assumes 4 planes
for pln in range(planes):
    ii=0; 
    saline_dff = []
    drug_dff = []
    idx_to_catch = []
    
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'condition'].values[0]!='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                if 'drug' in dy:
                    drug_dff.append([meanrewdFF, rewdFF_prewin, dy[:4]])
                else:
                    saline_dff.append([meanrewdFF, rewdFF_prewin, dy[:4]])
            else: idx_to_catch.append(ii)
            ii+=1

    ax = axes[pln,1]
    meanrewdFF_s = np.vstack([x[0] for x in saline_dff])
    rewdFF_s = np.hstack([x[1] for x in saline_dff])
    meanrewdFF_d = np.vstack([x[0] for x in drug_dff])
    rewdFF_d = np.hstack([x[1] for x in drug_dff])

    # plot
    meancond_s = np.nanmean(meanrewdFF_s,axis=0)-ctrl_mean_trace_per_pln[pln]
    meancond_d = np.nanmean(meanrewdFF_d,axis=0)-ctrl_mean_trace_per_pln[pln]
    meancond = meancond_s-meancond_d
    rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewdFF_s.T]).T
    rewcond = np.array([xx for xx in rewdFF_s.T]).T
    rewcond = np.array([xx-meancond_d for xx in rewcond.T]).T

    ax.plot(meancond,linewidth=1.5,color='indigo',label='none-drug')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.4,color='indigo')  
    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.3))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.13))
    ax.axhline(0,color='k',linestyle='--')
    ii+=1
    ax.set_title(f'\nPlane {planelut[pln]}')
    ax.set_ylim([ymin,ymax])
    if pln==3: ax.legend()
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
fig.suptitle(f'Control vs. drug \ Halo-Halo+D1 anatagonist')

fig.tight_layout()
#%%
# collect values for ttest
plndff =[]
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'condition'].values[0]!='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]]) # save an per trial
            else: idx_to_catch.append(ii)
            ii+=1    

    meanrewdFF = np.vstack([x[0] for x in condition_dff])
    rewdFF = np.hstack([x[1] for x in condition_dff])
    animals = np.hstack([x[2] for x in condition_dff])
    # p-val per plane
    meancond = np.nanmean(meanrewdFF,axis=0)-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in rewdFF.T]).T
    meantest = np.nanmean(rewcond[int(range_val/binsize):int((range_val/binsize)+(stimsec/binsize))],
                            axis=0)
    t,pval = scipy.stats.ttest_1samp(meantest,popmean=0)
    print(f'Plane {pln}, P-value={pval:.5f}')
    save=(meantest,pval,animals)
    plndff.append(save)
#%%
plt.rc('font', size=25)
dfs = []
for pln in range(planes):
    df = pd.DataFrame()
    df['mean_dff_during_stim'] = plndff[pln][0]
    pval=plndff[pln][1]
    df['pval']=[pval]*len(plndff[pln][0])
    df['plane'] =[planelut[pln]]*len(df)
    df['animal'] = plndff[pln][2]    
    if (pln<3):
        plnsg = 'superficial'
    elif (pln==3):
        plnsg = 'deep'
    else:
        plnsg=np.nan
    df['plane_subgroup'] = np.concatenate([[plnsg]*len(df)])
    dfs.append(df)
bigdf = pd.concat(dfs)

import seaborn as sns

fig,ax = plt.subplots(figsize=(6,5))
cmap = ['darkcyan', 'k']
g=sns.boxplot(x='plane',y='mean_dff_during_stim',hue='plane',data=bigdf,fill=False,
            ax=ax,linewidth=3)
sns.stripplot(x='plane',y='mean_dff_during_stim',hue='plane',data=bigdf,s=11,
        alpha=0.3,ax=ax)
ax.axhline(0, color='k', linestyle='--')
ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))

y=0.02
fs=14
i=0
for k,v in planelut.items():
    pval = bigdf.loc[bigdf.plane==planelut[k], 'pval'].values[0]
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs)
    i+=1
    
ax.set_title('Per trial quantification n=3 animals',pad=60)

#%%
# per animal 

bigdfan = bigdf.groupby(['animal', 'plane']).mean(numeric_only=True)
# # # Specify the desired order
# desired_order = ['SLM', 'SR', 'SP', 'SO']

# # Convert the 'City' column to a categorical type with the specified order
# bigdfan['plane'] = pd.Categorical(bigdfan['plane'], categories=desired_order, ordered=True)

# # Sort the DataFrame by the 'City' column
# bigdfan.sort_values('plane')

fig,ax = plt.subplots(figsize=(3.5,5))
g=sns.barplot(x='plane',y='mean_dff_during_stim',hue='plane',data=bigdfan,fill=False,
            errorbar='se',ax=ax,linewidth=3,err_kws={'linewidth': 3})
sns.stripplot(x='plane',y='mean_dff_during_stim',hue='plane',data=bigdfan,s=15,
            alpha=0.8,ax=ax)
ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))

y=0.001
fs=14
i=0
for k,v in planelut.items():
    values = bigdfan.loc[bigdfan.index.get_level_values('plane')==planelut[k], 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_1samp(values,popmean=0)
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
    i+=1
    
ax.set_title('n=3 animals',pad=60)
#%%
# by plane subgroup

fig,ax = plt.subplots(figsize=(3,5))
bigdf.sort_values(['plane_subgroup'])
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
            data=bigdf,fill=False,
            errorbar='se',ax=ax,linewidth=3,err_kws={'linewidth': 3})
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
        data=bigdf,s=11,alpha=0.3,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('')
# ax.legend(bbox_to_anchor=(1.01, 1.05))

x1=bigdf.loc[((bigdf.plane_subgroup=='superficial')), 
        'mean_dff_during_stim'].values
t,pvalsup=scipy.stats.ttest_1samp(x1,popmean=0)
x2=bigdf.loc[((bigdf.plane_subgroup=='deep')), 
        'mean_dff_during_stim'].values
t,pvaldeep=scipy.stats.ttest_1samp(x2,popmean=0)

y=0.02
fs=12
pval = [pvaldeep,pvalsup]
xs = [x2,x1]
for i in range(2):    
    ax.text(i, y, f'p={pval[i]:.7f}\n n={xs[i].shape[0]} trials', ha='center', fontsize=fs)
    
ax.set_title('Per trial quantification, 3 animals',pad=60)
#%%

# per animal 

bigdfan = bigdf.groupby(['animal', 'plane_subgroup']).mean(numeric_only=True)
bigdfan = bigdfan.sort_values('plane_subgroup')

fig,ax = plt.subplots(figsize=(2,5))
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,
            errorbar='se',ax=ax,linewidth=3,err_kws={'linewidth': 3})
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,s=15,
            alpha=0.8,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('')

y=0.0001
fs=14
i=0
for layer in ['deep', 'superficial']:
    values = bigdfan.loc[bigdfan.index.get_level_values('plane_subgroup')==layer, 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_1samp(values,popmean=0)
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
    i+=1
    
ax.set_title('n=3 animals',pad=60)