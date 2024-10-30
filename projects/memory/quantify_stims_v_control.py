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
range_val = 15; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_excel(r'Z:\opn3_grabda\opn3_key_zd_updated.xlsx',sheet_name='opn3')
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs=False
# animals=['e222']
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = conddf.loc[((conddf.animal==animal) & (conddf.artifact!=True)), 'day'].values.astype(int)    
    if ((animal=='e219') or (animal=='e221') or (animal=='e222')): src = r'Y:\opto_control_grabda_2m'
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
            dff = np.hstack(dffdf.rolling(10).mean().values)
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
            if show_figs==True:
                plt.show()
            else:
                plt.close('all')
    
        day_date_dff[f'{animal}_{day}'] = plndff

#%%

# power tests quantification
# condition_org = [[conddf.loc[((conddf.animal==an) & (conddf.artifact!=True) & \
#     (conddf.day==dy)), 'optopower'].values.astype(int)[0] for dy in \
#         conddf.loc[((conddf.animal==an) & (conddf.artifact!=True)), 
#     'day'].values.astype(int)] for an in animals]
# condition_org=list(np.concatenate(condition_org)[:-3])
# condition_col = {280:'k', 200:'darkcyan',80:'slategray'}
# settings
ymin=-0.04
ymax=0.03
height=ymax-(ymin)
planes=4
stimsec=5.5
norm_window=.5
# subtract ctrl
ctrl_mean_trace_per_pln=[] # 200 ma
for pln in range(planes):
    ii=0; condition_dff = []
    idx_to_catch = []
    for dy,v in day_date_dff.items():
        if (('e219' in dy) or ('e221' in dy) or ('e222' in dy)):
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                condition_dff.append([meanrewdFF, rewdFF_prewin])
            else: idx_to_catch.append(ii)
    meanrewdFF = np.nanmean(np.vstack([x[0] for x in condition_dff]),axis=0)
    ctrl_mean_trace_per_pln.append(meanrewdFF)
    

# plot deep vs. superficial
# plot control vs. drug
plt.rc('font', size=11)
# assumes 4 planes
deep_rewdff = []
sup_rewdff = []
norm_window=2 #s
for pln in range(planes):
    ii=0; 
    cond_dff = []    
    idx_to_catch = []
    
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'optocond'].values[0]!='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                cond_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
            else: idx_to_catch.append(ii)
            ii+=1

    meanrewdFF_s = np.vstack([x[0] for x in cond_dff])
    rewdFF_s = np.hstack([x[1] for x in cond_dff])
    if pln==3:
        deep_rewdff.append([rewdFF_s,np.hstack([x[2] for x in cond_dff])])
    else:
        sup_rewdff.append([rewdFF_s,np.hstack([x[2] for x in cond_dff])])

# get animals
# add all layers together
an_sup_rewdff=np.hstack([xx[1] for xx in sup_rewdff])
sup_rewdff=np.hstack([xx[0] for xx in sup_rewdff])

an_deep_rewdff=np.hstack([xx[1] for xx in deep_rewdff])
deep_rewdff=np.hstack([xx[0] for xx in deep_rewdff])

ymin=-0.03
ymax=0.03
stimsec=5.5
# plot
saline = [deep_rewdff, sup_rewdff]
lbls = ['Deep (SO)', 'Superficial (SP, SR, SLM)']
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(3.5,5), sharex=True)
for i in range(len(saline)):
    # plot
    ax=axes[i]
    # subtract controls
    meancond = np.nanmean(saline[i],axis=1)-ctrl_mean_trace_per_pln[i]
    rewcond = np.array([xx-ctrl_mean_trace_per_pln[pln] for xx in saline[i].T]).T
    ax.plot(meancond,linewidth=1.5,color='k',label='Saline')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='k')  
 
    ax.axhline(0,color='k',linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))

    ii+=1
    if i==0:
        ax.set_title(f'eOPN3-Control \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    # if i==0: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')

fig.tight_layout()
# %%
# ttest
an = [an_deep_rewdff, an_sup_rewdff]

save = []
for i in range(2): # deep vs. sup
    rewcond_h = saline[i]    
    stimdff_h = np.nanmean(rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)    
    t,pval = scipy.stats.ttest_1samp(stimdff_h, popmean=0)
    save.append([stimdff_h, pval, an[i]])    
# superficial vs. deep
deep_rewcond_h = saline[0]
sup_rewcond_h = saline[1]
deep_stimdff_h = np.nanmean(deep_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
sup_stimdff_h = np.nanmean(sup_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
t,pval_deep_vs_sup = scipy.stats.ranksums(deep_stimdff_h, sup_stimdff_h)

#%%
plt.rc('font', size=25)
lbls = ['Deep', 'Superficial']
dfs = []
for pln in range(2):
    df = pd.DataFrame()
    df['mean_dff_during_stim'] = save[pln][0]
    pval=save[pln][1]
    df['pval']=[pval]*len(df)
    df['plane_subgroup'] =lbls[pln]
    df['animal'] =  save[pln][2]
    # df['plane_subgroup'] = np.concatenate([[plnsg]*len(df)])
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf = bigdf.reset_index()
import seaborn as sns

fig,ax = plt.subplots(figsize=(2,5))
cmap = ['mediumvioletred', 'slategray']
g=sns.boxplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
        data=bigdf,fill=False,palette=cmap,
            linewidth=3)
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
        data=bigdf,s=13,palette=cmap,
        alpha=0.2,ax=g,dodge=True)
ax.axhline(0, color='k', linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

y=0.04
fs=12
i=0
for i in range(len(lbls)):
    pval = bigdf.loc[bigdf.plane_subgroup==lbls[i], 'pval'].values[0]
    ax.text(i, y, f'p={pval:.3e}', ha='center', fontsize=fs, rotation=45)
    i+=1

ax.text(i, y, f' deep vs. super\np={pval_deep_vs_sup:.7f}', ha='center', 
        fontsize=fs, rotation=45)
ax.set_title('n=trials, 3 animals',pad=100,fontsize=14)

#%%
# per animal 

bigdfan = bigdf.groupby(['animal', 'plane_subgroup']).mean(numeric_only=True)
# # # Specify the desired order
# desired_order = ['SLM', 'SR', 'SP', 'SO']

# # Convert the 'City' column to a categorical type with the specified order
# bigdfan['plane'] = pd.Categorical(bigdfan['plane'], categories=desired_order, ordered=True)

# # Sort the DataFrame by the 'City' column
# bigdfan.sort_values('plane')
cmap = ['mediumvioletred', 'slategray']

fig,ax = plt.subplots(figsize=(2,5))
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,
        errorbar='se',ax=ax,linewidth=4,err_kws={'linewidth': 4},
        palette=cmap)
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,
        s=17,alpha=0.6,ax=ax,palette=cmap,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

y=0.0035
fs=14
i=0
for i in range(len(lbls)):
    halo = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[i])), 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_1samp(halo, popmean=0)
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
    i+=1

halo_d = bigdfan.loc[(bigdfan.index.get_level_values('plane_subgroup')==lbls[0]), 'mean_dff_during_stim'].values
halo_s = bigdfan.loc[(bigdfan.index.get_level_values('plane_subgroup')==lbls[1]), 'mean_dff_during_stim'].values
t,pval = scipy.stats.ttest_rel(halo_d, halo_s)
ax.text(i, y, f'deep vs. super \n p={pval:.4f}', ha='center',
    fontsize=fs, rotation=45)

ax.set_title('n=3 animals',pad=100,fontsize=14)
