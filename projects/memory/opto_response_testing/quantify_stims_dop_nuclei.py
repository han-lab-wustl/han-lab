"""zahra
dec 2024
# Can you make a “Selectivity Index” for the opto results for LC, VTA, and SNc? 
# It should be something like (deep response – super response)/(deep response + super response). Something that is completely targeted to deep should be 1 and completely superficial should be -1.  
# This should be done per mouse. Let me know if you have any questions.
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
from scipy.ndimage import label

from pathlib import Path
import matplotlib.backends.backend_pdf
import matplotlib, seaborn as sns
from projects.memory.behavior import consecutive_stretch
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.patches as patches
from projects.memory.dopamine import get_rewzones

# plt.rc('font', size=12)          # controls default text sizes
#%%
plt.close('all')

range_val = 5; binsize=0.2 #s
dur=1# s stim duration
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
prewin = 2 # normalize pre window
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
conddf = pd.read_excel(r"X:\dopamine_opto_vtalcsnc_newsnc.xlsx",sheet_name='Sheet1') # day vs. condition LUT
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs = False # show individual days peri stim plots 
rolling_win = 3
day_date_dff = {}

for ii,an in enumerate(animals):
    days = conddf.loc[((conddf.animal==an)), 'day'].values.astype(int)   

    for day in days: 
        print(f'*******Animal: {an}, Day: {day}*******\n')
        # for each plane
        src = conddf.loc[((conddf.animal==an)&((conddf.day==day))), 'src'].values[0]
        stimspth = list(Path(os.path.join(src, an, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        plndff = []
        condition = conddf.loc[((conddf.animal==an) & (conddf.day==day)), 'antagonist'].values[0]    
        nuc = conddf.loc[((conddf.animal==an) & (conddf.day==day)), 'nucleus'].values[0]    

        fig,axes=plt.subplots(nrows=3, ncols=4, figsize=(12,6))

        for path in Path(os.path.join(src, an, str(day))).rglob('params.mat'):
            params = scipy.io.loadmat(path)
            timedFF = np.hstack(params['timedFF'])
            planenum = os.path.basename(os.path.dirname(os.path.dirname(path)))
            pln = int(planenum[-1])
            layer = planelut[pln]
            params_keys = params.keys()
            keys = params['params'].dtype
            # dff is in row 6 - roibasemean3/average
            # raw in row 7
            row =  6
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])            
            # plot mean img
            ax=axes[0,pln]
            ax.imshow(params['params'][0][0][0],cmap='Greys_r')
            ax.imshow(params['params'][0][0][5][0][0],cmap="Greens",alpha=0.4)

            ax.axis('off')
            ax.set_title(f'{an}, day {day}, {planelut[pln]}')
            # nan out stims
            dff[stims[pln::4].astype(bool)] = np.nan
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(3).mean().values)
            if nuc == 'SNc': # gerardos way with old stim detect????
                # Assuming stims, utimedFF, and solenoid2ALL are defined numpy arrays
                utimedFF = params['utimedFF'][0]
                # Step 1: Label the regions in stims greater than 0.5
                abfstims, num_features = label(stims > 0.5)
                # Step 2: Loop through each feature
                for dw in range(1, num_features):
                    index_next = np.where(abfstims == (dw + 1))[0]
                    index_current = np.where(abfstims == dw)[0]
                    if len(index_next) > 0 and len(index_current) > 0:
                        time_diff = utimedFF[index_next[0]] - utimedFF[index_current[-1]]
                        
                        if time_diff < 0.5:
                            abfstims[index_current[0]: index_next[0]] = dw + 1
                # Step 3: Filter with abfstims > 0.5
                abfstims = abfstims > 0.5
                # Step 4: Find consecutive stretches
                abfrect = consecutive_stretch(np.where(abfstims)[0])
                min_iind = [int(np.floor(min(xx)/4))-2 for xx in abfrect]
            else:   
                # get off plane stim
                offpln=pln-1 if pln<3 else pln+1
                startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
                min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            # # remove rewarded stims
            cs=params['solenoid2'][0]
            # # cs within 50 frames of start of stim - remove
            framelim=20
            unrewstimidx = [idx for idx in min_iind if sum(cs[idx-framelim:idx+framelim])==0]            
            startofstims = np.zeros_like(dff)
            startofstims[unrewstimidx]=1
            
            ax=axes[1,pln]
            ax.plot(dff-1,label=f'plane: {pln}')
            ax.plot(startofstims-1)
            ax.set_ylim([-.1,.5])
            ax.set_title(f'Stim events')
            # peri stim binned activity
            normmeanrewdFF, meanrewdFF, normrewdFF, \
                rewdFF= eye.perireward_binned_activity(dff, startofstims, 
                    timedFF, range_val, binsize)
            
            binss = np.ceil(prewin/binsize).astype(int)
            bound = int(range_val/binsize)
            #normalize
            meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[(bound-binss):bound])
            rewdFF = np.array([rewdFF[:,tr]-np.nanmean(meanrewdFF[(bound-binss):bound]) \
                for tr in range(rewdFF.shape[1])]).T

            ax=axes[2,pln]
            ax.plot(meanrewdFF, color = 'k')   
            xmin,xmax = ax.get_xlim()     
            ax.fill_between(range(0,int(range_val/binsize)*2), 
            meanrewdFF-scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),
            meanrewdFF+scipy.stats.sem(rewdFF,axis=1,nan_policy='omit'),            
            color='k',alpha=0.4)
            ax.set_ylim([-0.1,0.1])
            ax.axhline(0, color='k', linestyle='--')
            # ax.axhline(-.01, color='k', linestyle='--')
            ymin=min(meanrewdFF)-.05
            ymax=max(meanrewdFF)+.05-ymin
            ax.add_patch(
                patches.Rectangle(
            xy=(range_val/binsize,ymin),  # point of origin.
            width=dur/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim')
            plndff.append(rewdFF)
            fig.tight_layout()

        day_date_dff[f'{an}_{day}_{condition}_{nuc}'] = plndff

#%%
nuclei = ['SNc', 'VTA', 'LC']
deep_rewdff_saline_per_nuc = {}
deep_rewdff_drug_per_nuc = {}
sup_rewdff_saline_per_nuc = {}
sup_rewdff_drug_per_nuc = {}

for nuc in nuclei:
    # plot deep vs. superficial
    # plot control vs. drug
    plt.rc('font', size=12)
    norm_window = 2 #s
    planes=4
    # assumes 4 planes
    deep_rewdff_saline = []
    deep_rewdff_drug = []
    sup_rewdff_saline = []
    sup_rewdff_drug = []
    # halo
    for pln in range(planes):
        ii=0; 
        saline_dff = []
        drug_dff = []
        idx_to_catch = []
        
        for dy,v in day_date_dff.items():
            if nuc in dy:
                rewdFF = day_date_dff[dy][pln]
                if rewdFF.shape[1]>0:            
                    meanrewdFF = np.nanmean(rewdFF,axis=1)
                    meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                    rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                    if 'drug' in dy:
                        drug_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
                    else:
                        saline_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
                else: idx_to_catch.append(ii)
                ii+=1
        meanrewdFF_s = np.vstack([x[0] for x in saline_dff])
        rewdFF_s = np.hstack([x[1] for x in saline_dff])
        if len(drug_dff)>0:
            meanrewdFF_d = np.vstack([x[0] for x in drug_dff])
            rewdFF_d = np.hstack([x[1] for x in drug_dff])
        if pln==3:
            if len(drug_dff)>0: deep_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
            deep_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
        else:
            if len(drug_dff)>0: sup_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
            sup_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
    
    deep_rewdff_saline_per_nuc[nuc]=deep_rewdff_saline
    deep_rewdff_drug_per_nuc[nuc]=deep_rewdff_drug
    sup_rewdff_saline_per_nuc[nuc]=sup_rewdff_saline
    sup_rewdff_drug_per_nuc[nuc]=sup_rewdff_drug
#%%

# params per nuc
# ymin,ymax,stimsec,antag
nuc_params = {'SNc': [-0.02,0.05,1.2,'Eticlopride'],'LC': [-0.03,0.07,1.2,'SCH23390'],
            'VTA': [-0.03,0.03,1.2,'SCH23390']}
save_pvals = {}
for nuc in nuclei:
    print(nuc)
    ymin,ymax,stimsec,antag=nuc_params[nuc]
    deep_rewdff_saline=deep_rewdff_saline_per_nuc[nuc]
    deep_rewdff_drug=deep_rewdff_drug_per_nuc[nuc]
    sup_rewdff_saline=sup_rewdff_saline_per_nuc[nuc]
    sup_rewdff_drug=sup_rewdff_drug_per_nuc[nuc]

    # chop pre window
    pre_win_to_show=2
    frames_to_show = int((range_val/binsize)-(pre_win_to_show/binsize))
    if len(sup_rewdff_drug):
        an_sup_rewdff_drug=np.hstack([xx[1] for xx in sup_rewdff_drug])
        sup_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in sup_rewdff_drug])
    an_sup_rewdff_saline=np.hstack([xx[1] for xx in sup_rewdff_saline])
    sup_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in sup_rewdff_saline])

    an_deep_rewdff_saline=np.hstack([xx[1] for xx in deep_rewdff_saline])
    deep_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_saline])
    if len(deep_rewdff_drug):
        an_deep_rewdff_drug=np.hstack([xx[1] for xx in deep_rewdff_drug])
        deep_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_drug])

    patch_start = int(pre_win_to_show/binsize)
    # plot
    drug = [deep_rewdff_drug, sup_rewdff_drug]
    saline = [deep_rewdff_saline, sup_rewdff_saline]
    lbls = ['Deep', 'Superficial']
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(7,6), sharex=True)
    height=ymax-ymin
    for i in range(len(saline)):
        # plot
        ax=axes[i,0]
        meancond = np.nanmean(saline[i],axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]
        rewcond = saline[i] #-ctrl_mean_trace_per_pln[pln]
        ax.plot(meancond,linewidth=1.5,color='gray',label='Saline')   
        xmin,xmax = ax.get_xlim()         
        ax.fill_between(range(0,int(range_val/binsize)*2-frames_to_show), 
        meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        alpha=0.5,color='gray')  
        # also plot drug
        meancond = np.nanmean(drug[i],axis=1)#-ctrl_mean_trace_per_pln_d[pln]
        rewcond = drug[i] # -ctrl_mean_trace_per_pln_d[pln]    
        ax.plot(meancond,linewidth=1.5,color='royalblue',label=antag)   
        xmin,xmax = ax.get_xlim()         
        ax.fill_between(range(0,int(range_val/binsize)*2-frames_to_show), 
        meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        alpha=0.5,color='royalblue')        
        ax.axhline(0,color='k',linestyle='--')
        ax.spines[['top','right']].set_visible(False)

        # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
        ax.add_patch(
            patches.Rectangle(
        xy=(patch_start,ymin),  # point of origin.
        width=stimsec/binsize, height=height, linewidth=1, # width is s
        color='mediumspringgreen', alpha=0.2))
        ax.axhline(0,color='k',linestyle='--')
        
        ii+=1
        if i==0: ax.legend(); ax.set_title(f'{lbls[i]}')
        else: ax.set_title(f'{lbls[i]}')
        ax.set_ylim([ymin,ymax])
        if i==1: ax.set_xlabel('Time from LED onset (s)')
        ax.set_ylabel('$\Delta$ F/F')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)-frames_to_show+1,15))
        ax.set_xticklabels(range(-pre_win_to_show, range_val+1, 3))
    # plot control-drug
    # plot
    drug = [deep_rewdff_drug, sup_rewdff_drug]
    saline = [deep_rewdff_saline, sup_rewdff_saline]
    startframe = int(range_val/binsize)-frames_to_show
    for i in range(len(saline)):
        # plot
        ax=axes[i,1]
        drugtrace = np.nanmean(drug[i],axis=1)
        drugtrace_padded = np.zeros_like(drugtrace)
        drugtrace_padded[startframe:int((stimsec+1.5)/binsize+startframe)] = \
            drugtrace[startframe:int((stimsec+1.5)/binsize+startframe)] 
        rewcond = np.array([xx-drugtrace for xx in saline[i].T]).T #-ctrl_mean_trace_per_pln[pln]
        meancond = np.nanmean(rewcond,axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]

        ax.plot(meancond,linewidth=1.5,color='k',label=f'Saline-{antag}')   
        xmin,xmax = ax.get_xlim()         
        ax.fill_between(range(0,(int(range_val/binsize)*2)-frames_to_show), 
        meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
        alpha=0.5,color='k')  
        ax.add_patch(
            patches.Rectangle(
        xy=(patch_start,ymin),  # point of origin.
        width=stimsec/binsize, height=height, linewidth=1, # width is s
        color='mediumspringgreen', alpha=0.2))
        ax.axhline(0,color='k',linestyle='--')
        ii+=1    
        ax.set_ylim([ymin,ymax])
        if i==0: ax.legend(); ax.set_title(f'Sensor-dependent \n {lbls[i]}')
        else: ax.set_title(f'{lbls[i]}')
        ax.set_xticks(range(0, (int(range_val/binsize)*2)-frames_to_show+1,15))
        ax.set_xticklabels(range(-pre_win_to_show, range_val+1, 3))
        if i==1: ax.set_xlabel('Time from LED onset (s)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{nuc} axons, ChR2')    
    fig.tight_layout()
    # collect values for ttest
    # get subtraction
    drug = [deep_rewdff_drug, sup_rewdff_drug]
    saline = [deep_rewdff_saline, sup_rewdff_saline]
    andrug = [an_deep_rewdff_drug, an_sup_rewdff_drug]
    ansaline = [an_deep_rewdff_saline, an_sup_rewdff_saline]
    start_frame = int((range_val/binsize)-frames_to_show)
    save = []
    stimsec=1.2
    for i in range(2): # deep vs. sup
        # drug subtracted
        rewcond_h = np.array([xx-np.nanmean(drug[i],axis=1) for xx in saline[i].T]).T 
        # rewcond_h = np.array([xx for xx in saline[i].T]).T 
        stimdff_h = np.nanmean(rewcond_h[start_frame:start_frame+int(stimsec/binsize)],
                    axis=0)    
        t,pval = scipy.stats.ttest_1samp(stimdff_h, popmean=0)
        save.append([stimdff_h, pval, ansaline[i]])    
    # superficial vs. deep
    deep_rewcond_h = np.array([xx-np.nanmean(drug[0],axis=1) for xx in saline[0].T]).T 
    sup_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[1].T]).T 
    deep_stimdff_h = np.nanmean(deep_rewcond_h[start_frame:start_frame+int(stimsec/binsize)],
                    axis=0)
    sup_stimdff_h = np.nanmean(sup_rewcond_h[start_frame:start_frame+int(stimsec/binsize)],
                    axis=0)
    t,pval_deep_vs_sup = scipy.stats.ttest_ind(deep_stimdff_h, sup_stimdff_h)
    
    save_pvals[nuc]=save
    plt.savefig(os.path.join(savedst, f'{nuc}_axon_traces_v_subtracted.svg'))
#%%
# plots per trial vs. animal
sldf = []
nuc_n = {'SNc': 2, 'VTA': 2, 'LC': 3}
for nuc in nuclei:
    save=save_pvals[nuc]
    lbls = ['Deep', 'Superficial']
    plt.rc('font', size=25)
    dfs = []
    for pln in range(2):
        df = pd.DataFrame()
        df['mean_dff_during_stim'] = save[pln][0]
        if pln==0:
            selectivity_index = save[0]
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
    # pink and grey
    cmap = [np.array([230, 84, 128])/255,np.array([153, 153, 153])/255]
    g=sns.boxplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
            data=bigdf,fill=False,palette=cmap,
                linewidth=3)
    # sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',
    #         data=bigdf,s=11,palette=cmap,
    #         alpha=0.2,ax=ax,dodge=True)
    ax.axhline(0, color='k', linestyle='--',linewidth=3)
    ax.spines[['top','right']].set_visible(False)
    # ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation =45, ha='right')
    ax.set_ylabel('Mean $\Delta F/F$ during stim.')
    y=0.03
    fs=12
    i=0
    for i in range(len(lbls)):
        pval = bigdf.loc[bigdf.plane_subgroup==lbls[i], 'pval'].values[0]
        trials = bigdf[bigdf.plane_subgroup==lbls[i]]
        ax.text(i, y, f'p={pval:.7f}, \n{len(trials)} trials', ha='center', fontsize=fs, rotation=45)
        i+=1

    ax.text(i, y, f'deep vs. super\np={pval_deep_vs_sup:.7f}', ha='center', 
            fontsize=fs, rotation=45)
    ax.set_title(f'{nuc}\n n=trials, {nuc_n[nuc]} animals',pad=40,fontsize=14)
    plt.savefig(os.path.join(savedst, f'{nuc}_axon_subtracted_trial_quant.svg'))
    # per animal
    bigdfan = bigdf.groupby(['animal', 'plane_subgroup']).mean(numeric_only=True)

    # pink and grey
    fig,ax = plt.subplots(figsize=(2,5))
    g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,
            errorbar='se',ax=ax,linewidth=4,err_kws={'linewidth': 4},
            palette=cmap)
    sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,
            s=17,alpha=0.8,ax=ax,palette=cmap,dodge=True)
    ax.spines[['top','right']].set_visible(False)
    # ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Mean $\Delta F/F$ during stim.')
    ax.axhline(0, color='k', linestyle='--',linewidth=3)

    y=0.02
    fs=14
    i=0
    for i in range(len(lbls)):
        halo = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[i])), 'mean_dff_during_stim'].values
        t,pval = scipy.stats.ttest_1samp(halo, popmean=0)
        ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
        i+=1

    halo_d = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[0])), 'mean_dff_during_stim'].values
    halo_s = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[1])), 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_rel(halo_d, halo_s)
    ax.text(i, y, f'deep vs. super \n p={pval:.4f}', ha='center',
        fontsize=fs, rotation=45)

    ax.set_title(f'{nuc}\n n={nuc_n[nuc]} animals',pad=80,fontsize=14)
    plt.savefig(os.path.join(savedst, f'{nuc}_axon_subtracted_animal_quant.svg'))
    
    #selectivity index per nuclei
    df = pd.DataFrame()
    #0=deep,1=super    
    # num trials to compare
    ntrials=len(save[0][0])
    df['selectivity_index'] = (save[0][0]-save[1][0][:ntrials])/(abs(save[0][0])+abs(save[1][0][:ntrials]))    
    df['animal'] =  save[pln][2][:ntrials]
    df['nucleus'] =  [nuc]*len(df)
    sldf.append(df)
#%%
# selectivity ind
cmap=sns.color_palette('deep')
sldfbig=pd.concat(sldf)
sldfbig = sldfbig.groupby(['animal', 'nucleus']).mean(numeric_only=True)
fig,ax = plt.subplots(figsize=(5,7))
g = sns.boxplot(x='nucleus', y='selectivity_index', data=sldfbig, linewidth=3, palette=cmap, ax=ax,fill=False)
sns.stripplot(x='nucleus', y='selectivity_index', data=sldfbig, s=16, alpha=0.8, ax=ax, dodge=True, palette=cmap)

ax.spines[['top', 'right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Selectivity Index\n (Deep vs. Superficial)')

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

sldfbig=sldfbig.reset_index()
# Perform ANOVA
model = ols('selectivity_index ~ nucleus', data=sldfbig).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)

# Perform pairwise t-tests and apply Bonferroni correction
nuclei = sldfbig['nucleus'].unique()
p_values = []
comparisons = []

for i in range(len(nuclei)):
    for j in range(i + 1, len(nuclei)):
        group1 = sldfbig[sldfbig['nucleus'] == nuclei[i]]['selectivity_index']
        group2 = sldfbig[sldfbig['nucleus'] == nuclei[j]]['selectivity_index']
        t_stat, p_value = scipy.stats.ttest_ind(group1, group2)
        p_values.append(p_value)
        comparisons.append((nuclei[i], nuclei[j]))

# Apply Bonferroni correction
_, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')

# Annotate the plot
def add_stat_annotation(ax, x1, x2, y, p_value, adjusted_p):
    h = 0.03  # height offset
    line_offset = 0.02  # offset of the horizontal line
    # Draw horizontal line
    ax.plot([x1, x1, x2, x2], [y, y + line_offset, y + line_offset, y], lw=1.5, c='k')
    # Determine text format
    text = 'p={:.3f}'.format(adjusted_p)
    # Add p-value text
    ax.text((x1 + x2) * 0.5, y + line_offset, text, ha='center', va='bottom', color='k',
            fontsize=10)

# y position and add annotations
y_max = sldfbig['selectivity_index'].max()  # maximum y value of the plot
y_range = sldfbig['selectivity_index'].max() - sldfbig['selectivity_index'].min()  # range of y values

for i, (group1, group2) in enumerate(comparisons):
    x1 = list(sldfbig['nucleus'].unique()).index(group1)
    x2 = list(sldfbig['nucleus'].unique()).index(group2)
    y = y_max + (i + 1) * y_range * 0.05  # adjust y position for each annotation
    add_stat_annotation(ax, x1, x2, y, p_values[i], p_adjusted[i])

plt.tight_layout()
plt.savefig(os.path.join(savedst, 'selectivity_index_per_nuc.svg'))
