"""zahra
dec 2024
lc chr2 power tests
"""
#%%
import os, numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.DLC_behavior_classification import eye
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
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'X:\lc_chr2_grabda'
range_val = 5; binsize=0.2 #s
dur=1# s stim duration
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
prewin = 2 # for which to normalize

conddf = pd.read_csv(r"C:\Users\Han\Downloads\data_organization - chr2_lc_grabda3m.csv") # day vs. condition LUT
animals = np.unique(conddf.Animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs = False # show individual days peri stim plots 
# animals = ['e241', 'e242', 'e243']
rolling_win = 3
day_date_dff = {}

for ii,animal in enumerate(animals):
    days = conddf.loc[((conddf.Animal==animal)), 'Day'].values.astype(int)    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
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
            row =  6
            dff = np.hstack(params['params'][0][0][row][0][0])/np.nanmean(np.hstack(params['params'][0][0][row][0][0]))#/np.hstack(params['params'][0][0][9])            
            # plot mean img
            ax=axes[0,pln]
            ax.imshow(params['params'][0][0][0],cmap='Greys_r')
            ax.axis('off')
            ax.set_title(f'{animal}, day {day}, {planelut[pln]}')
            # nan out stims
            # dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')
            # plt.legend()
            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(2).mean().values)
            # get off plane stim
            offpln=pln+1 if pln<3 else pln-1
            startofstims = consecutive_stretch(np.where(stims[offpln::4])[0])
            min_iind = [min(xx) for xx in startofstims if len(xx)>0]
            # # remove rewarded stims
            cs=params['solenoid2'][0]
            # # cs within 50 frames of start of stim - remove
            framelim=20
            unrewstimidx = [idx for idx in min_iind if sum(cs[idx-framelim:idx+framelim])==0]            
            startofstims = np.zeros_like(dff)
            startofstims[unrewstimidx]=1
            # # get on plane stim for red laser
            # offpln=pln
            # ss = consecutive_stretch(np.where(stims[offpln::4])[0])
            # min_iind = [min(xx) for xx in ss if len(xx)>0]
            # # remove rewarded stims
            # cs=params['solenoid2'][0]
            # # cs within 50 frames of start of stim - remove
            # nan out stims
            # dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')

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
            # plt.show()      
        condition = conddf.loc[((conddf.Animal==animal) & (conddf.Day==day)), 'antagonist'].values[0]    

        day_date_dff[f'{animal}_{day}_{condition}'] = plndff

#%%
# plot deep vs. superficial
# plot control vs. drug
plt.rc('font', size=12)
norm_window = 3 #s
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
        rewdFF = day_date_dff[dy][pln] # so only
        if rewdFF.shape[1]>0:            
            meanrewdFF = np.nanmean(rewdFF,axis=1)
            meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
            rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
            if 'True' in dy:
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
# chop pre window
pre_win_to_show=3
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
ymin=-0.05
ymax=0.1
stimsec=1.2
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
    ax.plot(meancond,linewidth=1.5,color='royalblue',label='SCH23390')   
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
    if i==0: ax.legend(); ax.set_title(f'Raw \n\n {lbls[i]}')
    else: ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    if i==1: ax.set_xlabel('Time from LED onset (s)')
    ax.set_ylabel('$\Delta$ F/F')
    ax.set_xticks(range(0, (int(range_val/binsize)*2)-frames_to_show+1,15))
    ax.set_xticklabels(range(-pre_win_to_show, range_val+1, 3))
