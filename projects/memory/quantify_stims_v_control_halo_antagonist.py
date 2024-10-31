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

plt.close('all')
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'Y:\halo_grabda'
range_val = 15; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_excel(r'Y:\halo_grabda\halo_key.xlsx',sheet_name='halo') # day vs. condition LUT
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs = False # show individual days peri stim plots 
# animals = ['e241', 'e242', 'e243']
rolling_win = 10
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
            dff = np.hstack(dffdf.rolling(rolling_win).mean().values)
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
            if not show_figs: plt.close('all')
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
height=ymax-ymin
planes=4
norm_window = 2 #s
# subtract ctrl
fig,axes=plt.subplots(nrows=planes,figsize=(3,6))
ctrl_mean_trace_per_pln=[]; ctrl_mean_trace_per_pln_d=[] # split into saline/none vs. drug days
for pln in range(planes):
    ii=0; condition_dff = []; condition_dff_d = []
    idx_to_catch = []
    for dy,v in day_date_dff.items():
        if (conddf.loc[conddf.animal==dy[:4],'condition'].values[0]=='control'):
            if 'drug' not in dy:
                rewdFF = day_date_dff[dy][pln] 
                if rewdFF.shape[1]>0:            
                    meanrewdFF = np.nanmean(rewdFF,axis=1)
                    meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                    rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                    condition_dff.append([meanrewdFF, rewdFF_prewin])
                else: idx_to_catch.append(ii)
            elif 'drug' in dy:
                rewdFF = day_date_dff[dy][pln] 
                if rewdFF.shape[1]>0:            
                    meanrewdFF = np.nanmean(rewdFF,axis=1)
                    meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                    rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                    condition_dff_d.append([meanrewdFF, rewdFF_prewin])
                else: idx_to_catch.append(ii)

    meanrewdFF = np.nanmean(np.hstack([x[1] for x in condition_dff]),axis=1) # mean across days
    meanrewdFF_d = np.nanmean(np.hstack([x[1] for x in condition_dff_d]),axis=1) # mean across days
    ctrl_mean_trace_per_pln.append(meanrewdFF); ctrl_mean_trace_per_pln_d.append(meanrewdFF_d)
    ax = axes[pln]
    ax.plot(meanrewdFF, label='control saline')
    ax.plot(meanrewdFF_d, label='control drug')
    ax.set_title(f'Plane {pln}')
    ax.axvline(int(range_val/binsize),color='k',linestyle='--')
    ax.set_ylim([ymin, ymax])
    if pln==3: ax.legend()
#%%
# plot control vs. drug
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
    meancond = np.nanmean(meanrewdFF_s,axis=0)# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx for xx in rewdFF_s.T]).T #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='k',label='none')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='k')  
    # also plot drug
    meancond = np.nanmean(meanrewdFF_d,axis=0)#-ctrl_mean_trace_per_pln_d[pln]
    rewcond = np.array([xx for xx in rewdFF_d.T]).T # -ctrl_mean_trace_per_pln_d[pln]
    ax.plot(meancond,linewidth=1.5,color='royalblue',label='drug')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='royalblue')        
    ax.axhline(0,color='k',linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))

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
    meancond_s = np.nanmean(meanrewdFF_s,axis=0)#-ctrl_mean_trace_per_pln[pln]
    meancond_d = np.nanmean(meanrewdFF_d,axis=0)#-ctrl_mean_trace_per_pln[pln]
    meancond = meancond_s-meancond_d
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
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))
    ax.axhline(0,color='k',linestyle='--')
    ii+=1
    ax.set_title(f'\nPlane {planelut[pln]}')
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim([ymin,ymax])
    if pln==3: ax.legend()
ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
ax.set_xticklabels(range(-range_val, range_val+1, 2))
fig.suptitle(f'Control vs. drug \ LEDon- LEDon+SCH23390 (3.2mg/kg)')

fig.tight_layout()

#%%
# plot deep vs. superficial
# plot control vs. drug
plt.rc('font', size=11)
# assumes 4 planes
deep_rewdff_saline = []
deep_rewdff_drug = []
sup_rewdff_saline = []
sup_rewdff_drug = []
deep_rewdff_saline_c = []
deep_rewdff_drug_c = []
sup_rewdff_saline_c = []
sup_rewdff_drug_c = []
# halo
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
                    drug_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
                else:
                    saline_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
            else: idx_to_catch.append(ii)
            ii+=1

    meanrewdFF_s = np.vstack([x[0] for x in saline_dff])
    rewdFF_s = np.hstack([x[1] for x in saline_dff])
    meanrewdFF_d = np.vstack([x[0] for x in drug_dff])
    rewdFF_d = np.hstack([x[1] for x in drug_dff])
    if pln==3:
        deep_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
        deep_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
    else:
        sup_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
        sup_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
# ctrl
for pln in range(planes):
    ii=0; 
    saline_dff = []
    drug_dff = []
    idx_to_catch = []
    
    for dy,v in day_date_dff.items():
        if conddf.loc[conddf.animal==dy[:4],'condition'].values[0]=='control':
            rewdFF = day_date_dff[dy][pln] # so only
            if rewdFF.shape[1]>0:            
                meanrewdFF = np.nanmean(rewdFF,axis=1)
                meanrewdFF = meanrewdFF-np.nanmean(meanrewdFF[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) #pre-window
                rewdFF_prewin = np.array([xx-np.nanmean(xx[int((range_val/binsize)-norm_window/binsize):int(range_val/binsize)]) for xx in rewdFF.T]).T
                if 'drug' in dy:
                    drug_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]]) # save animals
                else:
                    saline_dff.append([meanrewdFF, rewdFF_prewin, [dy[:4]]*rewdFF_prewin.shape[1]])
            else: idx_to_catch.append(ii)
            ii+=1

    meanrewdFF_s = np.vstack([x[0] for x in saline_dff])
    rewdFF_s = np.hstack([x[1] for x in saline_dff])
    meanrewdFF_d = np.vstack([x[0] for x in drug_dff])
    rewdFF_d = np.hstack([x[1] for x in drug_dff])
    if pln==3:
    
        deep_rewdff_drug_c.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
        deep_rewdff_saline_c.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
    else:
        sup_rewdff_drug_c.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
        sup_rewdff_saline_c.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])

#%%
# get animals
# add all layers together
an_sup_rewdff_drug=np.hstack([xx[1] for xx in sup_rewdff_drug])
sup_rewdff_drug=np.hstack([xx[0] for xx in sup_rewdff_drug])
an_sup_rewdff_saline=np.hstack([xx[1] for xx in sup_rewdff_saline])
sup_rewdff_saline=np.hstack([xx[0] for xx in sup_rewdff_saline])

an_deep_rewdff_saline=np.hstack([xx[1] for xx in deep_rewdff_saline])
deep_rewdff_saline=np.hstack([xx[0] for xx in deep_rewdff_saline])
an_deep_rewdff_drug=np.hstack([xx[1] for xx in deep_rewdff_drug])
deep_rewdff_drug=np.hstack([xx[0] for xx in deep_rewdff_drug])

an_sup_rewdff_drug_c=np.hstack([xx[1] for xx in sup_rewdff_drug_c])
sup_rewdff_drug_c=np.hstack([xx[0] for xx in sup_rewdff_drug_c])
an_sup_rewdff_saline_c=np.hstack([xx[1] for xx in sup_rewdff_saline_c])
sup_rewdff_saline_c=np.hstack([xx[0] for xx in sup_rewdff_saline_c])

an_deep_rewdff_saline_c=np.hstack([xx[1] for xx in deep_rewdff_saline_c])
deep_rewdff_saline_c=np.hstack([xx[0] for xx in deep_rewdff_saline_c])
an_deep_rewdff_drug_c=np.hstack([xx[1] for xx in deep_rewdff_drug_c])
deep_rewdff_drug_c=np.hstack([xx[0] for xx in deep_rewdff_drug_c])

ymin=-0.01
ymax=0.01
# plot
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
drug_c = [deep_rewdff_drug_c, sup_rewdff_drug_c]
saline_c = [deep_rewdff_saline_c, sup_rewdff_saline_c]
lbls = ['Deep (SO)', 'Superficial (SP, SR, SLM)']
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(6,5), sharex=True)
for i in range(len(saline)):
    # plot
    ax=axes[i,0]
    meancond = np.nanmean(saline[i],axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = saline[i] #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='k',label='Saline')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='k')  
    # also plot drug
    meancond = np.nanmean(drug[i],axis=1)#-ctrl_mean_trace_per_pln_d[pln]
    rewcond = drug[i] # -ctrl_mean_trace_per_pln_d[pln]
    ax.plot(meancond,linewidth=1.5,color='royalblue',label='SCH23390')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='royalblue')        
    ax.axhline(0,color='k',linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))

    ii+=1
    if i==0:
        ax.set_title(f'eNpHR3.0 \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    if i==0: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')

# ctrl
for i in range(len(saline)):
    # plot
    ax=axes[i,1]
    meancond = np.nanmean(saline_c[i],axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = saline_c[i] #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='slategray',label='Saline')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='slategray')  
    # also plot drug
    meancond = np.nanmean(drug_c[i],axis=1)#-ctrl_mean_trace_per_pln_d[pln]
    rewcond = drug_c[i] # -ctrl_mean_trace_per_pln_d[pln]
    ax.plot(meancond,linewidth=1.5,color='skyblue',label='SCH23390')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='skyblue')        
    ax.axhline(0,color='k',linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))

    ii+=1
    if i==0:
        ax.set_title(f'Control-YFP \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    
    ax.set_ylim([ymin,ymax])
    if i==0: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    ax.set_xlabel('Time from LED onset (s)')
    if i==1: ax.set_xlabel('Time from LED onset (s)')
fig.tight_layout()

#%%
# plot control-drug
ymin=-0.01
ymax=0.01
plt.rc('font', size=12) 
# plot
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
drug_c = [deep_rewdff_drug_c, sup_rewdff_drug_c]
saline_c = [deep_rewdff_saline_c, sup_rewdff_saline_c]
lbls = ['Deep (SO)', 'Superficial (SP, SR, SLM)']
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(8,6), sharex=True)
# halo
for i in range(len(saline)):
    # plot
    ax=axes[i,0]
    
    drugtrace = np.nanmean(drug[i],axis=1)
    drugtrace_padded = np.zeros_like(drugtrace)
    drugtrace_padded[int(range_val/binsize):int((stimsec+1.5)/binsize+range_val/binsize)]  = drugtrace[int(range_val/binsize):int((stimsec+1.5)/binsize+range_val/binsize)] 
    meancond = np.nanmean(saline[i],axis=1)-drugtrace# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-drugtrace for xx in saline[i].T]).T #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='k',label='Saline-SCH23390')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='k')  
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))
    ax.axhline(0,color='k',linestyle='--')

    ii+=1
    if i==0:
        ax.set_title(f'eNpHR3.0 \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    if i==0: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')
    ax.spines[['top','right']].set_visible(False)

# ctrl
for i in range(len(saline)):
    # plot
    ax=axes[i,1]
    drugtrace = np.nanmean(drug_c[i],axis=1)
    drugtrace_padded = np.zeros_like(drugtrace)
    drugtrace_padded[int(range_val/binsize):int((stimsec+1.5)/binsize+range_val/binsize)]  = drugtrace[int(range_val/binsize):int((stimsec+1.5)/binsize+range_val/binsize)] 
    meancond = np.nanmean(saline_c[i],axis=1)-drugtrace_padded# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-drugtrace_padded for xx in saline_c[i].T]).T #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='slategray',label='Saline-SCH23390')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='slategray')  
    # if pln==3: ymin=-0.06; ymax=0.06-(ymin)
    ax.add_patch(
        patches.Rectangle(
    xy=(range_val/binsize,ymin),  # point of origin.
    width=stimsec/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.2))
    # plot taper
    ax.add_patch(
        patches.Rectangle(
    xy=((range_val/binsize)+stimsec/binsize,ymin),  # point of origin.
    width=1.5/binsize, height=height, linewidth=1, # width is s
    color='mediumspringgreen', alpha=0.1))
    ax.axhline(0,color='k',linestyle='--')

    ii+=1
    if i==0:
        ax.set_title(f'Control-YFP \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    
    ax.set_ylim([ymin,ymax])
    if i==0: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout()


#%%
# collect values for ttest
# get subtraction
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
drug_c = [deep_rewdff_drug_c, sup_rewdff_drug_c]
saline_c = [deep_rewdff_saline_c, sup_rewdff_saline_c]

andrug = [an_deep_rewdff_drug, an_sup_rewdff_drug]
ansaline = [an_deep_rewdff_saline, an_sup_rewdff_saline]
andrug_c = [an_deep_rewdff_drug_c, an_sup_rewdff_drug_c]
ansaline_c = [an_deep_rewdff_saline_c, an_sup_rewdff_saline_c]

save = []
for i in range(2): # deep vs. sup
    rewcond_h = np.array([xx-np.nanmean(drug[i],axis=1) for xx in saline[i].T]).T 
    rewcond_c = np.array([xx-np.nanmean(drug_c[i],axis=1) for xx in saline_c[i].T]).T 
    stimdff_h = np.nanmean(rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
    stimdff_c = np.nanmean(rewcond_c[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
    t,pval = scipy.stats.ranksums(stimdff_h, stimdff_c[~np.isnan(stimdff_c)])
    save.append([stimdff_h, stimdff_c, pval, ansaline[i], ansaline_c[i]])    
# superficial vs. deep
deep_rewcond_h = np.array([xx-np.nanmean(drug[0],axis=1) for xx in saline[0].T]).T 
sup_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[1].T]).T 
deep_stimdff_h = np.nanmean(deep_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
sup_stimdff_h = np.nanmean(sup_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
t,pval_deep_vs_sup = scipy.stats.ranksums(deep_stimdff_h, sup_stimdff_h)
#%%
plt.rc('font', size=25)
dfs = []
for pln in range(2):
    df = pd.DataFrame()
    df['mean_dff_during_stim'] = np.concatenate([save[pln][0],save[pln][1]])
    df['condition'] = np.concatenate([['eNpHR3.0']*len(save[pln][0]),['Control-YFP']*len(save[pln][1])])
    pval=save[pln][2]
    df['pval']=[pval]*len(df)
    df['plane_subgroup'] =lbls[pln]
    df['animal'] =  np.concatenate([save[pln][3],save[pln][4]])
    # df['plane_subgroup'] = np.concatenate([[plnsg]*len(df)])
    dfs.append(df)
bigdf = pd.concat(dfs)
bigdf = bigdf.reset_index()
import seaborn as sns

fig,ax = plt.subplots(figsize=(3,5))
# pink and grey
cmap = [np.array([230, 84, 128])/255,np.array([153, 153, 153])/255]
g=sns.boxplot(x='condition',y='mean_dff_during_stim',hue='plane_subgroup',
        data=bigdf,fill=False,palette=cmap,
            linewidth=3)
sns.stripplot(x='condition',y='mean_dff_during_stim',hue='plane_subgroup',
        data=bigdf,s=11,palette=cmap,
        alpha=0.2,ax=ax,dodge=True)
ax.axhline(0, color='k', linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

y=0.04
fs=12
i=0
for i in range(len(lbls)):
    pval = bigdf.loc[bigdf.plane_subgroup==lbls[i], 'pval'].values[0]
    ax.text(i, y, f'p={pval:.7f}', ha='center', fontsize=fs, rotation=45)
    i+=1

ax.text(i, y, f'halo deep vs. super\np={pval_deep_vs_sup:.7f}', ha='center', 
        fontsize=fs, rotation=45)
ax.set_title('n=trials, 3 animals',pad=100,fontsize=14)

#%%
# per animal 

bigdfan = bigdf.groupby(['animal', 'condition','plane_subgroup']).mean(numeric_only=True)
# # # Specify the desired order
# desired_order = ['SLM', 'SR', 'SP', 'SO']

# # Convert the 'City' column to a categorical type with the specified order
# bigdfan['plane'] = pd.Categorical(bigdfan['plane'], categories=desired_order, ordered=True)

# # Sort the DataFrame by the 'City' column
# bigdfan.sort_values('plane')
# pink and grey
cmap = [np.array([230, 84, 128])/255,np.array([153, 153, 153])/255]
fig,ax = plt.subplots(figsize=(3.5,5))
g=sns.barplot(x='condition',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,
        errorbar='se',ax=ax,linewidth=4,err_kws={'linewidth': 4},
        palette=cmap)
sns.stripplot(x='condition',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,
        s=17,alpha=0.8,ax=ax,palette=cmap,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

y=0.0035
fs=14
i=0
for i in range(len(lbls)):
    halo = bigdfan.loc[((bigdfan.index.get_level_values('condition')=='eNpHR3.0') & \
        (bigdfan.index.get_level_values('plane_subgroup')==lbls[i])), 'mean_dff_during_stim'].values
    ctrl = bigdfan.loc[((bigdfan.index.get_level_values('condition')=='Control-YFP') & \
        (bigdfan.index.get_level_values('plane_subgroup')==lbls[i])), 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_ind(halo, ctrl)
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
    i+=1

halo_d = bigdfan.loc[((bigdfan.index.get_level_values('condition')=='eNpHR3.0') & \
    (bigdfan.index.get_level_values('plane_subgroup')==lbls[0])), 'mean_dff_during_stim'].values
halo_s = bigdfan.loc[((bigdfan.index.get_level_values('condition')=='eNpHR3.0') & \
    (bigdfan.index.get_level_values('plane_subgroup')==lbls[1])), 'mean_dff_during_stim'].values
t,pval = scipy.stats.ttest_rel(halo_d, halo_s)
ax.text(i, y, f'halo deep vs. super \n p={pval:.4f}', ha='center',
    fontsize=fs, rotation=45)

ax.set_title('n=3 animals',pad=100)

# Step 1: Calculate the means and standard deviations
mean1 = np.mean(halo_d)
mean2 = np.mean(halo_s)
std1 = np.std(halo_d, ddof=1)
std2 = np.std(halo_s, ddof=1)

# Step 2: Calculate pooled standard deviation
n1, n2 = len(halo_d), len(halo_s)
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

# Step 3: Calculate Cohen's d
cohens_d = (mean1 - mean2) / pooled_std

# Step 4: Perform Power Analysis using the calculated Cohen's d
alpha = 0.05  # Significance level
power = 0.8   # Desired power

import statsmodels.stats.power as smp
analysis = smp.TTestIndPower()
sample_size = analysis.solve_power(effect_size=cohens_d, alpha=alpha, power=power, alternative='two-sided')

print(f"Cohen's d: {cohens_d:.4f}")
print(f"Required sample size per group: {sample_size:.2f}")
