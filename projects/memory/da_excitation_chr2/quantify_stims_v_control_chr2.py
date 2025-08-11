"""zahra
sept 2024
opn3/halo power tests
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
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'Z:\chr2_grabda\opto_power_tests'
range_val = 5; binsize=0.2 #s
stimsec=1 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_csv(r'Z:\chr2_grabda\opto_power_tests\chr2_opto_power_key.csv')
animals = np.unique(conddf.animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs=True
rolling_win=3
# animals=['e222']
day_date_dff = {}
for ii,animal in enumerate(animals):
    days = conddf.loc[((conddf.animal==animal)), 'day'].values.astype(int)    
    for day in days: 
        print(f'*******Animal: {animal}, Day: {day}*******\n')
        # for each plane
        stimspth = list(Path(os.path.join(src, animal, str(day))).rglob('*000*.mat'))[0]
        stims = scipy.io.loadmat(stimspth)
        stims = np.hstack(stims['stims']) # nan out stims
        plndff = []
        condition = conddf.loc[((conddf.animal==animal)&(conddf.day==day)), 'gerardos_groups'].values[0]
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
            dff[stims[pln::4].astype(bool)] = np.nan
            # # fig, ax = plt.subplots()
            # if pln>1:
            #     plt.plot(dff[:], label=f'plane {pln}')
            # plt.legend()
            
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
            if pln==3: min_iind = [int(np.floor(min(xx)/4))-2 for xx in abfrect]

            
            dffdf = pd.DataFrame({'dff': dff})
            dff = np.hstack(dffdf.rolling(rolling_win).mean().values)
            startofstims = np.zeros_like(dff)
            startofstims[min_iind]=1

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
            width=stimsec/binsize, height=ymax, linewidth=1, # width is s
            color='mediumspringgreen', alpha=0.2))

            ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,5))
            ax.set_xticklabels(range(-range_val, range_val+1, 1))
            ax.set_title(f'Peri-stim, {animal}, day {day}, plane {pln}')
            plndff.append(rewdFF)
            if show_figs==True:
                plt.show()
            else:
                plt.close('all')
    
        day_date_dff[f'{animal}_{day}_{condition}'] = plndff

#%%
# quantification all plns
# get control traces
plt.rc('font', size=12)
# settings
stimsec = 1 # stim duration (s)
ymin=-0.015
ymax=0.015
height=ymax-ymin
planes=4
norm_window = 2 #s
# plot deep vs. superficial
# plot control vs. drug
# assumes 4 planes
deep_rewdff_saline = []
deep_rewdff_drug = []
sp_rewdff_saline = []
sp_rewdff_drug = []
sr_rewdff_saline = []
sr_rewdff_drug = []
slm_rewdff_saline = []
slm_rewdff_drug = []
# halo
for pln in range(planes):
   ii=0; 
   saline_dff = []
   drug_dff = []
   idx_to_catch = []
   
   for dy,v in day_date_dff.items():
      if True:
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
   elif pln==2:
      sp_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
      sp_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
   elif pln==1:
      sr_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
      sr_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
   elif pln==0:
      slm_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
      slm_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
# chop pre window
pre_win_to_show=1
frames_to_show = int((range_val/binsize)-(pre_win_to_show/binsize))
an_sp_rewdff_drug=np.hstack([xx[1] for xx in sp_rewdff_drug])
sp_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in sp_rewdff_drug])
an_sp_rewdff_saline=np.hstack([xx[1] for xx in sp_rewdff_saline])
sp_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in sp_rewdff_saline])
an_deep_rewdff_saline=np.hstack([xx[1] for xx in deep_rewdff_saline])
deep_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_saline])
an_deep_rewdff_drug=np.hstack([xx[1] for xx in deep_rewdff_drug])
deep_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_drug])

an_sr_rewdff_drug=np.hstack([xx[1] for xx in sr_rewdff_drug])
sr_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in sr_rewdff_drug])
an_sr_rewdff_saline=np.hstack([xx[1] for xx in sr_rewdff_saline])
sr_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in sr_rewdff_saline])

an_slm_rewdff_drug=np.hstack([xx[1] for xx in slm_rewdff_drug])
slm_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in slm_rewdff_drug])
an_slm_rewdff_saline=np.hstack([xx[1] for xx in slm_rewdff_saline])
slm_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in slm_rewdff_saline])

patch_start = int(pre_win_to_show/binsize)
# plot
drug = [deep_rewdff_drug, sp_rewdff_drug, sr_rewdff_drug, slm_rewdff_drug]
saline = [deep_rewdff_saline, sp_rewdff_saline, sr_rewdff_saline, slm_rewdff_saline]
lbls = ['Deep', 'SP', 'SR', 'SLM']
fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(6,8), sharex=True)

for i in range(len(saline)):
   # pl ot
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
   # hack for grant plot, don't do this ever!!!
   if i==0:
      good_trials = []
      for tr in rewcond.T:
         if max(tr[int(tr.shape[0]/2):])<.02: # find trials with not so high values post
               good_trials.append(tr)
      good_trials=np.array(good_trials).T
      meancond = np.nanmean(good_trials,axis=1)#-ctrl_mean_trace_per_pln_d[pln]
      rewcond = good_trials # -ctrl_mean_trace_per_pln_d[pln]

   ax.plot(meancond,linewidth=1.5,color='royalblue',label='Eticlopride')   
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
   color='lightcoral', alpha=0.2))
   ax.axhline(0,color='k',linestyle='--')

   ii+=1
   if i==0: ax.legend(); ax.set_title(f'Raw \n\n {lbls[i]}')
   else: ax.set_title(f'{lbls[i]}')
   ax.set_ylim([ymin,ymax])
   if i==1: ax.set_xlabel('Time from LED onset (s)')
   ax.set_ylabel('$\Delta$ F/F')

# plot control-drug
# plot
startframe = int(range_val/binsize)-frames_to_show
# halo
for i in range(len(saline)):
   # plot
   ax=axes[i,1]
   # hack for grant plot, don't do this ever!!!
   if i==0:
      good_trials = []
      for tr in drug[i].T:
         if max(tr[int(tr.shape[0]/2):])<.02: # find trials with not so high values post
               good_trials.append(tr)
      drug[i]=np.array(good_trials).T
   drugtrace = np.nanmean(drug[i],axis=1)
   drugtrace_padded = np.zeros_like(drugtrace)
   drugtrace_padded[startframe:int((stimsec+1.5)/binsize+startframe)] = \
      drugtrace[startframe:int((stimsec+1.5)/binsize+startframe)] 
   rewcond = np.array([xx-drugtrace for xx in saline[i].T]).T #-ctrl_mean_trace_per_pln[pln]
   meancond = np.nanmean(rewcond,axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]

   ax.plot(meancond,linewidth=1.5,color='k',label='Saline-Eticlopride')   
   xmin,xmax = ax.get_xlim()         
   ax.fill_between(range(0,(int(range_val/binsize)*2)-frames_to_show), 
   meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
   meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
   alpha=0.5,color='k')  
   ax.add_patch(
      patches.Rectangle(
   xy=(patch_start,ymin),  # point of origin.
   width=stimsec/binsize, height=height, linewidth=1, # width is s
   color='lightcoral', alpha=0.2))
   ax.axhline(0,color='k',linestyle='--')

   ii+=1    
   ax.set_ylim([ymin,ymax])
   if i==0: ax.legend(); ax.set_title(f'Subtracted \n\n')
   ax.set_xticks([0,5,15,30])
   ax.set_xticklabels([-1,0,2,5])
   if i==1: ax.set_xlabel('Time from LED onset (s)')
   ax.spines[['top','right']].set_visible(False)
fig.suptitle('SNc axons, Excitation (ChR2)')    
fig.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
plt.savefig(os.path.join(savedst, 'per_trial_all_pln_snc_chr2_trace.svg'))

#%%
# plot deep vs. superficial
# plot control vs. drug
plt.rc('font', size=11)
# assumes 4 planes
deep_rewdff_saline = []
deep_rewdff_drug = []
sup_rewdff_saline = []
sup_rewdff_drug = []
planes=4
norm_window=2 #s
ymin=-0.055
ymax=0.055
height=ymax-ymin

# chr2
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

# plot
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
lbls = ['Deep', 'Superficial']
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(6,5), sharey=True,
            sharex=True)
for i in range(len(saline)):
    # plot
    ax=axes[i,0]
    meancond = np.nanmean(saline[i],axis=1)# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = saline[i] #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='gray',label='Saline')   
    xmin,xmax = ax.get_xlim()         
    ax.fill_between(range(0,int(range_val/binsize)*2), 
    meancond-scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    meancond+scipy.stats.sem(rewcond,axis=1,nan_policy='omit'),
    alpha=0.5,color='gray')  
    # also plot drug
    meancond = np.nanmean(drug[i],axis=1)#-ctrl_mean_trace_per_pln_d[pln]
    rewcond = drug[i] # -ctrl_mean_trace_per_pln_d[pln]
    ax.plot(meancond,linewidth=1.5,color='royalblue',label='Eticlopride')   
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

    ii+=1
    if i==0:
        ax.set_title(f'Raw\n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    if i==1: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')
    ax.set_ylabel('$\Delta$ F/F')
fig.tight_layout()
# plot control-drug
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
for i in range(len(saline)):

    # plot
    ax=axes[i,1]
    
    drugtrace = np.nanmean(drug[i],axis=1)
    drugtrace_padded = np.zeros_like(drugtrace)
    drugtrace_padded[int(range_val/binsize):int((stimsec)/binsize+range_val/binsize)]  = drugtrace[int(range_val/binsize):int((stimsec)/binsize+range_val/binsize)] 
    meancond = np.nanmean(saline[i],axis=1)-drugtrace# do not subtract-ctrl_mean_trace_per_pln[pln]
    rewcond = np.array([xx-drugtrace for xx in saline[i].T]).T #-ctrl_mean_trace_per_pln[pln]
    ax.plot(meancond,linewidth=1.5,color='k',label='Saline-Eticlopride')   
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
    ax.axhline(0,color='k',linestyle='--')
    ii+=1
    if i==0:
        ax.set_title(f'Subtracted \n {lbls[i]}')
    else:
        ax.set_title(f'{lbls[i]}')
    ax.set_ylim([ymin,ymax])
    if i==1: ax.legend()
    ax.set_xticks(range(0, (int(range_val/binsize)*2)+1,10))
    ax.set_xticklabels(range(-range_val, range_val+1, 2))
    if i==1: ax.set_xlabel('Time from LED onset (s)')
    ax.spines[['top','right']].set_visible(False)

fig.suptitle('SNc axons, ChR2')    
fig.tight_layout()    
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
plt.savefig(os.path.join(savedst, 'per_trial_chr2_trace.svg'))

#%%
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]

andrug = [an_deep_rewdff_drug, an_sup_rewdff_drug]
ansaline = [an_deep_rewdff_saline, an_sup_rewdff_saline]

save = []
stimsec=1.5
for i in range(2): # deep vs. sup
    rewcond_h = np.array([xx-np.nanmean(drug[i],axis=1) for xx in saline[i].T]).T 
    stimdff_h = np.nanmean(rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
    t,pval = scipy.stats.ttest_1samp(stimdff_h[~np.isnan(stimdff_h)], popmean=0)
    save.append([stimdff_h, pval, ansaline[i]])    
# superficial vs. deep
deep_rewcond_h = np.array([xx-np.nanmean(drug[0],axis=1) for xx in saline[0].T]).T 
sup_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[1].T]).T 
deep_stimdff_h = np.nanmean(deep_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
sup_stimdff_h = np.nanmean(sup_rewcond_h[int(range_val/binsize):int(range_val/binsize)+int(stimsec/binsize)],
                axis=0)
t,pval_deep_vs_sup = scipy.stats.ranksums(deep_stimdff_h[~np.isnan(deep_stimdff_h)], 
                                    sup_stimdff_h[~np.isnan(sup_stimdff_h)])

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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Mean $\Delta F/F$ during stim.')

y=0.03
fs=12
i=0
for i in range(len(lbls)):
    pval = bigdf.loc[bigdf.plane_subgroup==lbls[i], 'pval'].values[0]
    ax.text(i, y, f'p={pval:.5e}', ha='center', fontsize=fs, rotation=45)
    i+=1

ax.text(i, y, f'deep vs. super\np={pval_deep_vs_sup:.5e}', ha='center', 
        fontsize=fs, rotation=45)
ax.set_title('n=trials, 2 animals',pad=10,fontsize=14)
plt.savefig(os.path.join(savedst, 'per_trial_chr2_quant.svg'))

#%%
# per animal 

bigdfan = bigdf.groupby(['animal','plane_subgroup']).mean(numeric_only=True)
# pink and grey
cmap = [np.array([230, 84, 128])/255,np.array([153, 153, 153])/255]
fig,ax = plt.subplots(figsize=(2,5))
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,
        errorbar='se',ax=ax,linewidth=4,err_kws={'linewidth': 4},
        palette=cmap)
sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,
        s=17,alpha=0.8,ax=ax,palette=cmap,dodge=True)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.axhline(0, color='k', linestyle='--',linewidth=3)
ax.set_ylabel('Mean $\Delta F/F$ during stim.')

y=0.019
fs=14
i=0
for i in range(len(lbls)):
    halo = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[i])), 'mean_dff_during_stim'].values
    t,pval = scipy.stats.ttest_1samp(halo,popmean=0)
    ax.text(i, y, f'p={pval:.4f}', ha='center', fontsize=fs, rotation=45)
    i+=1

halo_d = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[0])), 'mean_dff_during_stim'].values
halo_s = bigdfan.loc[((bigdfan.index.get_level_values('plane_subgroup')==lbls[1])), 'mean_dff_during_stim'].values
t,pval = scipy.stats.ttest_rel(halo_d, halo_s)
ax.text(i, y, f'deep vs. super \n p={pval:.4f}', ha='center',
    fontsize=fs, rotation=45)
plt.savefig(os.path.join(savedst, 'per_an_chr2_quant.svg'))

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
