"""zahra
aug 2025
snc power tests
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
from scipy.ndimage import label

plt.close('all')
# save to pdf
# dst = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects"
# pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(dst,
#     f"halo_opto.pdf"))

src = r'X:\chrimson_snc_grabda'
range_val = 5; binsize=0.2 #s
planelut  = {0: 'SLM', 1: 'SR' , 2: 'SP', 3: 'SO'}
conddf = pd.read_csv(r'C:\Users\Han\Downloads\data_organization - chrimson_snc.csv')# day vs. condition LUT
animals = np.unique(conddf.Animal.values.astype(str))
animals = np.array([an for an in animals if 'nan' not in an])
show_figs = False # show individual days peri stim plots 
# animals = ['e241', 'e242', 'e243']
rolling_win =3
day_date_dff = {}
for ii,animal in enumerate(animals):
   days = conddf.loc[((conddf.Animal==animal) & (conddf.led==True)), 'Day'].values.astype(int)    
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
         stims=params['optoEventALL'][0]

         if np.sum(stims[pln::4].astype(bool))>0:
            dff[stims[pln::4].astype(bool)] = np.nan
         else:
            try:
               dff[stims[pln-2::4].astype(bool)] = np.nan
            except:
               dff[stims[pln+2::4].astype(bool)] = np.nan
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
      condition = conddf.loc[((conddf.Animal==animal) & (conddf.Day==day)), 'antagonist'].values[0] 
      if condition==True:
         condition='drug'
      else:
         condition='none'  
      
      print(condition) 
      day_date_dff[f'{animal}_{day}_{condition}'] = plndff

#%%
# quantification all plns
# get control traces
plt.rc('font', size=14)
# settings
stimsec = 2 # stim duration (s)
ymin=-0.015 # chr2
ymax=0.01
ymin=-0.006# chr2
ymax=0.004

height=ymax-ymin
planes=4
norm_window = 3 #s
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
lbls = ['SO', 'SP', 'SR', 'SLM']
fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(6,8), sharex=True,sharey=True)

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
   if i==3: ax.set_xlabel('Time from LED onset (s)')
   ax.spines[['top','right']].set_visible(False)
fig.suptitle('SNc axons, Excitation (Chrimson)')    
fig.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
plt.savefig(os.path.join(savedst, 'per_trial_all_pln_chrimson_trace.svg'))

#%%
# quantification deep v superficial
# get control traces
plt.rc('font', size=12)
# settings
stimsec = 2 # stim duration (s)
height=ymax-ymin
planes=4
norm_window = 2 #s
# plot deep vs. superficial
# plot control vs. drug
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
   else:
      sup_rewdff_drug.append([rewdFF_d,np.hstack([x[2] for x in drug_dff])])
      sup_rewdff_saline.append([rewdFF_s,np.hstack([x[2] for x in saline_dff])])
# chop pre window
pre_win_to_show=1
frames_to_show = int((range_val/binsize)-(pre_win_to_show/binsize))
an_sup_rewdff_drug=np.hstack([xx[1] for xx in sup_rewdff_drug])
sup_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in sup_rewdff_drug])
an_sup_rewdff_saline=np.hstack([xx[1] for xx in sup_rewdff_saline])
sup_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in sup_rewdff_saline])

an_deep_rewdff_saline=np.hstack([xx[1] for xx in deep_rewdff_saline])
deep_rewdff_saline=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_saline])
an_deep_rewdff_drug=np.hstack([xx[1] for xx in deep_rewdff_drug])
deep_rewdff_drug=np.hstack([xx[0][frames_to_show:] for xx in deep_rewdff_drug])

patch_start = int(pre_win_to_show/binsize)
# plot
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
lbls = ['Deep', 'Superficial']
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(7,6), sharex=True)

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
drug = [deep_rewdff_drug, sup_rewdff_drug]
saline = [deep_rewdff_saline, sup_rewdff_saline]
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
fig.suptitle('SNc axons, Excitation (Chrimson)')    
fig.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects'
plt.savefig(os.path.join(savedst, 'per_trial_chrimson_trace.svg'))

#%%
# collect values for ttest
# get subtraction
drug = [deep_rewdff_drug, sp_rewdff_drug,sr_rewdff_drug,slm_rewdff_drug]
saline = [deep_rewdff_saline, sp_rewdff_saline,sr_rewdff_saline,slm_rewdff_saline]

andrug = [an_deep_rewdff_drug, an_sp_rewdff_drug,an_sr_rewdff_drug,an_slm_rewdff_drug]
ansaline = [an_deep_rewdff_saline, an_sp_rewdff_saline,an_sr_rewdff_saline,an_slm_rewdff_saline]
start_frame = int(range_val/binsize-frames_to_show)

save = []
for i in range(4): # deep vs. sup
   # subtract entire trace
   rewcond_h = np.array([xx-np.nanmean(drug[i],axis=1) for xx in saline[i].T]).T 
   stimdff_h = np.nanmean(rewcond_h[start_frame:start_frame+int(stimsec/binsize)],axis=0)    
   stimdff_h[np.isnan(stimdff_h)]=0
   t,pval = scipy.stats.ttest_1samp(stimdff_h, popmean=0)
   save.append([stimdff_h, pval, ansaline[i]])    
# superficial vs. deep
deep_rewcond_h = np.array([xx-np.nanmean(drug[0],axis=1) for xx in saline[0].T]).T 
sp_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[1].T]).T 
sr_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[2].T]).T 
slm_rewcond_h = np.array([xx-np.nanmean(drug[1],axis=1) for xx in saline[3].T]).T 
deep_stimdff_h = np.nanmean(deep_rewcond_h[start_frame:start_frame+int(stimsec/binsize)],axis=0)
sp_stimdff_h = np.nanmean(sp_rewcond_h[start_frame:start_frame+int(stimsec/binsize)],axis=0)

t,pval_deep_vs_sup = scipy.stats.ranksums(deep_stimdff_h[~np.isnan(deep_stimdff_h)], sp_stimdff_h[~np.isnan(sp_stimdff_h)])
#%%
lbls = ['SLM','SR','SP','SO']
lbls=np.array(lbls)[::-1]

plt.rc('font', size=16)
dfs = []
for pln in range(4):
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
fig,ax = plt.subplots(figsize=(3,4))
# pink and grey
g=sns.boxplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdf,fill=False,order=lbls,
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
    ax.text(i, y, f'p={pval:.2g}, \n{len(trials)} trials', ha='center', fontsize=fs, rotation=45)
    i+=1
ax.set_title('n=trials',pad=40,fontsize=14)
plt.savefig(os.path.join(savedst, 'per_trial_snc_chrims_quant.svg'))

#%%
# per animal 
# only take trials >0?
# bigdf=bigdf[bigdf.mean_dff_during_stim>0]
bigdfan = bigdf.groupby(['animal', 'plane_subgroup']).mean(numeric_only=True).reset_index()
bigdfan['mean_dff_during_stim']=bigdfan['mean_dff_during_stim']*100
# # # Specify the desired order
# bigdfan=bigdfan[bigdfan.animal!='e292']
fig,ax = plt.subplots(figsize=(4,5))
g=sns.barplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,fill=False,order=lbls,palette='Dark2',
        errorbar='se',ax=ax)

sns.stripplot(x='plane_subgroup',y='mean_dff_during_stim',hue='plane_subgroup',data=bigdfan,order=lbls,palette='Dark2',
        s=10,alpha=0.8,ax=ax)
ax.spines[['top','right']].set_visible(False)
# Connect lines per animal
for animal, subdf in bigdfan.groupby('animal'):
    # Sort by x-position to connect in the right order
    subdf = subdf.set_index('plane_subgroup').loc[lbls].reset_index()
    ax.plot(
        range(len(subdf)),  # x positions
        subdf['mean_dff_during_stim'],
        color='gray', alpha=0.5, linewidth=1.5
    )

# ax.legend(bbox_to_anchor=(1.01, 1.05),fontsize=12)
ax.set_ylabel('Mean % $\Delta F/F$ during stim.')
ax.set_xlabel('')

y=0.3
fs=14
i=0
from statsmodels.stats.multitest import multipletests

# Collect all p-values first
pvals = []
for lbl in lbls:
    halo = bigdfan.loc[bigdfan.plane_subgroup == lbl, 'mean_dff_during_stim'].values
    _, pval = scipy.stats.ttest_1samp(halo, popmean=0)
    pvals.append(pval)

# Apply correction (Bonferroni example)
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# Annotate corrected p-values
for i, pval_corr in enumerate(pvals_corrected):
    ax.text(i, y, f'p={pval_corr:.2g}', ha='center', fontsize=12, rotation=45)
fig.suptitle('SNc axons, Chrimson')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'per_an_snc_chrimson_quant.svg'))
