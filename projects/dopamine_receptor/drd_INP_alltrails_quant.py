"""
zahra
sept 2024
"""
#%%
import os, sys, scipy, imageio, pandas as pd, re
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf

# Add custom path for MATLAB functions
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') 

# Formatting for figures
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)
from projects.dopamine_receptor.drd import extract_plane_number
from scipy.io import loadmat
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# Define save path for PDF
condition = 'drd2ko'

# fluorescence mean threshold
fluor_thres = 100
# Define source directory and mouse name
src = r'Y:\drd'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'
if condition=='drd1':
    mice = ['e255', 'e254']
    days_s = [list(np.arange(3,13)), list(np.arange(1,9))]
elif condition=='drd2':
    mice = ['e256', 'e253']
    days_s = [list(np.arange(3,16)), list(np.arange(1,10))]
elif condition=='drd2ko':
    mice = ['e261','e262']
    days_s = [np.arange(1,7),[1,2,3,4,5,6,7,8,9]]
days_to_analyse=2
# days = [3,4,5,6,7,8,9,10,12]
range_val, binsize = 6 , 0.2 # seconds
d1_postrew_dff_all_mice = []
# Iterate through specified days
for ii,mouse_name in enumerate(mice):
    days = days_s[ii]
    postrew_dff_all_days = []
    for dy in days[-days_to_analyse:]: # only last 2 days for now
        day_dir = os.path.join(src, mouse_name, str(dy))
        postrew_dff_all_planes = []
        for root, dirs, files in os.walk(day_dir):
            for file in files:
                if 'plane' in root and file.endswith('roibyclick_F.mat'):
                    f = loadmat(os.path.join(root, file))
                    print(os.path.join(root, file))
                    plane = extract_plane_number(os.path.join(root, file))
                    
                    eps = np.where(f['changeRewLoc'] > 0)[1]
                    eps = np.append(eps, len(f['changeRewLoc'][0]))
                    rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                    dFF_iscell = f['dFF']
                    F_iscell = f['F']
                    means = np.nanmean(F_iscell, axis=0)
                    # remove dim cells
                    dFF_iscell = dFF_iscell[:, means>fluor_thres]
                    dFF_iscell_filtered = dFF_iscell.T
                    dff_res = []
                    perirew = []
                    if len(dFF_iscell_filtered)>0:
                        # Iterate through cells
                        for cll in range(dFF_iscell_filtered.shape[0]):
                            X = np.array([f['forwardvel'][0]]).T 
                            X = sm.add_constant(X)
                            y = dFF_iscell_filtered[cll, :]
                            
                            model = sm.GLM(y, X, family=sm.families.Gaussian())
                            result = model.fit()
                            dff_res.append(result.resid_pearson)
                            
                            dff = result.resid_pearson
                            dffdf = pd.DataFrame({'dff': dff})
                            dff = np.hstack(dffdf.rolling(5).mean().values)
                        
                            normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
                                dff, 
                                f['solenoid2'][0].astype(int), 
                                f['timedFF'][0], 
                                f['trialnum'][0], 
                                range_val, 
                                binsize,
                            )
                            perirew.append([meanrewdFF, rewdFF])
                    
                        dff_res = np.array(dff_res)
                        # normalize post reward activity
                        clls = dff_res.shape[0]
                        prewin = 1
                        binss = np.ceil(prewin/binsize).astype(int)
                        bound = int(range_val/binsize)
                        postwin = 2 #s
                        postbound = np.ceil(postwin/binsize).astype(int)
                        meanrewall = np.array([perirew[cll][0]-np.nanmean(perirew[cll][0][(bound-binss):bound]) for cll in range(clls)])
                        if condition=='drd2': #or condition=='drd2ko'
                            postrew_dff = np.nanmean(meanrewall[:, bound:bound+postbound],axis=1)
                        # # or quantile
                        else:
                            postrew_dff = np.nanquantile(meanrewall[:, bound:bound+postbound], 
                                .75, axis=1)
                        # area under curve
                        # postrew_dff = [np.trapz(xx, dx=5) for xx in meanrewall]
                        postrew_dff_all_planes.append(postrew_dff)
                    
                    
        postrew_dff_all_days.append(postrew_dff_all_planes)
    d1_postrew_dff_all_mice.append(postrew_dff_all_days)
#%%
# separate d2
# Define save path for PDF
condition = 'drd2'

# fluorescence mean threshold
fluor_thres = 100
# Define source directory and mouse name
src = r'Y:\drd'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'
if condition=='drd1':
    mice = ['e255', 'e254']
    days_s = [list(np.arange(3,13)), list(np.arange(1,9))]
elif condition=='drd2':
    mice = ['e256', 'e253']
    days_s = [list(np.arange(3,16)), list(np.arange(1,10))]
elif condition=='drd2ko':
    mice = ['e261','e262']
    days_s = [[1,2,3,4],[1,2,3,4,5,6,7,8,9]]
    fluor_thres = 600
days_to_analyse=2
# days = [3,4,5,6,7,8,9,10,12]
range_val, binsize = 6 , 0.2 # seconds
d2_postrew_dff_all_mice = []
# Iterate through specified days
for ii,mouse_name in enumerate(mice):
    days = days_s[ii]
    postrew_dff_all_days = []
    for dy in days[-days_to_analyse:]: # only last 2 days for now
        day_dir = os.path.join(src, mouse_name, str(dy))
        postrew_dff_all_planes = []
        for root, dirs, files in os.walk(day_dir):
            for file in files:
                if 'plane' in root and file.endswith('roibyclick_F.mat'):
                    f = loadmat(os.path.join(root, file))
                    print(os.path.join(root, file))
                    plane = extract_plane_number(os.path.join(root, file))
                    
                    eps = np.where(f['changeRewLoc'] > 0)[1]
                    eps = np.append(eps, len(f['changeRewLoc'][0]))
                    rewlocs = f['changeRewLoc'][0][f['changeRewLoc'][0] > 0]

                    dFF_iscell = f['dFF']
                    F_iscell = f['F']
                    means = np.nanmean(F_iscell, axis=0)
                    # remove dim cells
                    dFF_iscell = dFF_iscell[:, means>fluor_thres]
                    dFF_iscell_filtered = dFF_iscell.T
                    dff_res = []
                    perirew = []
                    if len(dFF_iscell_filtered)>0:
                        # Iterate through cells
                        for cll in range(dFF_iscell_filtered.shape[0]):
                            X = np.array([f['forwardvel'][0]]).T 
                            X = sm.add_constant(X)
                            y = dFF_iscell_filtered[cll, :]
                            
                            model = sm.GLM(y, X, family=sm.families.Gaussian())
                            result = model.fit()
                            dff_res.append(result.resid_pearson)
                            
                            dff = result.resid_pearson
                            dffdf = pd.DataFrame({'dff': dff})
                            dff = np.hstack(dffdf.rolling(5).mean().values)
                        
                            normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF = perireward_binned_activity(
                                dff, 
                                f['solenoid2'][0].astype(int), 
                                f['timedFF'][0], 
                                f['trialnum'][0], 
                                range_val, 
                                binsize,
                            )
                            perirew.append([meanrewdFF, rewdFF])
                    
                        dff_res = np.array(dff_res)
                        # normalize post reward activity
                        clls = dff_res.shape[0]
                        prewin = 1
                        binss = np.ceil(prewin/binsize).astype(int)
                        bound = int(range_val/binsize)
                        postwin = 2 #s
                        postbound = np.ceil(postwin/binsize).astype(int)
                        meanrewall = np.array([perirew[cll][0]-np.nanmean(perirew[cll][0][(bound-binss):bound]) for cll in range(clls)])
                        if condition=='drd2': #or condition=='drd2ko'
                            postrew_dff = np.nanmean(meanrewall[:, bound:bound+postbound],axis=1)
                        # # or quantile
                        else:
                            postrew_dff = np.nanquantile(meanrewall[:, bound:bound+postbound], 
                                .75, axis=1)
                        # area under curve
                        # postrew_dff = [np.trapz(xx, dx=5) for xx in meanrewall]
                        postrew_dff_all_planes.append(postrew_dff)
                    
                    
        postrew_dff_all_days.append(postrew_dff_all_planes)
    d2_postrew_dff_all_mice.append(postrew_dff_all_days)
#%%    
# per day histogram for fig
condition='d2ko'
import seaborn as sns

cols=sns.color_palette("colorblind") # Set the global palette

arr = np.concatenate([np.concatenate([np.concatenate(pr) for pr in dy if len(pr)>0]) for dy \
    in d1_postrew_dff_all_mice])
arr = arr[arr<10] # remove outliers
bins=np.histogram(arr, bins=30)[1] #get the bin edges
plns = 4

# concatenate across mice
pr = [np.concatenate([np.concatenate(pr) for pr in pr_dy]) for pr_dy in d1_postrew_dff_all_mice]
allplnpr = np.concatenate(pr)

fig,ax = plt.subplots(figsize = (7,5))
# allplnpr = allplnpr[allplnpr<5]
ax.hist(allplnpr,bins=bins,color=cols[2])        
ax.axvline(0, color='slategray', linestyle='--', linewidth=6)
ax.spines[['top','right']].set_visible(False)                        
ax.set_xlim([min(allplnpr)-.2, max(allplnpr)+.2])            

# ax.legend(bbox_to_anchor=(1.01, 1.01))
ax.set_xlabel('Post-reward $\Delta$ F/F-Baseline')
ax.set_ylabel('# cells')        

fig.suptitle(f'{condition}')
fig.tight_layout()
plt.savefig(os.path.join(savedst, f'{condition}_histogram.svg'), 
    bbox_inches='tight')
#%%
condition='d2'
arr = np.concatenate([np.concatenate([np.concatenate(pr) for pr in dy if len(pr)>0]) for dy \
    in d2_postrew_dff_all_mice])
arr = arr[arr<10] # remove outliers
binsd2=np.histogram(arr, bins=50)[1] #get the bin edges
plns = 4

# concatenate across mice
pr = [np.concatenate(pr_dy[-2]) for pr_dy in d2_postrew_dff_all_mice]
allplnprd2 = np.concatenate(pr)

fig,ax = plt.subplots(figsize = (7,5))
# allplnpr = allplnpr[allplnpr<5]
ax.hist(allplnprd2,bins=binsd2,color=cols[1])        
ax.axvline(0, color='slategray', linestyle='--', linewidth=6)
ax.spines[['top','right']].set_visible(False)                        
ax.set_xlim([min(allplnprd2)-.2, max(allplnprd2)+.2])            
# ax.legend(bbox_to_anchor=(1.01, 1.01))
ax.set_xlabel('Post-reward $\Delta$ F/F-Baseline')
ax.set_ylabel('# cells')        

fig.suptitle(f'{condition}')
fig.tight_layout()
plt.savefig(os.path.join(savedst, f'{condition}_histogram.svg'), 
    bbox_inches='tight')

#%%
# across conditions
fig,ax = plt.subplots(figsize = (7,5))

# concatenate across mice
ax.hist(allplnpr,bins=bins, label='D2 CRISPR-KO',alpha=0.85,color=cols[2])        
ax.hist(allplnprd2,bins=binsd2, label='D2', alpha=0.85,color=cols[1])
ax.axvline(0, color='slategray', linestyle='--', linewidth=6)
ax.spines[['top','right']].set_visible(False)                        
ax.set_xlim([min(allplnprd2)-.2, max(allplnpr)+.2])            
ax.legend(bbox_to_anchor=(.6, 1.0))
ax.set_xlabel('Post-reward $\Delta$ F/F-Baseline')
ax.set_ylabel('# cells')        

fig.tight_layout()
plt.savefig(os.path.join(savedst, f'd2_d2ko_histogram.svg'), 
    bbox_inches='tight')

#%%
# histogram of post rew activity per plane
# arr = np.concatenate([np.concatenate([np.concatenate(pr) for pr in dy if len(pr)>0]) for dy in postrew_dff_all_mice])
# # arr = arr[arr<5] # remove outliers
# bins=np.histogram(arr, bins=20)[1] #get the bin edges
# plns = 4

# for dd, pr_dy in enumerate(postrew_dff_all_mice):
#     fig,axes = plt.subplots(nrows=plns, sharex=True, figsize = (14,12))
#     for d,pr in enumerate(pr_dy):
#         for pln in range(plns):
#             ax =axes[pln]
#             try:
#                 pr[pln] = pr[pln][pr[pln]<5] # remove outliers
#                 ax.hist(pr[pln],bins=bins,label = f'Day {d:02d}', alpha=0.4)
#                 ax.set_title(f'Plane {pln}')
#                 ax.axvline(0, color='slategray', linestyle='--', linewidth=4)
#                 ax.spines[['top','right']].set_visible(False)                        
#                 ax.set_xlim([-1, 1])        
#             except Exception as e:
#                 print(e)
#             if pln==3: ax.legend(bbox_to_anchor=(1.01, 1.01))
                
#         ax.set_xlabel('Mean $\Delta$ F/F-Baseline(pre-reward)')
#         ax.set_ylabel('# cells')      
#     fig.suptitle(f'{condition} \n Post reward activity; mouse {mice[dd]}\nplanes (0: superficial; 4: deep)')
#     fig.tight_layout()

#%%
# all planes
# histogram of post rew activity per plane
# early vs. late days
# arr = np.concatenate([np.concatenate([np.concatenate(pr) for pr in dy if len(pr)>0]) for dy in postrew_dff_all_mice])
# # arr = arr[arr<5] # remove outliers
# bins=np.histogram(arr, bins=40)[1] #get the bin edges
# plns = 4
# for dd, pr_dy in enumerate(postrew_dff_all_mice):
#     fig,axes = plt.subplots(ncols =2,figsize = (15,8))
#     for d,pr in enumerate(pr_dy):
#         if d <= 3:
#             ax = axes[0]
#         else: 
#             ax = axes[1]
#         if len(pr)>0:
#             allplnpr = np.concatenate(pr)
            
#             ax.hist(allplnpr,bins=bins,label = f'Day {d+1:02d}', alpha=0.6)        
#             ax.axvline(0, color='slategray', linestyle='--', linewidth=4)
#             ax.spines[['top','right']].set_visible(False)                        
#             ax.set_xlim([-1, 1])            
#             ax.legend(bbox_to_anchor=(1.01, 1.01))
#     ax.set_xlabel('Mean $\Delta$ F/F-Baseline(pre-reward)')
#     ax.set_ylabel('# cells')        
#     fig.suptitle(f'{condition} \n Post reward activity; mouse {mice[dd]}, all planes\n\
#                 Early vs. late days')
#     fig.tight_layout()
#     plt.savefig(os.path.join(savedst, f'{condition}_{mice[dd]}histogram.svg'), bbox_inches='tight')
