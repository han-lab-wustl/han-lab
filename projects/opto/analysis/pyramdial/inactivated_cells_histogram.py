
"""
zahra's analysis for initial com and enrichment of pyramidal cell data
updated aug 2024
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math,  matplotlib as mpl, matplotlib.backends.backend_pdf
from placecell import get_pyr_metrics_opto, get_dff_opto, get_rewzones, consecutive_stretch, get_inactivated_cells_hist
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8 
mpl.rcParams["ytick.major.size"] = 8
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_com_inference.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
#%% - re-run dct making
dcts = []
for dd,day in enumerate(conddf.days.values):
    # define threshold to detect activation/inactivation
    if dd%10==0: print(f'{dd}/{len(conddf)}')
    dct = get_inactivated_cells_hist(dd, day, conddf)
    dcts.append(dct)
# save pickle of dcts
# saved new version for r21 2/25/25
# with open(r'Z:\dcts_com_opto_inference_wcomp.p', "wb") as fp:   #Pickling
#     pickle.dump(dcts, fp)   
    
#%%
# get inactivated cells distribution
diffs_nonopto = []; diffs_opto = []; animals_nonopto = []; days_nonopto = []
animals_opto = []; days_opto = []
for ii,dct in enumerate(dcts):
    try:
        if conddf.optoep.values[ii]<2:
            diff_nonopto = dct['activity_diff'][dct['skew']>2]
            diffs_nonopto.append(diff_nonopto)
            animals_nonopto.append(conddf.animals.values[ii])
            days_nonopto.append(conddf.days.values[ii])
        else:
            diff_opto = dct['activity_diff'][dct['skew']>2]
            diffs_opto.append(diff_opto)
            animals_opto.append(conddf.animals.values[ii])
            days_opto.append(conddf.days.values[ii])
    except Exception as e:
        print(e)


#%%
df = pd.DataFrame()
df['diffs'] = np.concatenate([np.concatenate(diffs_nonopto),np.concatenate(diffs_opto)])
df['condition'] = np.concatenate([['no_stim']*len(np.concatenate(diffs_nonopto)),
                                ['stim']*len(np.concatenate(diffs_opto))])
df['animal'] = np.concatenate([np.concatenate([[animals_nonopto[ii]]*len(diffs_nonopto[ii]) for ii in range(len(animals_nonopto))]),
                    np.concatenate([[animals_opto[ii]]*len(diffs_opto[ii]) for ii in range(len(animals_opto))])])


#%%
plt.rc('font', size=20)          # controls default text sizes

fig, ax = plt.subplots()
dfplt = df[((df.animal!='e216') | (df.animal!='e217') |(df.animal!='e218')) & (df.condition=='stim')]
sns.histplot(x='diffs',hue='condition',data=dfplt,fill=False,linewidth=1,
            palette={'no_stim': 'slategray', 'stim': 'coral'})
ax.set_xlim([-1,30])
ax.axvline(color='k', linestyle='--', linewidth=3)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('$\Delta F/F$ LEDoff-on')
ax.set_ylabel('Cell Count')
plt.savefig(os.path.join(savedst, 'ctrl_inactive_cells_stim.svg'), bbox_inches='tight')

fig, ax = plt.subplots()
dfplt = df[((df.animal!='e216') | (df.animal!='e217') |(df.animal!='e218')) & (df.condition=='no_stim')]
sns.histplot(x='diffs',hue='condition',data=dfplt,fill=False,linewidth=1,
            palette={'no_stim': 'slategray', 'stim': 'coral'})
ax.set_xlim([-1,30])
ax.axvline(color='k', linestyle='--', linewidth=3)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('$\Delta F/F$ LEDoff-on')
ax.set_ylabel('Cell Count')
plt.savefig(os.path.join(savedst, 'ctrl_inactive_cells_nostim.svg'), bbox_inches='tight')


fig, ax = plt.subplots()
dfplt = df[((df.animal!='e216') | (df.animal!='e217') |(df.animal!='e218'))]
sns.histplot(x='diffs',hue='condition',data=dfplt,fill=False,linewidth=1,
            palette={'no_stim': 'slategray', 'stim': 'coral'})
ax.set_xlim([-1,30])
ax.axvline(color='k', linestyle='--', linewidth=3)
ax.axvline(x=5, color='grey', linestyle='--', linewidth=3)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('$\Delta F/F$ LEDoff-on')
ax.set_ylabel('Cell Count')
plt.savefig(os.path.join(savedst, 'ctrl_inactive_cells_overlay.svg'), bbox_inches='tight')

# ax.set_xlim([-10, 10])
