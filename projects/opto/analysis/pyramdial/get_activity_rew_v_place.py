"""
get reward distance cells between opto and non opto conditions
oct 2024
mods in june 2025
control vs. opto epoch only
"""

#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.analysis.pyramdial.placecell import get_rew_place_activity_opto
# import warnings
# warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\vip_paper'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
radian_alignment_saved = {} # overwrite
results_all=[]
cm_window = 20
radian_alignment={}
#%%
# iterate through all animals
for ii in range(len(conddf)):
    day = int(conddf.days.values[ii])
    animal = conddf.animals.values[ii]
    # skip e217 day
    if ii!=179:#(conddf.optoep.values[ii]>1):
        if animal=='e145': pln=2  
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        pre_activity_opto,pre_activity_prev,post_activity_opto,post_activity_prev = get_rew_place_activity_opto(
            params_pth, pdf, radian_alignment_saved, animal, day, ii, conddf, 
            radian_alignment, cm_window=cm_window)
        results_all.append([pre_activity_opto,pre_activity_prev,post_activity_opto,post_activity_prev])

# %%

#%%
# separate out variables
df = conddf.copy()
df = df.drop([179]) # skipped e217 day
pre_activity_opto = [xx[0] for xx in results_all]
pre_activity_prev = [xx[1] for xx in results_all]
post_activity_opto = [xx[2] for xx in results_all]
post_activity_prev = [xx[3] for xx in results_all]
# average
pre_activity_opto = [np.nanmean(xx,axis=0) if len(xx)>0 else [] for xx in pre_activity_opto]
df['cond']=[xx if 'vip' in xx else 'ctrl' for xx in df.in_type.values]
arrs = [df.animals.values, df.days.values, df.optoep.values,df.cond.values]
beh_pre_activity_opto=[]
for arr in arrs:
    beh_pre_activity_opto.append([np.repeat(arr[iii],len(xx)) if len(xx)>0 else [] for iii,xx in enumerate(pre_activity_opto)])
pre_activity_prev = [np.nanmean(xx,axis=0) if len(xx)>0 else [] for xx in pre_activity_prev]
beh_pre_activity_prev=[]
for arr in arrs:
    beh_pre_activity_prev.append([np.repeat(arr[iii],len(xx)) if len(xx)>0 else [] for iii,xx in enumerate(pre_activity_prev)])
post_activity_opto = [np.nanmean(xx,axis=0) if len(xx)>0 else [] for xx in post_activity_opto]
beh_post_activity_opto=[]
for arr in arrs:
    beh_post_activity_opto.append([np.repeat(arr[iii],len(xx)) if len(xx)>0 else [] for iii,xx in enumerate(post_activity_opto)])
post_activity_prev = [np.nanmean(xx,axis=0) if len(xx)>0 else [] for xx in post_activity_prev]
beh_post_activity_prev=[]
for arr in arrs:
    beh_post_activity_prev.append([np.repeat(arr[iii],len(xx)) if len(xx)>0 else [] for iii,xx in enumerate(post_activity_prev)])

#%%
# concat all cell type goal cell prop
activity = [pre_activity_opto,pre_activity_prev,post_activity_opto,post_activity_prev]
an = [beh_pre_activity_opto[0],beh_pre_activity_prev[0],beh_post_activity_opto[0],beh_post_activity_prev[0]]
dys= [beh_pre_activity_opto[1],beh_pre_activity_prev[1],beh_post_activity_opto[1],beh_post_activity_prev[1]]
optoep= [beh_pre_activity_opto[2],beh_pre_activity_prev[2],beh_post_activity_opto[2],beh_post_activity_prev[2]]
intype= [beh_pre_activity_opto[3],beh_pre_activity_prev[3],beh_post_activity_opto[3],beh_post_activity_prev[3]]

act = np.concatenate([np.hstack(cll) for cll in activity])
realdf= pd.DataFrame()
realdf['mean_dff']=act
lbl = ['pre_opto', 'pre_prev', 'post_opto', 'post_prev']
realdf['cell_type']=np.concatenate([[lbl[ii]]*len(np.hstack(cll)) for ii,cll in enumerate(activity)])
realdf['animal']=np.concatenate([np.hstack(a) for a in an])
realdf['optoep']=np.concatenate([np.hstack(a) for a in optoep])
realdf['opto']=[True if xx>1 else False if xx<1 else np.nan for xx in realdf['optoep']]
realdf['condition']=np.concatenate([np.hstack(a) for a in intype])
realdf['condition']=[xx if 'vip' in xx else 'ctrl' for xx in realdf.condition.values]
realdf['day']=np.concatenate([np.hstack(a) for a in dys])
# todo: subtract opto - prev
realdf=realdf[realdf['mean_dff']>0]
realdf=realdf[(realdf.animal!='e189')&(realdf.animal!='e200')&(realdf.animal!='e190')]
# remove outlier days
realdf=realdf.drop([715,705,737,526,516,548,416,605])
dfagg = realdf.groupby(['animal', 'opto', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
fig,axes=plt.subplots(ncols=4,figsize=(16,5),sharey=True,sharex=True,)
for cl,cll in enumerate(dfagg.cell_type.unique()):
    ax=axes[cl]
    sns.barplot(x='condition',y='mean_dff',hue='opto',data=dfagg[dfagg.cell_type==cll],fill=False,ax=ax)
    ax.set_title(cll)
#%%
pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
activity_pairs = [
    ('pre', pre_activity_opto, pre_activity_prev,
     beh_pre_activity_opto, beh_pre_activity_prev),
    ('post', post_activity_opto, post_activity_prev,
     beh_post_activity_opto, beh_post_activity_prev)
]
diff_df = []

for label, opto_act_list, prev_act_list, opto_meta, prev_meta in activity_pairs:
    # Average each cell's activity over trials
    # Subtract mean traces
    diff = [np.array(opto_mean)-np.array(prev_act_list[jjj]) if len(opto_mean)==len(prev_act_list[jjj]) else [] for jjj, opto_mean in enumerate(opto_act_list)]  # shape: (n_cells, n_timepoints)
    diff_mean = np.hstack(diff)
    # Metadata
    animals = [an if len(opto_act_list[jjj])==len(prev_act_list[jjj]) else [] for jjj, an in enumerate(opto_meta[0])]
    animals = np.hstack(animals)  # animal names
    days = [an if len(opto_act_list[jjj])==len(prev_act_list[jjj]) else [] for jjj, an in enumerate(opto_meta[1])]
    days = np.hstack(days)     # day index
    optoep = [an if len(opto_act_list[jjj])==len(prev_act_list[jjj]) else [] for jjj, an in enumerate(opto_meta[2])]
    optoep = np.hstack(optoep)   # opto epoch
    condition = [an if len(opto_act_list[jjj])==len(prev_act_list[jjj]) else [] for jjj, an in enumerate(opto_meta[3])]
    condition = np.hstack(condition)  # condition
    df = pd.DataFrame({
        'mean_dff_diff': diff_mean,
        'animal': animals,
        'day': days,
        'optoep': optoep,
        'condition': condition,
        'cell_type': label
    })
    diff_df.append(df)

# Combine into single DataFrame
diff_df = pd.concat(diff_df, ignore_index=True)

# Optional filtering
diff_df = diff_df[(diff_df.animal != 'e189') &
                  (diff_df.animal != 'e200') &
                  (diff_df.animal != 'e190')]
diff_df = diff_df.drop([715, 705, 737, 526, 516, 548, 416, 605], errors='ignore')

# Aggregate for plotting
dfagg_diff = diff_df.groupby(['animal', 'cell_type', 'condition']).mean(numeric_only=True).reset_index()
# Plotting
fig, ax = plt.subplots(figsize=(5,5))
sns.barplot(data=dfagg_diff, x='cell_type', y='mean_dff_diff', hue='condition', fill=False, ax=ax, palette=pl,legend=False)
s=12;a=0.7
sns.stripplot(data=dfagg_diff, x='cell_type', y='mean_dff_diff', hue='condition', s=s,alpha=a, ax=ax, palette=pl,dodge=True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axhline(0, color='gray', linestyle='--')
ax.set_ylabel('Mean ΔF/F (LEDon-LEDoff)')
ax.set_title('Average Activity Difference per Cell Type')
sns.despine()