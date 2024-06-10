#%%
import numpy as np ,scipy, random, seaborn as sns
import pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ranksums
from placecell import calculate_global_remapping, get_place_field_widths
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8

# Load place cell activity data
# The data should be a DataFrame with columns representing different neurons
# and rows representing firing rates at different locations for different conditions
conddf = pd.read_csv(r"Z:\condition_df\conddf_neural_modeling.csv", index_col=None)
#%%
results = []
for dd in range(len(conddf)):
    animal = conddf.animals.values[dd]    
    day = conddf.days.values[dd]        
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane0_Fall.mat"
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'tuning_curves_early_trials',\
        'tuning_curves_late_trials', 'coms_early_trials'])
    coms = fall['coms'][0]
    coms_early = fall['coms_early_trials'][0]
    tcs_early = fall['tuning_curves_early_trials'][0]
    tcs_late = fall['tuning_curves_late_trials'][0]
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    eptest = conddf.optoep.values[dd]   
    print(f'\n*******animal={animal}, day={day}, optoep={eptest}*******') 
    eps = np.where(changeRewLoc>0)[0]
    rewlocs = changeRewLoc[eps]*1.5
    eps = np.append(eps, len(changeRewLoc))  
    if conddf.optoep.values[dd]<2: 
        eptest = random.randint(2,3)      
        if len(eps)<4: eptest = 2 # if no 3 epochs
    comp = [eptest-2,eptest-1] # eps to compare    

    tc1 = tcs_late[comp[0]]
    tc2 = tcs_late[comp[1]]

    # Calculate the correlation between place fields for the two reward conditions
    data_reward1, data_reward2 = tc1, tc2
    # global remapping calculation
    P, H, real_distribution, shuffled_distribution, \
    p_values, global_remapping = calculate_global_remapping(data_reward1, data_reward2)
    pc_widths_prev = get_place_field_widths(tc1, threshold=0.3)
    pc_widths_opto = get_place_field_widths(tc2, threshold=0.3)
    print(f"Rank-sum test p-value: {P:.4f}")
    print(f"Percentage of neurons showing global remapping: {np.sum(global_remapping) / len(global_remapping) * 100:.2f}%")
    # Save results
    results.append([real_distribution, p_values, P, pc_widths_prev, pc_widths_opto])

#%%
df = conddf
df['global_remap_ranksum_pvalue'] = [xx[2] for xx in results]
df['opto'] = conddf.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in conddf.in_type.values]
df['pc_width_opto'] = [np.nanmean(xx[4]) for xx in results]
df['pc_width_diff'] = [np.nanmean(xx[4])-np.nanmean(xx[3]) for xx in results]
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\figure_data\vip_inhibition'
df = df[(df.animals!='e190')&(df.animals!='e189')&(df.animals!='e217')]
dfagg = df#.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)
plt.figure(figsize=(3.5,6))
ax = sns.barplot(x="opto", y="global_remap_ranksum_pvalue",hue='condition', data=dfagg,
    palette={'ctrl': "slategray", 'vip': "red"},                
            errorbar='se', fill=False)
sns.stripplot(x="opto", y="global_remap_ranksum_pvalue",hue='condition', data=dfagg,
            palette={'ctrl': 'slategray','vip': "red"},                
            s=10, dodge=True, jitter=True)
ax.axhline(y=0.05, color='k',linestyle='--')
ax.tick_params(axis='x', labelrotation=90)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
# plt.savefig(os.path.join(savedst,'remap_rate_vipvctrl .jpg'), bbox_inches='tight')
plt.figure(figsize=(8,6))
ax = sns.barplot(x="opto", y="global_remap_ranksum_pvalue",hue='in_type', data=dfagg,                
            errorbar='se', fill=False)
sns.stripplot(x="opto", y="global_remap_ranksum_pvalue",hue='in_type', data=dfagg,                
            s=10, dodge=True, jitter=True)
ax.axhline(y=0.05, color='k',linestyle='--')
ax.tick_params(axis='x', labelrotation=90)
ax.spines[['top','right']].set_visible(False)
# ax.get_legend().set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.savefig(os.path.join(savedst,'remap_rate_intype.jpg'), bbox_inches='tight')
#%%
dfagg = df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True)

plt.figure(figsize=(3.5,6))
ax = sns.barplot(x="opto", y="pc_width_opto",hue='condition', data=dfagg,                
            errorbar='se', fill=False,
            palette={'ctrl': "slategray", 'vip': "red"})
sns.stripplot(x="opto", y="pc_width_opto",hue='condition', data=dfagg,                
            s=10, dodge=True, jitter=True,
            palette={'ctrl': "slategray", 'vip': "red"})
ax.axhline(y=0.05, color='k',linestyle='--')
ax.tick_params(axis='x', labelrotation=90)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().set_visible(False)
plt.savefig(os.path.join(savedst,'pc_width.jpg'), bbox_inches='tight')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


# look at distributions
df = conddf
df['global_remap_ranksum_pvalue'] = [xx[2] for xx in results]
df['opto'] = conddf.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in conddf.in_type.values]
df['pc_width_opto'] = [np.nanmean(xx[4]) for xx in results]
df['pc_width_diff'] = [np.nanmean(xx[4])-np.nanmean(xx[3]) for xx in results]

df['pc_width_opto_dist'] = [xx[4] for xx in results]

