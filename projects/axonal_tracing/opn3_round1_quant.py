#%%
import pandas as pd, seaborn as sns, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes

pth = r"D:\maggie_opn3_quant.csv"
dst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects\axonal_labeling'
dforg = pd.read_csv(pth)
# concat columns
# plane colors
# planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
palette=[np.array([230, 84, 128])/255,np.array([204, 164, 61])/255,np.array([0,255,0])/255,np.array([0, 0, 255])/255]
df=pd.DataFrame()
df['hipp_layer'] = np.concatenate([['SO']*len(dforg)*2,['SP']*len(dforg)*2,['SR']*len(dforg)*2,['SLM']*len(dforg)*2])
df['stain'] = np.concatenate([np.concatenate([['eOPN3-mScarlet']*len(dforg),['TH']*len(dforg)])]*4)
df['num_axons'] = pd.concat([dforg['cy3_so'], dforg['th_so'],dforg['cy3_sp'],dforg['th_sp'],dforg['cy3_sr'],dforg['th_sr'],dforg['cy3_slm'],dforg['th_slm']], ignore_index=True)
# 4 layers
df['injection'] = np.concatenate([dforg.injection.values]*8)
df['animal'] = np.concatenate([dforg.animal.values]*8)
df['injection'] = ['SNc' if xx=='snc' else 'VTA' for xx in df.injection]

order = ['SO','SP','SR','SLM']
df=df[df.num_axons>0]
dfan = df.groupby(['animal','injection','hipp_layer']).mean(numeric_only=True)
s=10
a=0.4
fig,ax=plt.subplots(figsize=(5,4))
sns.stripplot(x='injection',y='num_axons',hue='hipp_layer',hue_order=order,data=df,dodge=True,jitter=True,s=s,alpha=a,palette=palette)
# sns.stripplot(x='injection',y='num_axons',hue='hipp_layer',data=dfan,dodge=True,jitter=True,s=14,hue_order=order,palette=palette)
dfsum = df.groupby(['injection', 'hipp_layer'])['num_axons'].sum().reset_index()
sns.barplot(x='injection',y='num_axons',hue='hipp_layer',data=dfsum,hue_order=order,palette=palette,errorbar='se',fill=False,legend=False)
ax.legend(title='Layer', bbox_to_anchor=(.8, 1), loc='upper left', borderaxespad=0.)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(['SNc','VTA'])
ax.set_ylabel('Total # of axons')
ax.set_xlabel('Injection site')

# Map each bar's center using positions per group (e.g., SNc and VTA) and per hue (SO, SP, ...)
group_positions = {'SNc': 0, 'VTA': 1}
n_hues = len(order)
group_width = 0.8  # default total width of a group in seaborn
bar_width = group_width / n_hues
for kk, inj in enumerate(['SNc', 'VTA']):
    base_x = group_positions[inj] - group_width / 2 + bar_width / 2  # starting offset
    for ll, ly in enumerate(order):
        xpos = base_x + ll * bar_width
        count = len(df[(df['hipp_layer'] == ly) &
                       (df['injection'] == inj) &
                       (df['stain'] == "eOPN3-mScarlet")])
        ax.text(xpos, 50, f'{count}', ha='center', fontsize=14)

ax.set_title('n=1 animal    n=2 animals')
plt.savefig(os.path.join(dst, 'opn3_labeling.svg'),bbox_inches='tight')

#%%
# Sum axons by stain
# Group by injection site and stain, then sum axons
axon_counts_inj = df.groupby(['injection', 'stain'])['num_axons'].sum().unstack()

# Plot pie chart per injection site
colors = ['yellowgreen', 'tomato']  # TH+, TH−
labels = ['TH+', 'TH−']

fig, axs = plt.subplots(1, len(axon_counts_inj), figsize=(5 * len(axon_counts_inj), 4))

if len(axon_counts_inj) == 1:
    axs = [axs]  # ensure iterable

for ax, (inj, row) in zip(axs, axon_counts_inj.iterrows()):
    th_pos = row['TH']
    opn3_total = row['eOPN3-mScarlet']
    th_neg = opn3_total - th_pos

    ax.pie([th_pos, th_neg], labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title(f'{inj}')
    ax.axis('equal')

fig.suptitle('TH+ eOPN3 axons by injection site')
plt.tight_layout()
plt.savefig(os.path.join(dst, 'th_opn3_pie.svg'),bbox_inches='tight')
#%%
df['percent_axons'] = df[df.stain=='eOPN3-mScarlet'].groupby(['animal'])['num_axons'].transform(lambda x: 100 * x / x.sum())
# Plot
horder = ['SO', 'SP', 'SR', 'SLM']
s = 12
a = 0.7
# Filter for eOPN3-mScarlet only
df_filtered = df[df['stain'] == 'eOPN3-mScarlet']

# Compute mean ± SEM across animals per injection × layer
grouped = df_filtered.groupby(['animal', 'injection', 'hipp_layer'])['percent_axons'].mean().reset_index()
summary = grouped.groupby(['injection', 'hipp_layer'])['percent_axons'].agg(['mean', 'sem']).reset_index()
order=['SNc', 'VTA']
# Plot
fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(
    x='injection', y='percent_axons', hue='hipp_layer',order=order,
    data=grouped, hue_order=horder,palette=palette,fill=False,legend=False
)
sns.stripplot(
    x='injection', y='percent_axons', hue='hipp_layer',order=order,
    data=grouped, hue_order=horder,palette=palette,s=s,dodge=True
)
sns.stripplot(
    x='injection', y='percent_axons', hue='hipp_layer',order=order,
    data=df, hue_order=horder,palette=palette,s=8,alpha=a,dodge=True,
    legend=False
)
# Clean up
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('% Axons per layer')
ax.set_xlabel('Injection site')
plt.tight_layout()
plt.savefig(os.path.join(dst, 'opn3_layer_percent_avg.svg'), bbox_inches='tight')
#%%
# Filter to eOPN3-mScarlet only
df_filtered = df[df['stain'] == 'eOPN3-mScarlet']

# Sum % axons per animal × layer
df_sum = df_filtered.groupby(['animal', 'hipp_layer'])['percent_axons'].sum().unstack()

# Compute SP/SO ratio
df_sum['sp_so_ratio'] = df_sum['SO'] / df_sum['SP']

# Merge with injection info (once per animal)
animal_injection = df_filtered[['animal', 'injection']].drop_duplicates()
df_ratios = df_sum[['sp_so_ratio']].reset_index().merge(animal_injection, on='animal')

color=np.array([160, 100, 128])/255
print(df_ratios)
fig, ax = plt.subplots(figsize=(3,5))
sns.stripplot(data=df_ratios, x='injection', y='sp_so_ratio', jitter=True, size=12, alpha=0.7,order=order,color=color)
sns.barplot(data=df_ratios, x='injection', y='sp_so_ratio', order=order,
            fill=False,color=color)

ax.set_ylabel('SO/SP axon % ratio')
ax.set_xlabel('Injection site')
ax.set_title('Relative SO vs. SP axons')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(dst, 'opn3_sp_so_ratio.svg'), bbox_inches='tight')
