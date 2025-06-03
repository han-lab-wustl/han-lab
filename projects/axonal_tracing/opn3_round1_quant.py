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
sns.stripplot(x='injection',y='num_axons',hue='hipp_layer',hue_order=order,data=df,dodge=True,jitter=True,s=s,alpha=a,palette=palette,legend=False)
# sns.stripplot(x='injection',y='num_axons',hue='hipp_layer',data=dfan,dodge=True,jitter=True,s=14,hue_order=order,palette=palette)
dfsum = df.groupby(['injection', 'hipp_layer'])['num_axons'].sum().reset_index()
sns.barplot(x='injection',y='num_axons',hue='hipp_layer',data=dfsum,hue_order=order,palette=palette,errorbar='se',fill=False)
ax.legend(title='Hippocampal Layer', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(['SNc','VTA'])
ax.set_ylabel('Total # of axons')
ax.set_xlabel('Injection site')
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
