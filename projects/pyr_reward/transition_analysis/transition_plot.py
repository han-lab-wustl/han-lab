"""
plt transition data
"""
#%%
import pickle, os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt,scipy
import matplotlib.backends.backend_pdf, matplotlib as mpl
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\TransitionMatrix\Per_Animial_Mean_Transition"
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
#%%
df = pd.DataFrame(dct)
# RR2RR_m	RR2PP_m	RR2NL_m	PP2RR_m	PP2PP_m	PP2NL_m	NL2RR_m	NL2PP_m	NL2NL_m
df.columns = ['Reward-Reward', 'Reward-Place', 'Reward-Untuned', 'Place-Reward', 'Place-Place','Place-Untuned', 'Untuned-Reward', 'Untuned-Place', 'Untuned-Untuned']
order = ['Place-Place','Reward-Place','Place-Reward','Reward-Reward','Place-Untuned', 'Reward-Untuned','Untuned-Reward', 'Untuned-Place', 'Untuned-Untuned']
df = df[order]*100
fig,ax = plt.subplots(figsize=(6,4))
sns.heatmap(df, fmt=".1f", cmap="Blues", cbar_kws={'label': '% Cells that transition'},annot_kws={'size': 12})
ax.set_ylabel('Animal')
ax.set_xlabel('Transition')
plt.savefig(os.path.join(savedst, 'transition_matrix.svg'), bbox_inches='tight')

# draw graph
# Compute means
means = df.mean().to_dict()

# Build graph
G = nx.DiGraph()
for k, v in means.items():
    src, dst = k.split('-')
    G.add_edge(src, dst, weight=v)

# Node positions
pos = nx.spring_layout(G)

fig, ax = plt.subplots(figsize=(6,5))
matplotlib.rcParams['font.family'] = 'Arial'

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='w', edgecolors='black', ax=ax)
nx.draw_networkx_labels(G, pos, font_family='Arial', font_size=16, ax=ax)
fs=14
# Draw normal edges (no self-loops)
for u, v, d in G.edges(data=True):
    if u != v:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrowstyle='-|>', arrowsize=15,
                               edge_color='gray', width=2, ax=ax)
        x_text = (pos[u][0] + pos[v][0]) / 2
        y_text = (pos[u][1] + pos[v][1]) / 2
        ax.text(x_text, y_text, f"{d['weight']:.1f}%", fontsize=fs, ha='center', va='center')

# Manually draw self-loop arrows
loop_offsets = {
    'Place': (0.0, -0.5),
    'Reward': (0.0, 0.5),
    'Untuned': (0.0, -0.5),
}
for node, (dx, dy) in loop_offsets.items():
    x, y = pos[node]
    if G.has_edge(node, node):
        weight = G[node][node]['weight']
        loop = mpatches.FancyArrowPatch(
            (x, y), (x + dx, y + dy),
            connectionstyle="arc3,rad=0.6",
            arrowstyle='-|>',
            mutation_scale=20,
            color='gray',
            lw=2,
            zorder=10
        )
        ax.add_patch(loop)
        ax.text(x + dx * 1.2, y + dy * 1.2, f"{weight:.1f}%", ha='center', va='center', fontsize=fs)

# Adjust plot
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
# Final formatting
# ax.set_title("Transition Diagram", fontsize=16)
ax.axis('off')
# plt.tight_layout()
plt.savefig(os.path.join(savedst, 'transition_diagram.svg'), bbox_inches='tight')
#%%
# transition v shuffle
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\Truth_vs_Shuffle\ShuffleDistribution_PP2PP"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
placec = 'indigo'
rewc = 'cornflowerblue'
a=0.7
# histogram of shufflel v. real
fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True, sharey=True,figsize=(8,7))
axes=axes.flatten()
ax=axes[0]
shuf = [xx*100 for xx in dct['z9'][1]]
ax.hist(shuf, color='grey',alpha=a,label='shuffle')
ax.axvline(dct['z9'][0]*100, color=placec, linewidth=3
           )
p2p=[[dct[an][0] for an,v in dct.items()], [np.nanmean(dct[an][1]) for an,v in dct.items()]]
ax.set_ylabel('Shuffles')
ax.set_xlabel('% Cell transition')
ax.set_title('Place-Place')
ax.spines[['top','right']].set_visible(False)
ax.legend()
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\Truth_vs_Shuffle\ShuffleDistribution_RR2PP"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
ax=axes[1]
shuf = [xx*100 for xx in dct['z9'][1]]
ax.hist(shuf, color='grey',alpha=a)
ax.axvline(dct['z9'][0]*100, color=rewc, linewidth=3
           )
r2p=[[dct[an][0] for an,v in dct.items()], [np.nanmean(dct[an][1]) for an,v in dct.items()]]
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('Reward-Place')
ax.spines[['top','right']].set_visible(False)

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\Truth_vs_Shuffle\ShuffleDistribution_PP2RR"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
ax=axes[2]
shuf = [xx*100 for xx in dct['z9'][1]]
ax.hist(shuf, color='grey',alpha=a)
ax.axvline(dct['z9'][0]*100, color=placec, linewidth=3
           )
p2r=[[dct[an][0] for an,v in dct.items()], [np.nanmean(dct[an][1]) for an,v in dct.items()]]
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('Place-Reward')
ax.spines[['top','right']].set_visible(False)

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\Truth_vs_Shuffle\ShuffleDistribution_RR2RR"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
ax=axes[3]
shuf = [xx*100 for xx in dct['z9'][1]]
ax.hist(shuf, color='grey',alpha=a)
ax.axvline(dct['z9'][0]*100, color=rewc, linewidth=3
           )
# for each mouse
r2r=[[dct[an][0] for an,v in dct.items()], [np.nanmean(dct[an][1]) for an,v in dct.items()]]
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('Reward-Reward')
ax.spines[['top','right']].set_visible(False)
fig.suptitle('z9')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'place2place_transition_shuffle.svg'), bbox_inches='tight')
#%%
# plot signficance
df=pd.DataFrame()
real_transitions = np.concatenate([p2p[0], r2p[0],p2r[0],r2r[0]])
shuf_transitions = np.concatenate([p2p[1], r2p[1],p2r[1],r2r[1]])
transition_type = np.concatenate([['Place-Place']*len(p2p[0]),['Reward-Place']*len(r2p[0]),
                ['Place-Reward']*len(p2r[0]),['Reward-Reward']*len(r2r[0])])
# Create a DataFrame
# Combine into a DataFrame
animals = list(dct.keys())
df = pd.DataFrame({
    'Animal': np.concatenate([animals]*8),
    'Transition': np.concatenate([transition_type] * 2),
    'Value': np.concatenate([real_transitions, shuf_transitions]),
    'Condition': ['Real'] * len(real_transitions) + ['Shuffle'] * len(shuf_transitions)
})
df['Value']=df['Value']*100
n = len(p2p[0])  # assuming all arrays same length
animals = np.tile(np.arange(n), 4)  # 4 transition types
palette = {
    'Real': 'mediumslateblue',
    'Shuffle': 'grey'
}
# Create combined category for hue
df['Group'] = df.apply(
    lambda row: f"Real" if row['Condition'] == 'Real' else 'Shuffle',
    axis=1
)
fig, ax = plt.subplots(figsize=(7,6))
s = 10

# Plot lines connecting real and shuffle for each animal and transition
for animal in df['Animal'].unique():
    for tr, trans in enumerate(df['Transition'].unique()):
        d_an = df[(df['Animal'] == animal) & (df['Transition'] == trans)]
        if len(d_an) == 2:  # has both real and shuffle
            # x positions adjusted for dodge:
            x_real = tr - 0.2
            x_shuffle = tr + 0.2
            y_real = d_an[d_an['Condition']=='Real']['Value'].values[0]
            y_shuffle = d_an[d_an['Condition']=='Shuffle']['Value'].values[0]
            ax.plot([x_real, x_shuffle], [y_real, y_shuffle], color='lightgray', linewidth=1.5, zorder=0)

# Stripplot with dodge for side-by-side points
sns.stripplot(data=df, x='Transition', y='Value', hue='Group',
              dodge=True, palette=palette, alpha=0.7, s=s, ax=ax)

# Barplot with dodge for side-by-side bars
sns.barplot(data=df, x='Transition', y='Value', hue='Group',
            palette=palette, dodge=True, errorbar='se', fill=False, ax=ax)

ax.set_ylabel('% Cell transition')
plt.xticks(rotation=25)

ax.legend_.remove()  # remove legend for now
# --- Run paired t-tests and annotate ---
# Step 1: Collect p-values for all transitions
p_vals = []
t_stats = []
transitions = df['Transition'].unique()

for trans in transitions:
    real_vals = df[(df['Transition'] == trans) & (df['Condition'] == 'Real')].sort_values('Animal')['Value']
    shuf_vals = df[(df['Transition'] == trans) & (df['Condition'] == 'Shuffle')].sort_values('Animal')['Value']
    t_stat, p_val = scipy.stats.wilcoxon(real_vals, shuf_vals)
    t_stats.append(t_stat)
    p_vals.append(p_val)

# Step 2: Apply multiple comparisons correction
rejected, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Step 3: Function to get asterisks
def pval_to_asterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''  # no label for ns

# Step 4: Add asterisks to plot
y_max = df['Value'].max()
for i, (trans, p_corr) in enumerate(zip(transitions, p_vals_corrected)):
    stars = pval_to_asterisks(p_corr)
    ax.text(i, y_max + y_max*0.05, stars, ha='center', va='bottom', fontsize=46)
    ax.text(i, y_max + y_max*0.05-5, p_corr, ha='center', va='bottom', fontsize=8)

ax.spines[['top','right']].set_visible(False)
ax.set_title('Real vs Shuffle Transitions')

fig.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(savedst, 'transition_stats_v_shuffle.svg'), bbox_inches='tight')

#%%
# pre v post

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Post_Mean_Transition_RR2NL"
with open(fl, "rb") as fp: #unpickle
    post_r2un = pickle.load(fp)
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Post_Mean_Transition_RR2PP"
with open(fl, "rb") as fp: #unpickle
    post_r2p = pickle.load(fp)
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Post_Mean_Transition_RR2RR"
with open(fl, "rb") as fp: #unpickle
    post_r2r = pickle.load(fp)

fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Pre_Mean_Transition_RR2NL"
with open(fl, "rb") as fp: #unpickle
    pre_r2un = pickle.load(fp)
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Pre_Mean_Transition_RR2PP"
with open(fl, "rb") as fp: #unpickle
    pre_r2p = pickle.load(fp)
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\TransitionResults\PrePostAnaylsis\Per_Animal_Pre_Mean_Transition_RR2RR"
with open(fl, "rb") as fp: #unpickle
    pre_r2r = pickle.load(fp)

# Assemble DataFrame
data = {
    'Animal': list(range(len(pre_r2r))) * 6,
    'Transition': ['Reward-Reward'] * len(pre_r2r) + ['Reward-Place'] * len(pre_r2p) + ['Reward-Untuned'] * len(pre_r2un) +
                  ['Reward-Reward'] * len(post_r2r) + ['Reward-Place'] * len(post_r2p) + ['Reward-Untuned'] * len(post_r2un),
    'Period': ['Pre'] * len(pre_r2r) + ['Pre'] * len(pre_r2p) + ['Pre'] * len(pre_r2un) +
              ['Post'] * len(post_r2r) + ['Post'] * len(post_r2p) + ['Post'] * len(post_r2un),
    'Value': np.concatenate([list(pre_r2r.values()), list(pre_r2p.values()), list(pre_r2un.values()), list(post_r2r.values()), list(post_r2p.values()), list(post_r2un.values())])
}

df = pd.DataFrame(data)
df['Value']=df['Value']*100
# Perform paired t-tests
transitions = ['Reward-Reward', 'Reward-Place', 'Reward-Untuned']
results = []
for tr in transitions:
    pre_vals = df[(df['Transition'] == tr) & (df['Period'] == 'Pre')].sort_values('Animal')['Value'].values
    post_vals = df[(df['Transition'] == tr) & (df['Period'] == 'Post')].sort_values('Animal')['Value'].values
    t, p = scipy.stats.wilcoxon(post_vals, pre_vals)
    results.append({'Transition': tr, 't_stat': t, 'p_val': p})

# Multiple comparisons correction
p_vals = [r['p_val'] for r in results]
_, p_corr, _, _ = multipletests(p_vals, method='fdr_bh')
for i, r in enumerate(results):
    r['p_val_corrected'] = p_corr[i]
    r['significance'] = ('***' if p_corr[i] < 0.001 else
                         '**' if p_corr[i] < 0.01 else
                         '*' if p_corr[i] < 0.05 else 'ns')

results_df = pd.DataFrame(results)

print(results_df)
fig,ax = plt.subplots(figsize=(5,5))
# Lines connecting Pre and Post per animal
for animal in df['Animal'].unique():
    for tr in df['Transition'].unique():
        d_an = df[(df['Animal'] == animal) & (df['Transition'] == tr)]
        if len(d_an) == 2:
            xvals = [tr, tr]
            yvals = d_an['Value'].values
            xlocs = [list(df['Transition'].unique()).index(tr) - 0.2, list(df['Transition'].unique()).index(tr) + 0.2]
            ax.plot(xlocs, yvals, color='gray', linewidth=1.5, zorder=0,alpha=0.5)
palette = 'Dark2'

# Stripplot
sns.stripplot(data=df, x='Transition', y='Value', hue='Period',
              dodge=True, palette=palette, alpha=0.7, s=s, ax=ax)

# Barplot
sns.barplot(data=df, x='Transition', y='Value', hue='Period',
            palette=palette, dodge=True, errorbar='se', fill=False, ax=ax)

# Significance stars
x_ticks = df['Transition'].unique()
for i, row in results_df.iterrows():
    tr = row['Transition']
    sig = row['significance']
    p = row['p_val_corrected']
    if sig != 'ns':
        xpos = list(x_ticks).index(tr)
        y_pre = df[(df['Transition'] == tr) & (df['Period'] == 'Pre')]['Value'].mean()
        y_post = df[(df['Transition'] == tr) & (df['Period'] == 'Post')]['Value'].mean()
        y_max = max(y_pre, y_post) + 15  # adjust spacing
        ax.text(xpos, y_max, sig, ha='center', va='bottom', fontsize=46, color='black')
        ax.text(xpos, y_max-3, f'{p:.2g}', ha='center', va='bottom', fontsize=12, color='black')

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('% Cell transition')
ax.set_title('Pre v. Post Transitions')
plt.xticks(rotation=25)
plt.savefig(os.path.join(savedst, 'pre_post_transition.svg'), bbox_inches='tight')
