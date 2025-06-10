import numpy as np
from scipy.stats import ranksums, ttest_1samp

# Step 1: Load the 5 .npy files
filenames = ['grabda_e241.npy', 'grabda_e242.npy', 'grabda_e243.npy', 'grabda_e277.npy', 'grabne_e274.npy']  # Replace with your filenames
slope_list = [np.load(fname) for fname in filenames]

# Step 2: Stack into a single array of shape (5, 4)
slopes = np.vstack(slope_list)

# Step 3: Split groups
group1 = slopes[:4]   # First group: 4 subjects
group2 = slopes[4:5]  # Second group: 1 subject

# Step 4: Run rank-sum test for each plane
for plane in range(4):
    x1 = group1[:, plane]
    x2 = group2[:, plane]
    
    # Clean NaNs
    x1_clean = x1[~np.isnan(x1)]
    x2_clean = x2[~np.isnan(x2)]

    if len(x1_clean) > 0 and len(x2_clean) > 0:
        t, pval = ranksums(x1_clean, x2_clean)
        print(f"Plane {plane}: t = {t:.4f}, p = {pval:.4f}")
    else:
        print(f"Plane {plane}: Not enough valid data for test.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

# Load slope data
filenames = ['grabda_e241.npy', 'grabda_e242.npy', 'grabda_e243.npy', 'grabda_e277.npy', 'grabne_e274.npy']  # Replace with your filenames
slope_list = [np.load(f) for f in filenames]
slopes = np.vstack(slope_list)  # shape = (5, 4)

# Build DataFrame
df = pd.DataFrame(slopes, columns=['SLM', 'SR', 'SP', 'SO'])
df['group'] = ['GRABDA'] * 4 + ['GRABNE']

# Melt to long format
df_long = df.melt(id_vars='group', var_name='plane', value_name='slope')

# Set up figure
fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

planes = ['SLM', 'SR', 'SP', 'SO']

for i, p in enumerate(planes):
    ax = axes[i]

    sns.barplot(x='group', y='slope', data=df_long[df_long['plane'] == p],
                errorbar='se', fill=False,
                palette={'GRABDA': '#1f77b4', 'GRABNE': '#E24A33'}, ax=ax)

    sns.stripplot(x='group', y='slope', data=df_long[df_long['plane'] == p],
                  palette={'GRABDA': '#1f77b4', 'GRABNE': '#E24A33'},
                  s=10, ax=ax, alpha=0.6)

    # Statistical test
    x1 = df_long[(df_long['group'] == 'GRABDA') & (df_long['plane'] == p)]['slope'].values
    x2 = df_long[(df_long['group'] == 'GRABNE') & (df_long['plane'] == p)]['slope'].values
    t, pval = ttest_1samp(x1[~np.isnan(x1)], x2[~np.isnan(x2)])

    ax.set_title(f'{p}\np = {pval:.4f}')
    ax.set_xlabel('')
    ax.set_xticklabels(['GRABDA', 'GRABNE'])
    ax.spines[['top', 'right']].set_visible(False)

    # Hide legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.set_visible(False)

# Y-axis label on the first plot only
axes[0].set_ylabel('Slope')
plt.tight_layout()
plt.show()
