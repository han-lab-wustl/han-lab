"""
zahra
sept 2024
"""
#%%
import os, sys, scipy, pandas as pd, re
import numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.backend_pdf

# Add custom path for MATLAB functions
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') 
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\drd_grant_2024'

# Formatting for figures
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"

plt.rc('font', size=20)
df = pd.DataFrame()
df['Relative Intensity'] = [.67,.6,1.16,.56,.75,.56,1.42,.71]
df['Condition'] = ['Control Site', 'Injected Site','Control Site', 'Injected Site',
                'Control Site', 'Injected Site','Control Site', 'Injected Site']
import seaborn as sns
fig, ax = plt.subplots(figsize=(2.2,5))
sns.stripplot(x='Condition',y='Relative Intensity',data=df, s=10,
            color='k',ax=ax)
j=0
for i in range(4):
    sns.lineplot(x='Condition',y='Relative Intensity',data=df[j:j+2], 
                color='k',
                errorbar=None,ax=ax,linewidth=3)
    j+=2
sns.barplot(x='Condition',y='Relative Intensity',data=df, errorbar='se',
         color='k',fill=False,ax=ax,
            linewidth=3,err_kws={'linewidth': 3})
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('')
ax.tick_params(axis='x', labelrotation=45)

x1= df.loc[df.Condition=='Control Site', 'Relative Intensity'].values
x2= df.loc[df.Condition=='Injected Site', 'Relative Intensity'].values
t,pval=scipy.stats.ttest_rel(x1,x2)

ax.set_title(f'p={pval:.4f}')
plt.savefig(os.path.join(savedst, 'immuno_quant.svg'), bbox_inches='tight')
