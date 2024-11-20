"""zahra
sept 2024
"""
#%%
import os, sys, numpy as np, re, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt, scipy
from scipy.io import loadmat
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from projects.memory.behavior import get_success_failure_trials, calculate_lick_rate, \
    get_lick_selectivity, calculate_lick_rate
import matplotlib as mpl
from datetime import datetime
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes

def get_name_date(input_string):
    # Use regular expressions to find the date in 'dd_MMM_yyyy' format
    match = re.search(r'(\d{2})_(\w{3})_(\d{4})', input_string)
    date_str = np.nan
    if match:
        day = match.group(1)
        month_str = match.group(2)
        year = match.group(3)

        # Dictionary to convert month abbreviations to their numerical equivalents
        month_dict = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        
        if month_str in month_dict:
            month = month_dict[month_str]
            date_str = f"{year}{month}{day}"
            print(f"Converted date: {date_str}")
        else:
            print("Invalid month abbreviation.")
    else:
        print("Date not found in the string.")
        
    underscore_index = input_string.find('_')

    if underscore_index != -1:
        first_word = input_string[:underscore_index]
        return date_str, first_word
        
mice = ['e256', 'e255', 'e253', 'e254', 'e262', 'e261']

matsrc = r'Y:\drd\ko_behavior_analysis\srz'
mats = [os.path.join(matsrc,xx) for xx in os.listdir(matsrc)]
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\dopamine_projects\drd'

mouse_name_date = [get_name_date(xx) for xx in os.listdir(matsrc)]
rec_day = np.concatenate([[ii for ii,xx in enumerate(mouse_name_date) if xx[1]==an.upper()] for an in mice])
dct = {}

for i in range(1,len(mouse_name_date)):
    f = loadmat(mats[i])
    VR = f['VR'][0][0]
    # dtype=[('name_date_vr', 'O'), ('ROE', 'O'), ('lickThreshold', 'O'), ('reward', 'O'), 
    # ('time', 'O'), ('lick', 'O'), ('ypos', 'O'), 
    #          ('lickVoltage', 'O'), ('trialNum', 'O'), ('timeROE', 'O'), ('changeRewLoc', 'O'), ('pressedKeys', 'O'), ('world', 'O'), 
    #          ('imageSync', 'O'), ('scalingFACTOR', 'O'), ('wOff', 'O'),
    #          ('catchTrial', 'O'), ('optoTrigger', 'O'), ('settings', 'O')]) 
    velocity = VR[1][0]
    lick = VR[5][0]
    time = VR[4][0]
    gainf = VR[14][0][0]
    try:
        rewsize = VR[18][0][0][4][0][0]/gainf
    except:
        rewsize = 20
    velocity=-0.013*velocity[1:]/np.diff(time) # make same size
    velocity = np.append(velocity, np.interp(len(velocity)+1, np.arange(len(velocity)),velocity))
    velocitydf = pd.DataFrame({'velocity': velocity})
    velocity = np.hstack(velocitydf.rolling(10).mean().values)
    rewards = VR[3][0]
    ypos = VR[6][0]/gainf
    trialnum = VR[8][0]
    changerewloc = VR[10][0]
    rewlocs = changerewloc[changerewloc>0]/gainf
    fprev = loadmat(mats[i-1])
    # get previous reward loc
    diffday = rec_day[i]-rec_day[i-1]
    if diffday==1:
        fp = loadmat(mats[i-1]); VRp = fp['VR'][0][0]
        pchangerewloc = VRp[10][0]
        prevrewloc = pchangerewloc[pchangerewloc>0][0]/gainf
    catchtrialsnum = trialnum[VR[16][0].astype(bool)]        
    # probe trials
    probe = trialnum<3
    lick_selectivity_probes = get_lick_selectivity(ypos[probe], trialnum[probe], 
                    lick[probe], prevrewloc, rewsize,
                    fails_only=True)
    lick_selectivity_probes=np.nanmean(lick_selectivity_probes)
    # from vip opto
    window_size = 5
    # also estimate sampling rate
    lick_rate_probes = calculate_lick_rate(lick[probe], 
                window_size, sampling_rate=31.25*1.5)
    lick_rate_probes=np.nanmean(lick_rate_probes)

    eps = np.where(changerewloc>0)[0]
    eps=np.append(eps,len(changerewloc))
    # rews_centered = np.zeros_like(velocity)
    # rews_centered[(ypos >= rewloc-5) & (ypos <= rewloc)]=1
    # rews_iind = consecutive_stretch(np.where(rews_centered)[0])
    # min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
    # rews_centered = np.zeros_like(velocity)
    # rews_centered[min_iind]=1
    # lick rate
        # from vip opto
    window_size = 10
    # also estimate sampling rate
    # get licks before rew zone
    lick_outside_rew_mask = np.concatenate([ypos[eps[i]:eps[i+1]]<rewlocs[i]-(rewsize/2) \
        for i in range(len(eps)-1)])
    lick_rate = calculate_lick_rate(lick[lick_outside_rew_mask], 
                window_size, sampling_rate=31.25*1.5)

    rates = []; num_trials = []; ls = []
    # last few trials to get lick selectivity
    lasttr = 10
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        tr = trialnum[eprng]
        rew = (rewards==1).astype(int)[eprng]
        if np.max(tr)>11: # at least x trials
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(tr, rew)        
            rates.append(success/total_trials)
            num_trials.append(total_trials)
            lick_selectivity_success = get_lick_selectivity(ypos[eprng], 
                trialnum[eprng], lick[eprng], rewlocs[ep], rewsize,
                fails_only = False)     
            ls.append(np.nanmean(np.array(lick_selectivity_success)[:-lasttr]))      
    num_trials = np.sum(np.array(num_trials))
    rates_av = np.nanmean(np.array(rates))
    ls_av = np.nanmean(np.array(ls))

    dct[f'{mouse_name_date[i][0]}_{mouse_name_date[i][1]}']=[rates_av, 
        np.nanmean(velocity), num_trials/(time[-1]/60), len(eps)-1, ls_av,
    np.nanmean(lick_rate),lick_selectivity_probes,lick_rate_probes]
    
#%%
df = pd.DataFrame()

df['success_rate'] = [v[0] for k,v in dct.items()]
df['average_velocity'] = [v[1] for k,v in dct.items()]
df['date'] = [xx[0] for xx in mouse_name_date[1:]]
df['mouse_name'] = [xx[1].lower() for xx in mouse_name_date[1:]]
condition = []
for ii,xx in enumerate(df.mouse_name.values):
    if '26' in df.mouse_name.values[ii]:
        cond = 'drd2ko'
    else:
        cond = 'drd1_2'
    # elif xx=='e256' or xx=='e253':
    #     cond = 'drd2'
    # else:
    #     cond = 'drd1'
    condition.append(cond)
df['condition'] = condition
df = df.sort_values(by=['mouse_name','date'])
df['srz_day'] = np.concatenate([[ii+1 for ii,xx in enumerate(df.loc[df.mouse_name==nm, 'date'])] for nm in mice]).astype(int)
df['trials_per_min'] = [v[2] for k,v in dct.items()]
df['epochs_per_day'] = [v[3] for k,v in dct.items()]
df['lick_selectivity'] = [v[4] for k,v in dct.items()]
df['lick_rate'] = [v[5] for k,v in dct.items()]
df['lick_rate_probes'] = [v[7] for k,v in dct.items()]
df['lick_selectivity_probes'] = [v[6] for k,v in dct.items()]

fig,axes = plt.subplots(figsize=(12,12),nrows = 3, ncols=3)
metrics = ['success_rate', 'average_velocity', 'trials_per_min', 
        'lick_rate', 'lick_selectivity','lick_selectivity_probes']
axes = np.concatenate(axes)

plt.rc('font', size=20)          # controls default text sizes
for ii, m in enumerate(metrics):
    ax = axes[ii]
    sns.lineplot(x='srz_day', y=m, hue='mouse_name', data=df,ax=ax)
    sns.scatterplot(x='srz_day', y=m, hue='mouse_name', data=df,ax=ax,s=150)
    ax.legend_.remove()  # Remove the legend
    if ii==len(metrics)-1:
        ax.legend(bbox_to_anchor=(1.01, 1.01))

# Hide any remaining empty axes
for jj in range(len(metrics), len(axes)):
    fig.delaxes(axes[jj])

fig.tight_layout()
# %%

fig,axes = plt.subplots(figsize=(12,12),nrows = 3, ncols=3)
metrics = ['success_rate', 'average_velocity', 'trials_per_min', 
        'lick_rate', 'lick_selectivity','lick_selectivity_probes']
axes = np.concatenate(axes)

plt.rc('font', size=20)          # controls default text sizes
for ii, m in enumerate(metrics):
    ax = axes[ii]
    sns.lineplot(x='srz_day', y=m, hue='condition', 
        data=df,ax=ax)
    # sns.scatterplot(x='hrz_day', y=m, hue='condition', data=df,ax=ax,s=150)
    ax.legend_.remove()  # Remove the legend
    if ii==len(metrics)-1:
        ax.legend(bbox_to_anchor=(1.01, 1.01))
    ax.set_xlim([1,9])

# Hide any remaining empty axes
for jj in range(len(metrics), len(axes)):
    fig.delaxes(axes[jj])

fig.tight_layout()

#%%
# quantification for fig
sns.set_palette('colorblind')
cmap = [sns.color_palette('colorblind')[5],sns.color_palette('colorblind')[2]]
fig,axes = plt.subplots(figsize=(12,11),nrows=3, ncols=2)
metrics = ['success_rate', 'lick_rate', 'lick_selectivity','lick_selectivity_probes']
axes = np.concatenate(axes)

for ii, m in enumerate(metrics):
    ax = axes[ii]
    sns.pointplot(x='srz_day', y=m, hue='condition', data=df,ax=ax,
        palette=cmap)
    # sns.scatterplot(x='hrz_day', y=m, hue='condition', data=df,ax=ax,s=150)
    ax.legend_.remove()  # Remove the legend
    if ii==len(metrics)-1:
        ax.legend()#bbox_to_anchor=(1, 1))
    ax.set_xlim([0,11])

# Hide any remaining empty axes
for jj in range(len(metrics), len(axes)):
    fig.delaxes(axes[jj])

fig.tight_layout()
# plt.savefig(os.path.join(savedst,'ko_behavior.svg'),bbox_inches='tight')

#%%
# condition = []
# for ii,xx in enumerate(df.mouse_name.values):
#     if '26' in df.mouse_name.values[ii]:
#         cond = 'drd2ko'
#     elif (df.mouse_name.values[ii]=='e256')|(df.mouse_name.values[ii]=='e253'):
#         cond = 'drd2'
#     elif (df.mouse_name.values[ii]=='e255')|(df.mouse_name.values[ii]=='e254'):
#         cond = 'drd1'
#     # elif xx=='e256' or xx=='e253':
#     #     cond = 'drd2'
#     # else:
#     #     cond = 'drd1'
#     condition.append(cond)
# df['condition'] = condition
plt.rc('font', size=24)          # controls default text sizes

# df = df[df.srz_day.values<6]
dfan = df.groupby(['mouse_name', 'condition']).mean(numeric_only=True)

metrics=['success_rate', 'lick_rate','lick_selectivity','lick_selectivity_probes']
for m in metrics:
    # average across first 5 days
    sr = dfan.loc[(dfan.index.get_level_values('condition')=='drd1_2'), m].values
    srko = dfan.loc[(dfan.index.get_level_values('condition')=='drd2ko'), m].values
    t,pval = scipy.stats.ranksums(sr, srko)
    # tau, pval = scipy.stats.mannwhitneyu(sr,srko)

    fig,ax = plt.subplots(figsize=(2,4))
    sns.barplot(x='condition',y=m, data=dfan, hue='condition',
            palette=cmap,fill=False,linewidth=3.5,errwidth=3.5)
    sns.stripplot(x='condition',y=m, data=dfan, hue='condition',
            palette=cmap,s=12,alpha=.7)
    text_x = ax.get_xlim()[0];text_y = ax.get_ylim()[1]
    ax.text(text_x, text_y,f'ranskum pval={pval:.9f}', fontsize=9)
    ax.set_xticklabels(['D1 & D2', 'CRISPR-KO D2'],rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    # plt.savefig(os.path.join(savedst, f'{m}_ko_behavior_quant.svg'), bbox_inches='tight')