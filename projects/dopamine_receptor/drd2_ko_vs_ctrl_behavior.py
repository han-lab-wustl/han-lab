"""zahra
sept 2024
"""

import os, sys, numpy as np, re, pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab')
from projects.memory.behavior import get_success_failure_trials, calculate_lick_rate
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
#%%
        
mice = ['e256', 'e262']
matsrc = r'Y:\drd\ko_behavior_analysis'
mats = [os.path.join(matsrc,xx) for xx in os.listdir(matsrc)]

mouse_name_date = [get_name_date(xx) for xx in os.listdir(matsrc)]
dct = {}

for i in range(len(mouse_name_date)):
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
    eps = np.where(changerewloc>0)[0]
    eps=np.append(eps,len(changerewloc))
    # rews_centered = np.zeros_like(velocity)
    # rews_centered[(ypos >= rewloc-5) & (ypos <= rewloc)]=1
    # rews_iind = consecutive_stretch(np.where(rews_centered)[0])
    # min_iind = [min(xx) for xx in rews_iind if len(xx)>0]
    # rews_centered = np.zeros_like(velocity)
    # rews_centered[min_iind]=1
    rates = []
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        tr = trialnum[eprng]
        rew = (rewards==1).astype(int)[eprng]
        if np.max(tr)>11: # at least 8 trials
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(tr, rew)        
            rates.append(success/total_trials)
    rates_av = np.nanmean(np.array(rates))
    dct[f'{mouse_name_date[i][0]}_{mouse_name_date[i][1]}']=[rates_av, 
        np.nanmean(velocity)]
    
#%%
df = pd.DataFrame()
df['rates'] = [v[0] for k,v in dct.items()]
df['date'] = [xx[1] for xx in mouse_name_date]
df['mouse_name'] = [xx[1] for xx in mouse_name_date]
df.sort_values(by=['date'])