# Zahra
# find videos that match vr files with a specific behavior
# used to pool animals for dlc analysis

import pandas as pd, os, numpy as np, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from datetime import datetime
from utils.utils import listdir

src = r'Y:\DLC\VR_data\dlc'

rrcsv = pd.read_csv(os.path.join(src,'random_reward.csv'))
# add animal and dates to column
rrcsv['animal'] = [xx[:4].upper() for xx in rrcsv.Var1.values]
dates = []
for xx in rrcsv.Var1.values:
    try:
        dates.append(str(datetime.strptime(xx[5:16],'%d_%b_%Y').date()))
    except Exception as e:
        print(e)
        dates.append(np.nan)
rrcsv['date'] = dates
vidsrcs = [r'F:\eye\eye_videos', r'H:\eye_videos']
for vidsrc in vidsrcs:
    for vids in listdir(vidsrc,".avi"):
        date = os.path.basename(vids)[:6]
        viddate = str(datetime.strptime(date, '%y%m%d').date())
        if os.path.basename(vids)[-6:-5] == "_":
            animal = os.path.basename(vids)[-10:-6]
        else:
            animal = os.path.basename(vids)[-8:-4]
        rrcsv.loc[(rrcsv['animal']==animal) & (rrcsv['date']==viddate), 'video'] = vids
rrcsv.dropna().to_csv(os.path.join(src, 'random_reward_videos.csv'))

hrzcsv = pd.read_csv(os.path.join(src,'hrz.csv'))
# add animal and dates to column
hrzcsv['animal '] = [xx[:4].upper() for xx in hrzcsv.Var1.values]
dates = []
for xx in hrzcsv.Var1.values:
    try:
        dates.append(str(datetime.strptime(xx[5:16],'%d_%b_%Y').date()))
    except Exception as e:
        print(e)
        dates.append(np.nan)
hrzcsv['date'] = dates
vidsrcs = [r'F:\eye\eye_videos', r'H:\eye_videos']
for vidsrc in vidsrcs:
    for vids in listdir(vidsrc,".avi"):
        date = os.path.basename(vids)[:6]
        viddate = str(datetime.strptime(date, '%y%m%d').date())
        if os.path.basename(vids)[-6:-5] == "_":
            animal = os.path.basename(vids)[-10:-6]
        else:
            animal = os.path.basename(vids)[-8:-4]
        hrzcsv.loc[(hrzcsv['animal']==animal) & (hrzcsv['date']==viddate), 'video'] = vids
hrzcsv.dropna().to_csv(os.path.join(src, 'hrz_videos.csv'))
