# build dataframe based on day-to-day data
# sst experiment

import os, shutil, pandas as pd, numpy as np

hd = 'Z:\sstcre_imaging'
df = pd.DataFrame()
df["animal"]=[]
df["session_id"]=[]
animals = os.listdir(hd)
for i,animal in enumerate(animals):
    days = os.listdir(os.path.join(hd, animal))
    if i==0:
        df["animal"]=[animal]*len(days)
        df["session_id"]=days
    else:
        an=df["animal"].append(pd.Series([animal]*len(days)))
        id=df["session_id"].append(pd.Series(days))