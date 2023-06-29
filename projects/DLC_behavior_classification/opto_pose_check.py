# zahra
# plot poses for opto days
# see if there are any behaviors/poses different during opto sessions

import pandas as pd, os, numpy as np, matplotlib.pyplot as plt
from scipy.io import loadmat

pth = r'X:\eye_videos\230516_E201DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
df = pd.read_csv(pth)
cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
df.columns = cols
df=df.drop([0,1])
# plot
fig, ax = plt.subplots()
ax.plot(df.PawTop_x.values.astype(float)[df.PawTop_likelihood.values.astype(float)>0.9])
ax.plot(df.PawMiddle_x.values.astype(float)[df.PawMiddle_likelihood.values.astype(float)>0.9])
# ax.plot(df.PawBottom_x.values.astype(float))
# ax.plot(df.WhiskerLower_x.values.astype(float))
ax.axvspan(19710*2, 22630*2,
alpha=0.5)