import os, pandas as pd, scipy, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pth = r'Y:\DLC\dlc_mixedmodel2\for_analysis\230505_E201DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'

df = pd.read_csv(pth)
df = df.drop(columns = ['Unnamed: 0'])
df['TongueTip_x'][df['TongueTip_likelihood'].astype('float32') < 0.9] = 0
df['TongueTip_y'][df['TongueTip_likelihood'].astype('float32') < 0.9] = 0
df['TongueTop_x'][df['TongueTop_likelihood'].astype('float32') < 0.9] = 0
df['TongueTop_y'][df['TongueTop_likelihood'].astype('float32') < 0.9] = 0
df['TongueBottom_x'][df['TongueBottom_likelihood'].astype('float32') < 0.9] = 0
df['TongueBottom_y'][df['TongueBottom_likelihood'].astype('float32') < 0.9] = 0
#paw
df['PawTop_x'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
df['PawTop_y'][df['PawTop_likelihood'].astype('float32') < 0.9] = 0
df['PawMiddle_x'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
df['PawMiddle_y'][df['PawMiddle_likelihood'].astype('float32') < 0.9] = 0
df['PawBottom_x'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0
df['PawBottom_y'][df['PawBottom_likelihood'].astype('float32') < 0.9] = 0

#whisker
df['WhiskerUpper1_x'][df['WhiskerUpper1_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerUpper1_y'][df['WhiskerUpper1_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerUpper_x'][df['WhiskerUpper_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerUpper_y'][df['WhiskerUpper_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerUpper3_x'][df['WhiskerUpper3_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerUpper3_y'][df['WhiskerUpper3_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower_x'][df['WhiskerLower_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower_y'][df['WhiskerLower_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower1_x'][df['WhiskerLower1_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower1_y'][df['WhiskerLower1_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower3_x'][df['WhiskerLower3_likelihood'].astype('float32') < 0.9] = 0
df['WhiskerLower3_y'][df['WhiskerLower3_likelihood'].astype('float32') < 0.9] = 0
whiskerUpper = df[['WhiskerUpper_x', 'WhiskerUpper1_x', 'WhiskerUpper3_x']].astype('float32').mean(axis=1)
whiskerLower = df[['WhiskerLower_x','WhiskerLower3_x']].astype('float32').mean(axis=1)

paw = df[['PawTop_y','PawBottom_y','PawMiddle_y']].astype('float32').mean(axis=1)
nose=df[['NoseTopPoint_y', 'NoseBottomPoint_y', 'NoseTip_y']].astype('float32').mean(axis=1, skipna=False).astype('float32')
tongue=df[['TongueTip_x','TongueTop_x','TongueBottom_x']].astype('float32').mean(axis=1, skipna=False)
tongue_scaled=StandardScaler().fit_transform(tongue.values.reshape(-1,1))
nose_scaled=StandardScaler().fit_transform(nose.values.reshape(-1,1))
paw_scaled=StandardScaler().fit_transform(paw.values.reshape(-1,1))
wu_scaled=StandardScaler().fit_transform(whiskerUpper.values.reshape(-1,1))
wl_scaled=StandardScaler().fit_transform(whiskerLower.values.reshape(-1,1))
tongue_gf = scipy.ndimage.gaussian_filter(tongue_scaled,20)
nose_gf = scipy.ndimage.gaussian_filter(nose_scaled,20)
paw_gf = scipy.ndimage.gaussian_filter(paw_scaled,20)
wu_gf = scipy.ndimage.gaussian_filter(wu_scaled,5)
wl_gf = scipy.ndimage.gaussian_filter(wl_scaled,5) 
plt.figure(); plt.plot(tongue_gf[30000:45000]); plt.plot(nose_gf[30000:45000]);
plt.plot(paw_gf[30000:45000]); plt.plot(wu_gf[30000:45000]) 
plt.figure(); plt.plot(tongue_scaled[:10000])
# plt.plot(tongue.values[:10000]); 