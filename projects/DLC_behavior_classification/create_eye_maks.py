# zahra
# eye mask
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?
def fixcsvcols(csv):
    if type(csv) == str:
        df = pd.read_csv(csv)
        cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
        cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
        df.columns = cols
        df=df.drop([0,1])

    else:
        print("\n ******** please pass path to csv ********")
    return df
pth = r'Y:\DLC\dlc_mixedmodel2\230505_E200DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
df = fixcsvcols(pth)

eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
            'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']
eye_coords = [];
for i in range(len(df)):
    eye_coords.append([(float(df[xx+"_x"].iloc[i]), 
                                  float(df[xx+"_y"].iloc[i])) for xx in eye]),

video = r'G:\eye\eye_videos\230505_E200.avi'
cap = cv2.VideoCapture(video)
ret, frame = cap.read()    
fig = plt.figure()
plt.imshow(frame)
plt.imsave(r'C:\Users\Han\Desktop\test.jpg', frame)
img = Image.new('L', (600, 422), 0) # L is imagetype, 600, 422 is image dim

ImageDraw.Draw(img).polygon(eye_coords[0], outline=1, fill=1)
mask = np.array(img)

plt.figure()

plt.imshow(frame)
plt.imshow(mask, 'Reds', alpha=0.1)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
                    cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
area = cv2.contourArea(cnt)  # Area of first contour
perimeter = cv2.arcLength(cnt, True)  # Perimeter of first contour 

#%%
# plot test
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
rng1 = 13000; rng2 = 15000
ax1.plot(mat['ybinned'][rng1:rng2])
ax2.plot(areafil[rng1:rng2],'r')
ax3.plot(mat['forwardvel'][rng1:rng2],'k')
ax4.plot((mat['rewards']==1)[rng1:rng2],'g')
#ax2.set_ylim(1000,1500) #Define limit/scale for primary Y-axis