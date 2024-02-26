# zahra
# eye mask
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tiffile
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
# pth = r'Y:\DLC\dlc_mixedmodel2\230505_E200DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
# df = fixcsvcols(pth)

# eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
#             'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']
# eye_coords = [];
# for i in range(len(df)):
#     eye_coords.append([(float(df[xx+"_x"].iloc[i]), 
#                                   float(df[xx+"_y"].iloc[i])) for xx in eye]),


import time
pixel_values_mask = []
video = r'Y:\DLC\eye_videos\230402_E201.avi'
output_loc = r'Y:\DLC\eye_videos\test'
cap = cv2.VideoCapture(video)
# Find the number of frames
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print ("Number of frames: ", video_length)
count = 0
print ("Converting video..\n")
# Start converting the video
while cap.isOpened():
    # Extract the frame
    ret, frame = cap.read()
    if not ret:
        continue
    if count/2 >= 13000 and count/2 < 15000:
        if count%1000==0: print(count)
        pixel_values_mask.append(get_video_values_of_eye_mask([(float(df[xx+"_x"].iloc[i]), 
                   float(df[xx+"_y"].iloc[i])) for xx in eye],frame))
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
    count = count + 1

    # If there are no more frames left
    if (count > (video_length-1)):
        # Log the time again
        time_end = time.time()
        # Release the feed
        cap.release()
        # Print stats
        print ("Done extracting frames.\n%d frames extracted" % count)
        print ("It took %d seconds forconversion." % (time_end-time_start))
        break

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
#%%
# plot test

pth = r'Y:\\DLC\\dlc_mixedmodel2\\230402_E201DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
# pth = r'Y:\\DLC\\dlc_mixedmodel2\\230418_E200DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
try:
    df = fixcsvcols(pth)
except:
    df = pd.read_csv(pth)
matfl = r'Y:\\DLC\\dlc_mixedmodel2\\E201_02_Apr_2023_vr_dlc_align.p'
# matfl = r'Y:\\DLC\\dlc_mixedmodel2\\E200_18_Apr_2023_vr_dlc_align.p'
with open(matfl,'rb') as fp: #unpickle
        mat = pickle.load(fp)
idx = len(df) - 1 if len(df) % 2 else len(df)
df = df[:idx].groupby(df.index[:idx] // 2).mean()

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()
mat['licks'][mat['licks']<1]=np.nan
rng1 = 10000; rng2 = 13000
ax1.plot(mat['ybinned'][rng1:rng2])
ax2.plot(areas[rng1:rng2],'r')
ax3.plot(mat['forwardvel'][rng1:rng2],'k')
ax4.plot(mat['rewards'][rng1:rng2],'g')
ax5.plot(mat['licks'][rng1:rng2],'ob')
#ax2.set_ylim(1000,1500) #Define limit/scale for primary Y-axis#ax2.set_ylim(1000,1500) #Define limit/scale for primary Y-axis