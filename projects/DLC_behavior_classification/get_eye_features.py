# zahra
# eye centroid and feature detection from vralign.p

import numpy as np, pandas
import os, cv2, pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_eye_features(eye_coords, eyelbl = False):
    # eye coords format = list of (x,y) tuples
    img = Image.new('L', (600, 422), 0) # L is imagetype, 600, 422 is image dim

    ImageDraw.Draw(img).polygon(eye_coords, outline=1, fill=1)
    mask = np.array(img)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)  # Area of first contour
    perimeter = cv2.arcLength(cnt, True)  # Perimeter of first contour 

    return area, perimeter

# example on how to open the pickle file
pdst = r"Y:\DLC\dlc_mixedmodel2\E201_06_Apr_2023_vr_dlc_align.p"
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)
# edit name of eye points
eye = ['EyeNorth', 'EyeNorthWest', 'EyeWest', 'EyeSouthWest', 
        'EyeSouth', 'EyeSouthEast', 'EyeEast', 'EyeNorthEast']

for e in eye:
    vralign[e+'_x'][vralign[e+'_likelihood'].astype('float32')<0.9]=np.nan
    vralign[e+'_y'][vralign[e+'_likelihood'].astype('float32')<0.9]=np.nan

#eye centroids, area, perimeter
centroids_x = []; centroids_y = [];
areas = []; circumferences = [];
for i in range(len(vralign['EyeNorthWest_y'])):
    eye_x = np.array([vralign[xx+"_x"] for xx in eye])
    eye_y = np.array([vralign[xx+"_y"] for xx in eye])
    eye_coords = np.array([eye_x, eye_y]).astype(float)
    centroid_x, centroid_y = centeroidnp(eye_coords)
    area, circumference = get_eye_features([(vralign[xx+"_x"][i], 
                                vralign[xx+"_y"][i]) for xx in eye])
    centroids_x.append(centroid_x)
    centroids_y.append(centroid_y)
    areas.append(area); circumferences.append(circumference)
    
# mpl.use('TkAgg')
plt.figure()
rng = np.arange(18000,23000) # restrict number of frames
plt.plot(np.array(circumferences)[rng])
plt.plot(vralign['ybinned'][rng])
plt.plot(vralign['rewards'][rng]*50)
plt.ylim([0,200])
plt.xlabel('frames')
plt.ylabel('pupil circumference')