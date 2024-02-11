<<<<<<< HEAD
import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign
pdst = [path to pickle]
gainf, rewsize = 3/2, 20
areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = get_area_circumference_from_vralign(pdst, gainf, rewsize)

open vralign pickle

plt.figure()
areas = scipy.ndimage.gaussian_filter(areas,2)
r = np.random.randint(1000, len(areas))

plt.plot(areas[r:r+1000])
plt.plot((vralign['rewards']==0.5)[r:r+1000])
=======
import pickle, os, sys, matplotlib.pyplot as plt
import numpy as np, scipy
from eye import get_area_circumference_from_vralign
pdst = r'D:\PupilTraining-Matt-2023-07-07\E217_20_Jan_2024_vr_dlc_align.p'
gainf, rewsize = 3/2, 20
areas, circumferences, centroids_x, centroids_y, normmeanrew_t, \
            normrewall_t, normmeanlicks_t, meanlicks, normlickall_t, \
            lickall, normmeanvel_t, meanvel, normvelall_t, \
            velall = get_area_circumference_from_vralign(pdst, gainf, rewsize)
savedst = r"D:\PupilTraining-Matt-2023-07-07"
pdst = os.path.join(savedst, "E217_20_Jan_2024_vr_dlc_align.p")
with open(pdst, "rb") as fp: #unpickle
        vralign = pickle.load(fp)

plt.figure()

areas = scipy.ndimage.gaussian_filter(areas,1.5)
r = np.random.randint(1000, len(areas))

plt.plot(areas[r:r+3000])
licks = np.hstack(vralign['licks']*500)
plt.plot(licks[r:r+3000])
plt.plot((vralign['rewards']==0.5)[r:r+3000]*600)
plt.plot(vralign['forwardvel'][r:r+3000]*2)
circumferences = scipy.ndimage.gaussian_filter(circumferences,2)
<<<<<<< HEAD
plt.plot(vralign['forwardvel'][r:r+3000]*2)
#plt.plot(circumferences[r:r+1000])
plt.figure()
plt.plot((vralign['rewards']==0.5)[0:3000]*600)
plt.plot(licks[0:3000])
plt.plot(areas[0:3000])
#ffmpeg -i I:\eye_videos\240120_E217.avi -c:v rawvideo I:\eye_videos\240120_E217_conv.avi

plt.figure()
plt.plot(areas)
plt.plot(vralign['forwardvel']*2)

plt.figure()
plt.plot(normmeanrew_t)
plt.imshow(normrewall_t)
plt.figure()
#plt.plot(normmeanvel_t)
plt.plot(normmeanlicks_t)
=======
plt.plot(vralign['forwardvel'][r:r+3000])
#plt.plot(circumferences[r:r+1000])
>>>>>>> 097c18b96088986bd88fb2454ae335227b13e6c2
>>>>>>> 03257286f1285e1a4af2ef484af53f0470f37fb3
