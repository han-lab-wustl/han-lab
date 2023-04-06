# Zahra
# visualize mean images of axonal-gcamp images

import os, numpy as np, tifffile, matplotlib.pyplot as plt
from preprocessing import getmeanimg

dys = [2,3,4,5,6,7,8] #may change based on mouse
animal = 'e194' #mouse name in folder
src = r'X:\dopamine_imaging'

for dy in dys:
    imgfld = os.path.join(src, animal, str(dy))
    imgfl = [os.path.join(imgfld, xx) for xx in os.listdir(imgfld) if "ZD" in xx][0]
    meanimg=[]
    for pln in range(3):
        reg_tif = os.path.join(imgfl, "suite2p", f"plane{pln}","reg_tif","file2500_chan0.tif") # get a portion of motion corr movie
        meanimg.append(getmeanimg(reg_tif))

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    ax = axes[0]
    ax.imshow(meanimg[2], cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SO")
    ax = axes[1]
    ax.imshow(meanimg[1], cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SP")
    ax = axes[2]
    ax.imshow(meanimg[0], cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SR")

    fig.suptitle(f"{animal}, day{dy}",fontsize=18)