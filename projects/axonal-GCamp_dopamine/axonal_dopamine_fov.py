# Zahra
# visualize mean images of axonal-gcamp images

import os, numpy as np, tifffile, matplotlib.pyplot as plt

dys = [2,3,4,5] #may change based on mouse
animal = 'e194' #mouse name in folder
src = r'X:\dopamine_imaging'

def getmeanimg(pth):
    """coverts tif to mean img

    Args:
        pth (str): path to tif

    Returns:
        tif: meanimg
    """
    img = tifffile.imread(pth)
    meanimg = np.mean(img,axis=0)
    return meanimg

for dy in dys:
    imgfld = os.path.join(src, animal, str(dy))
    imgfl = [os.path.join(imgfld, xx) for xx in os.listdir(imgfld) if "ZD" in xx][0]
    tifs = [os.path.join(imgfl, xx) for xx in os.listdir(imgfl) if "tif" in xx]
    # plot mean images side by side per day
    plane1 = [xx for xx in tifs if "plane01" in xx]
    meanimg_pln1 = getmeanimg(plane1[0])

    plane2 = [xx for xx in tifs if "plane02" in xx]
    meanimg_pln2 = getmeanimg(plane2[0])

    plane3 = [xx for xx in tifs if "plane03" in xx]
    meanimg_pln3 = getmeanimg(plane3[0])

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    ax = axes[0]
    ax.imshow(meanimg_pln3, cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SO")
    ax = axes[1]
    ax.imshow(meanimg_pln2, cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SP")
    ax = axes[2]
    ax.imshow(meanimg_pln1, cmap="Greys_r")
    ax.axis("off")
    ax.set_title("SR")

    fig.suptitle(f"{animal}, day{dy}",fontsize=18)