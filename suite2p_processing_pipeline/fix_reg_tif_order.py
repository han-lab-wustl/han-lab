import os, re, numpy as np

pths = []
planes = [0,1,2]
for plane in planes:
    pths.append(rf'\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\E179 dark reward\Day_08\suite2p\plane{plane}\reg_tif')
for pth in pths:
    fls = [os.path.join(pth, xx) for xx in os.listdir(pth)]
    order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[0]) for xx in fls])
    for i,fl in enumerate(fls):
        os.rename(fl,os.path.join(pth,f'file{order[i]:06d}.tif'))
        # print(os.path.join(pth,f'file{order[i]:06d}.tif'))