#%%
import os, re, numpy as np
#change
srcpth = r'Y:\halo_grabda\e243\48\250107_ZD_000_002'
pths = []
planes = [0,1,2,3] # specify number of planes
for plane in planes:
    pths.append(os.path.join(srcpth, rf'suite2p\plane{plane}\reg_tif'))
for pth in pths:
    fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'tif' in xx]
    order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[0]) for xx in fls])
    for i,fl in enumerate(fls):
        os.rename(fl,os.path.join(pth,f'file{order[i]:06d}.tif'))
        # print(os.path.join(pth,f'file{order[i]:06d}.tif'))
        
#%%
import os, re, numpy as np

pths = [r'Y:\halo_grabda\e241\26\241125_ZD_000_000\suite2p\plane2\reg_tif']
for pth in pths:
    fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'tif' in xx]
    order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[0]) for xx in fls])
    for i,fl in enumerate(fls):
        os.rename(fl,os.path.join(pth,f'file{order[i]:06d}.tif'))
        # print(os.path.join(pth,f'file{order[i]:06d}.tif'))
        
# %%
