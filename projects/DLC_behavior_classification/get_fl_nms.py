from pathlib import Path
import os
import numpy as np, re
src = r'J:\tail\all'
import glob
for f in glob.glob(r'J:\\tail\\all\\**\\23*E*', recursive=True):
    print(f)
an_fls = []
for path in Path(src).rglob('**/23*'):
    if 'tif' not in path.name and 'csv' not in path.name:
        an_fls.append(path.name)
        print(path.name)
dst = r'K:\tail_videos'
tiffs = np.array(os.listdir(src))
avis = np.array([xx[:-4] for xx in os.listdir(dst) if '.avi' in xx])
mask = np.isin(an_fls, avis)
print(mask)

##
import tifffile as tif, numpy as np, os, multiprocessing as mp, sys
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir
import SimpleITK as sitk

src = r'J:\tail\all\230313-230319\230314_E200'
flnm = r'I:\test.avi'
fls = np.array(listdir(src, ifstring='tif'))
# order by tif index, wrong if you just do sort!
order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[2]) for xx in fls])
fls = fls[np.argsort(order)]
y,x = sitk.GetArrayFromImage(sitk.ReadImage(fls[0])).shape
arr = np.zeros((len(fls),y,x))
for ii,fl in enumerate(fls):
    arr[ii] = sitk.GetArrayFromImage(sitk.ReadImage(fl))
    if ii%1000==0: print(ii)
#%%
arr2 = arr[:3000]
import ffmpeg
def vidwrite(fn, images, framerate=62, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray8', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='gray8', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()
    
vidwrite(flnm,arr)    

import h5py
h5f = h5py.File(r'I:\test.h5', 'w')
h5f.create_dataset('video', data=arr)
h5f.close()
