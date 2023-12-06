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
import SimpleITK as sitk, re

src = r"F:\231123-231126\231124_E218"
flnm = r'I:\test.avi'
fls = np.array(listdir(src, ifstring='tif'))
# order by tif index, wrong if you just do sort!
order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[2]) for xx in fls])
fls = fls[np.argsort(order)]
y,x = sitk.GetArrayFromImage(sitk.ReadImage(fls[0])).shape
arr = np.zeros((3000,y,x))
for ii,fl in enumerate(fls[:3000]):
    arr[ii] = sitk.GetArrayFromImage(sitk.ReadImage(fl))
    if ii%1000==0: print(ii)
#%%
arr2 = arr
import ffmpeg
def vidwrite(fn, images, framerate=31.25*2, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', r=framerate, pix_fmt='gray8', s='{}x{}'.format(width, height))
            .filter('fps', fps=framerate, round='up')
            .output(fn, pix_fmt='gray8', r=framerate, vcodec=vcodec)
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
