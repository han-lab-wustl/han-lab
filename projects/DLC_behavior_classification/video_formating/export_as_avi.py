import tifffile as tif, numpy as np, os, multiprocessing as mp, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir

def convert_to_memmap(memmap, fl, ii):
    arr = tif.imread(fl)
    memmap[ii] = arr
    print(ii)
    sys.stdout.flush()
    memmap.flush()
    return ii

src = r'Y:\DLC\200930_E140'
pth = r'Y:\DLC'
fls = listdir(src, ifstring="tif")
y,x = tif.imread(fls[0]).shape
flnm = r'Y:\DLC\200930_E140.npy'
# init memory map
arr = np.memmap(flnm, mode='r+')

memmap = np.memmap(flnm, mode='w+', shape=(len(fls), y, x))
iterlst = [(memmap, fl, ii) for ii,fl in enumerate(fls)]

with mp.Pool(processes=12) as pool:
    pool.map(convert_to_memmap,iterlst)

