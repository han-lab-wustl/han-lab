##
"""
convert tif to avis!
"""
import tifffile as tif, numpy as np, os, multiprocessing as mp, sys
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir
import SimpleITK as sitk, re
import ffmpeg
from multiprocessing import Pool
def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

def save_memmap_arr(pth, arr):
    '''Function to save memmaped array.
    
    pth = place to save, will overwrite
    arr = arr

    '''
    narr = np.lib.format.open_memmap(pth, dtype = arr.dtype, shape = arr.shape, mode = 'w+')
    narr[:] = arr
    narr.flush(); del narr
    return

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
    
    return fn \

def read_to_memmap(arr, ii, fl):
    arr[ii] = sitk.GetArrayFromImage(sitk.ReadImage(fl))
    arr.flush()
    if ii%1000==0: print(ii)
    return
src = r"F:\231123-231126"
dst = r"I:\eye_videos"
vids = listdir(src)
for vid in vids:
    print(vid)
    flnm = os.path.join(dst, os.path.basename(vid)+'.avi')
    fls = np.array(listdir(vid, ifstring='tif'))
    # order by tif index, wrong if you just do sort!
    order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[2]) for xx in fls])
    fls = fls[np.argsort(order)]
    y,x = sitk.GetArrayFromImage(sitk.ReadImage(fls[0])).shape
    arr = np.memmap(flnm[:-4]+'.npy', dtype='uint8', mode='w+', shape=(len(fls),y,x))
    args = [(arr, ii, fl) for ii,fl in enumerate(fls)]
    with Pool(5) as p:
        print(p.map(read_to_memmap, args))        
    # if no vid written 
    # load memmap array 
    arr = np.memmap(flnm[:-4]+'.npy', dtype='uint8', mode='r', shape=(len(fls),y,x))      
    if not os.path.exists(flnm): vidwrite(flnm,arr)   # make avi 
    del arr # remove var
    # now delete memmap array
    if os.path.exists(flnm): os.remove(flnm[:-4]+'.npy')