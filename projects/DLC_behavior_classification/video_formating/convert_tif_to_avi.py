##
"""
convert tif to avis!
"""
import tifffile as tif, numpy as np, os, sys, shutil
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir
from multiprocess import Pool
import SimpleITK as sitk, re
from avi import read_to_memmap, vidwrite
if __name__ == "__main__":
    delete_fld = True # deletes tif folder
    src = r"G:\eye\TI"
    dst = r"I:\eye_videos"
    # src = r"E:\tail\all\2023"
    # dst = r"K:\tail_videos"
    vids = listdir(src)
    print(vids)
    for vid in vids:
        # vid = vids[12]
        print(vid)
        flnm = os.path.join(dst, os.path.basename(vid)+'.avi')
        fls = np.array(listdir(vid, ifstring='tif'))
        # order by tif index, wrong if you just do sort!
        order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[2]) for xx in fls])
        fls = fls[np.argsort(order)]
        y,x = sitk.GetArrayFromImage(sitk.ReadImage(fls[0])).shape
        if not os.path.exists(flnm[:-4]+'.npy') and not os.path.exists(flnm):
            arr = np.memmap(flnm[:-4]+'.npy', dtype='uint8', 
                            mode='w+', shape=(len(fls),y,x))
            for ii,fl in enumerate(fls):
                read_to_memmap(arr, ii, fl)
            # args = [(arr, ii, fl) for ii,fl in enumerate(fls)]
            # with Pool(3) as p:
            #     p.starmap(read_to_memmap, args)     
            #     p.terminate   
            # if no vid written 
            # load memmap array 
            arr = np.memmap(flnm[:-4]+'.npy', dtype='uint8', mode='r', shape=(len(fls),y,x))      
            vidwrite(flnm,arr)   # make avi 
            del arr # remove var
            # now delete memmap array
            if os.path.exists(flnm): os.remove(flnm[:-4]+'.npy')
        if delete_fld==True and os.path.exists(flnm):
            print(f"***********deleting tif folder {vid} after making avi \n*********** \n")
            shutil.rmtree(vid)
            
# command line to convert avi to imagej comptabile
# ffmpeg -i K:\230609_E201.avi -c:v rawvideo K:\230609_E201_conv.avi
# all videos have undergone lossless compression :)