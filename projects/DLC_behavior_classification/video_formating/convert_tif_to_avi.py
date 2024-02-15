##
"""
convert tif to avis!
by Zahra
2/5/2024
1. takes tifs from bonsai (in an external drive)
2. makes a memory mapped array in dst
3. converts array to lossless avi
4. checks to make sure avi is made, deletes folder and array
"""
import tifffile as tif, numpy as np, os, sys, shutil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
sys.path.append(r'C:\Users\workstation2\Documents\MATLAB\han-lab') ## custom to your clone
from utils.utils import listdir
import SimpleITK as sitk, re
from avi import read_to_memmap, vidwrite
if __name__ == "__main__":
    delete_fld = True # deletes tif folder

    src = r"F:\tail\240205-240211"
    dst = r"Y:\videos_temp\tail"
    checkdst = r"Y:\videos_temp\tail"#r"\\storage1.ris.wustl.edu\ebhan\Active\new_eye_videos"

    vids = listdir(src)
    print(vids)
    for vid in vids:
        # vid = vids[12]
        print(vid)
        # check and save at diff locations
        checkflnm = os.path.join(checkdst, os.path.basename(vid)+'.avi')
        flnm = os.path.join(dst, os.path.basename(vid)+'.avi')
        fls = np.array(listdir(vid, ifstring='tif'))
        # order by tif index, wrong if you just do sort!
        order = np.array([int(re.findall(r'\d+', os.path.basename(xx))[2]) for xx in fls])
        fls = fls[np.argsort(order)]
        y,x = sitk.GetArrayFromImage(sitk.ReadImage(fls[0])).shape
        if not os.path.exists(checkflnm[:-4]+'.npy') and not os.path.exists(checkflnm):
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
        if delete_fld==True and (os.path.exists(checkflnm) or os.path.exists(flnm)):
            print(f"***********deleting tif folder {vid} after making avi \n*********** \n")
            shutil.rmtree(vid)

#######################
# for adina!           
####################### 
# command line to convert avi to imagej comptabile
# ffmpeg -i K:\230609_E201.avi -c:v rawvideo K:\230609_E201_conv.avi
# all videos have undergone lossless compression :)