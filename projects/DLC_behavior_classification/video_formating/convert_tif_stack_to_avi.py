##
"""
convert tif to avis!
by zahra
"""
import tifffile as tif, numpy as np, os, sys, shutil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import listdir
import SimpleITK as sitk, re
from avi import read_to_memmap, vidwrite
if __name__ == "__main__":
    delete_fld = True # deletes tif folder
    src = r"F:\eye\240205-240211\1"
    dst = r"Y:\videos_temp\eye"
    vids = listdir(src, ifstring='tif')
    print(vids)
    for vid in vids:
        # vid = vids[12]
        print(vid)
        flnm = os.path.join(dst, os.path.basename(vid)[:-4]+'.avi')
        if not os.path.exists(flnm): # if avi does not exist
            arr = tif.imread(vid)
            vidwrite(flnm,arr)   # make avi 
            del arr # remove var
        if delete_fld==True and os.path.exists(flnm):
            print(f"***********deleting tif {vid} after making avi \n*********** \n")
            os.remove(vid)
            
# command line to convert avi to imagej comptabile
# ffmpeg -i K:\230609_E201.avi -c:v rawvideo K:\230609_E201_conv.avi
# all videos have undergone lossless compression :)