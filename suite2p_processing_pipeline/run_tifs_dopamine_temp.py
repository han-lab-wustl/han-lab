
import os , numpy as np, tifffile, SimpleITK as sitk, sys
from math import ceil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone

def maketifs(imagingflnm,y1,y2,x1,x2,zplns=3000):
    """makes tifs out of sbx file

    Args:
        imagingflnm (_type_): folder containing sbx
        y1 (int): lower limit of crop in y
        y2 (int): upper limit of crop in y
        x1 (int): lower limit of crop in x
        x2 (int): upper limit of crop in x
        dtype (str): dopamine or pyramidal cell data, diff by # of planes etc.
        zplns (int, optional): zpln chunks to split the tifs. Defaults to 3000.

    Returns:
        stack: zstack,uint16
    """
    sbxfl=[os.path.join(imagingflnm,xx) for xx in os.listdir(imagingflnm) if "sbx" in xx][0]
    from sbxreader import sbx_memmap
    dat = sbx_memmap(sbxfl)
    #check if tifs exists
    tifs=[xx for xx in os.listdir(imagingflnm) if ".tif" in xx]
    frames=40000
    nplanes=4
    split = int(zplns/nplanes) # 3000 planes as normal
    if len(tifs)<ceil(frames/zplns): # if no tifs exists 
        #copied from ed's legacy version: loadVideoTiffNoSplit_EH2_new_sbx_uint16        
        for nn,i in enumerate(range(0, dat.shape[0], split)): #splits into tiffs of 3000 planes each
            print(nn)
            stack = np.array(dat[i:i+split,:,:,:])
            #crop in x
            stack=np.squeeze(stack)[:,:,y1:y2,x1:x2] #170:500,105:750] # crop based on etl artifacts                
            # reshape so planes are one after another
            stack = np.reshape(stack, (stack.shape[0]*stack.shape[1], stack.shape[2], stack.shape[3]))
            tifffile.imwrite(sbxfl[:-4]+f'_{nn+1:03d}.tif', stack)
        print("\n ******Tifs made!******\n")    
    else:
        print("\n ******Tifs exists! Run suite2p... ******\n")

    return imagingflnm


imagingflnm = r"\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\E181\Day_04"
y1,y2,x1,x2 = 99,499,109,717
maketifs(imagingflnm,y1,y2,x1,x2,zplns=3000)