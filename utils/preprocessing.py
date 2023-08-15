# Zahra
# preprocessing sbx into tifs (cropping, accounting for multiple planes, etc.)

import os , numpy as np, tifffile, SimpleITK as sitk, sys
from math import ceil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import makedir

def makeflds(datadir, mouse_name, day):
    if not os.path.exists(os.path.join(datadir,mouse_name)): #first make mouse dir
        makedir(os.path.join(datadir,mouse_name))
    if not os.path.exists(os.path.join(datadir,mouse_name, day)): 
        print(f"Folder for day {day} of mouse {mouse_name} does not exist. \n\
                Making folders...")
        makedir(os.path.join(datadir,mouse_name, day))
        #behavior folder
        makedir(os.path.join(datadir,mouse_name, day, "behavior"))
        makedir(os.path.join(datadir,mouse_name, day, "behavior", "vr"))
        makedir(os.path.join(datadir,mouse_name, day, "behavior", "clampex")) 
        #cameras (for processed data)
        makedir(os.path.join(datadir,mouse_name, day, "eye"))
        makedir(os.path.join(datadir,mouse_name, day, "tail")) 
        print("\n****Made folders!****\n")

def getmeanimg(pth):
    """coverts tif to mean img

    Args:
        pth (str): path to tif

    Returns:
        tif: meanimg
    """
    reader = sitk.ImageFileReader() # uses sitk for motion corrected 
                                    #mean images because does not work with tiffile>>
    reader.SetFileName(pth)
    image = reader.Execute()

    img = sitk.GetArrayFromImage(image)

    meanimg = np.mean(img,axis=0)
    return meanimg

def maketifs(imagingflnm,y1,y2,x1,x2,dtype='pyramidal',zplns=3000):
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
    if dtype == 'axonal':
        frames=20000
        nplanes=3
    elif dtype == 'pyramidal':
        frames=45000
        nplanes=1
    split = int(zplns/nplanes) # 3000 planes as normal
    if len(tifs)<ceil(frames/zplns): # if no tifs exists 
        #copied from ed's legacy version: loadVideoTiffNoSplit_EH2_new_sbx_uint16        
        for nn,i in enumerate(range(0, dat.shape[0], split)): #splits into tiffs of 3000 planes each
            stack = np.array(dat[i:i+split,:,:,:])
            #crop in x
            if dtype == 'axonal': 
                stack=np.squeeze(stack)[:,:,y1:y2,x1:x2] #170:500,105:750] # crop based on etl artifacts                
                # reshape so planes are one after another
                stack = np.reshape(stack, (stack.shape[0]*stack.shape[1], stack.shape[2], stack.shape[3]))
            elif dtype == 'pyramidal': 
                stack=np.squeeze(stack)[:,y1:y2,x1:x2]            
            tifffile.imwrite(sbxfl[:-4]+f'_{nn+1:03d}.tif', stack)
        print("\n ******Tifs made!******\n")    
    else:
        print("\n ******Tifs exists! Run suite2p... ******\n")

    return imagingflnm