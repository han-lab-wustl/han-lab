# Zahra
# preprocessing sbx into tifs (cropping, accounting for multiple planes, etc.)

import os , numpy as np, tifffile, SimpleITK as sitk, sys, shutil
from math import ceil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import makedir, listdir
from distutils.dir_util import copy_tree

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


def copy_folder(src_folder, dest_folder):
    """
    Copies a folder from src_folder to dest_folder, including all subfolders and files.

    Parameters:
    - src_folder: The path to the source folder to be copied.
    - dest_folder: The destination path where the folder should be copied.

    Returns:
    - None
    """
    # Check if the source folder exists
    if not os.path.exists(src_folder):
        print(f"The source folder '{src_folder}' does not exist.")
        return
    
    # Ensure the destination folder exists; if not, create it
    if not os.path.exists(dest_folder): os.mkdir(dest_folder)
    # Copy the entire folder structure and files
    print(f"\n***Folder '{src_folder}' moving to '{os.path.join(dest_folder,os.path.basename(src_folder))}'***")
    shutil.move(src_folder, os.path.join(dest_folder,os.path.basename(src_folder)))    
    # copy excel sheet
    xlsx = os.path.dirname(src_folder)
    xlsx = [xx for xx in listdir(xlsx, ifstring='xlsx')][0]
    shutil.copy(xlsx, os.path.join(dest_folder))
    print(f"\n***Folder {src_folder} and excel sheet has been copied to {os.path.join(dest_folder,os.path.basename(src_folder))} successfully ;)***")
    

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

def maketifs(imagingflnm,y1,y2,x1,x2,nplanes=2,
        zplns=3000):
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
    frames=dat.shape[0]
    split = int(zplns/nplanes) # 3000 planes as normal
    if len(tifs)<ceil(frames/zplns): # if no tifs exists 
        #copied from ed's legacy version: loadVideoTiffNoSplit_EH2_new_sbx_uint16        
        for nn,i in enumerate(range(0, dat.shape[0], split)): #splits into tiffs of 3000 planes each
            stack = np.array(dat[i:i+split,:])
            #crop in x
            if nplanes>1:
                stack=np.squeeze(stack)[:,:,y1:y2,x1:x2] #170:500,105:750] # crop based on etl artifacts                
                    # reshape so planes are one after another
                stack = np.reshape(stack, (stack.shape[0]*stack.shape[1], stack.shape[2], stack.shape[3]))
            else: 
                stack=np.squeeze(stack)[:,y1:y2,x1:x2]            
            tifffile.imwrite(sbxfl[:-4]+f'_{nn+1:03d}.tif', stack)
        print("\n ******Tifs made!******\n")    
    else:
        print("\n ******Tifs exists! Run suite2p... ******\n")

    return imagingflnm

def fillops(ops, params):
    """makes ops dict for suite2p processing
    hardcode s2p params! optimized for zahra's cell tracking pipelin
    Args:
        ops (_type_): default s2p ops
        params (_type_): params dict from run suite2p file (command line args)
    """
    ops["reg_tif"]=params["reg_tif"] # see default settings in params
    ops["nplanes"]=params["nplanes"] 
    ops['fs']=31.25/params["nplanes"] # fs of han lab 2p
    ops['tau']=0.7 # gerardo set?
    ops["delete_bin"]=params["delete_bin"] #False
    ops["move_bin"]=params["move_bin"]
    ops["save_mat"]=params["save_mat"]
    ops["threshold_scaling"]=1 #TODO: make modular
    ops["max_iterations"]=30
    # added temp
    ops["keep_movie_raw"]=1 #TODO: make modular
    ops["two_step_registration"]=1
    # ops["nimg_init"]=500
    # ops["1Preg"]=1
    
    return ops

def fillops_drd(ops, params):
    """makes ops dict for suite2p processing
    hardcode s2p params! optimized for zahra's cell tracking pipelin
    Args:
        ops (_type_): default s2p ops
        params (_type_): params dict from run suite2p file (command line args)
    """
    ops["reg_tif"]=params["reg_tif"] # see default settings in params
    ops["nplanes"]=params["nplanes"] 
    ops['fs']=31.25/params["nplanes"] # fs of han lab 2p
    ops['tau']=0.7 # gerardo set?
    ops["delete_bin"]=params["delete_bin"] #False
    ops["move_bin"]=params["move_bin"]
    ops["save_mat"]=params["save_mat"]
    ops["threshold_scaling"]=1 #TODO: make modular
    ops["max_iterations"]=30
    ops["delete_bin"]=True            
    ops["reg_tif"]=True            

    return ops