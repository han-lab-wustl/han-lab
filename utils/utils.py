# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:06:02 2023

@author: Han
"""

import os, sys, shutil, tifffile, numpy as np, pandas as pd, re, tarfile
from datetime import datetime
from pathlib import Path

def makedir(dr):
    if not os.path.exists(dr): os.mkdir(dr)
    return dr

def listdir(pth, ifstring=None):
    """prints out complete path of list in directory

    Args:
        pth (_type_): _description_
        ifstring (_type_, optional): _description_. Defaults to None.

    Returns:
        list: list of items in directory with their complete path
    """
    if not ifstring==None:
        lst = [os.path.join(pth, xx) for xx in os.listdir(pth) if ifstring in xx]
    else:
        lst = [os.path.join(pth, xx) for xx in os.listdir(pth)]
    return lst
    
def copyvr(usb, drive, animal, days=False): #TODO: find a way to do the same for clampex
    """copy vr files in bulk to internal drive
    assumes usb is plugged in!!
    but theoretically can copy from any drive to any another drive
    assumes images are copied!!! relies on date of images
    can copy > 1 mat file if multiple sessions recorded per animal per day

    Args:
        usb (str): path to usb drive (e.g. F:\2023_ZD_VR)
        drive (str): path to internal drive (e.g. Z:\sstcre_imaging)
        animal (str): animal name (e.g. e200)
    """
    if not days:
        days = listdir(os.path.join(drive, animal.lower())) # assumes drive > per animal folder structure
        days = [xx for xx in days if "week" not in xx and ".mat" not in xx] #excludes weeks        
    dates = []
    for day in days:
        print(day)
        fls = listdir(day)
        imgfl = [xx for xx in fls if "25" in xx or "ZD" in xx or "24" in xx][0] # change conditional strings if you need!
        date = os.path.basename(imgfl)[:6]
        datetime_object = datetime.strptime(date, '%y%m%d')
        dates.append(str(datetime_object.date()))
    vrfls = listdir(usb,ifstring=animal.upper())
    # matches dates on vr files to imaging dates
    # Regex pattern to match the date format '29_Apr_2024'
    # \d{1,2} matches 1 or 2 digits (for the day)
    # [A-Z][a-z]{2} matches three letters where the first is uppercase (for the month)
    # \d{4} matches exactly four digits (for the year)
    date_pattern = r'\d{1,2}_[A-Z][a-z]{2}_\d{4}'
    # Search for the pattern in the string
    match = re.search(date_pattern, vrfls[0])

    dates_vr = [str(datetime.strptime(re.search(date_pattern, xx).group(), '%d_%b_%Y').date()) for xx in vrfls]
    for flnm,datevr in enumerate(dates_vr):
        if datevr in dates:
            ind = dates.index(datevr)
            dst = os.path.join(days[ind], "behavior", "vr") # relies on Zahra's folder structure
            shutil.copy(vrfls[flnm], dst)
            print(f"*******Copied {vrfls[flnm]} to {dst}*******\n")
    
    return

def ig_f(dirr, files):
    return [f for f in files if os.path.isfile(os.path.join(dirr, f))]

def copydopaminefldstruct(src, dst, days, overwrite=False,
                fldstruct=None):
    """useful for sharing dopamine data
    e.g. dst = r'\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\e231_hrz'
    src = r'Z:\chr2_grabda\e231'
    days = ['cf1','cf2','cf3','cf4','cf5','cf6','cf7','cf8','cf9']
    fldstruct = if you use the scanbox fld as your day fld, arg = '000'
    """
    makedir(dst)
    days = [os.path.join(src,str(xx)) for xx in days]
    # move all converted fmats to separate folder
    for day in days:  
        dst_day = os.path.join(dst,os.path.basename(day))
        if not os.path.exists(dst_day): shutil.copytree(day, dst_day, ignore=ig_f)
        # for zahra, add the scanbox fld
        if fldstruct=='000':
            imgfl1=day
        else:
            imgfl1 = [os.path.join(day, xx) for xx in os.listdir(day) if "000" in xx][0]
        imgfl = [os.path.join(imgfl1, xx) for xx in os.listdir(imgfl1) if "suite2p" in xx][0]
        planes = range(len([xx for xx in listdir(imgfl) if "plane" in xx]))
        # imgfl = pth
        for plane in planes:
            mat = os.path.join(imgfl, f"plane{plane}", "reg_tif", "params.mat") 
            if os.path.exists(mat):
                copypth = os.path.join(dst_day, os.path.basename(imgfl1), "suite2p", f"plane{plane}", "reg_tif")
                # if not os.path.exists(os.path.dirname(copypth)): os.makedirs(os.path.dirname(copypth))
                if os.path.exists(os.path.join(copypth, 'params.mat')) and overwrite==False:
                    print(f"*********Params file for day {day} already exists in {dst}*********")    
                else:
                    shutil.copy(mat, copypth)            
                    print(f"*********Copied {day} Params file to {dst_day}*********")

def copyfmats(src, dst, animal, overwrite=False, days=False, 
            weeks=False, weekdir=False, planes=[0], combined=False):
    """useful for cell tracking, copies Fall to another location for each day in animal folder
    if you align to behavior can also use for further analysis 
    (run runVRalign.m in MATLAB, in projects > SST-cre inhibition)

    Args:
        src (str): drive with raw data and Fall.mat from suite2p, assumes animal folder exists inside it
        dst (str): drive to copy to, e.g.'Y:\\sstcre_imaging\\fmats'
        animal (str): e.g. e200
        days (list of integers): specify list of days(integers) corresponding to fld name
        weeks (list of strings): specify list of weeks(string, e.g. 'week4') corresponding to fld name
    """
    src = os.path.join(src, animal) #src="X:\sstcre_imaging"
    makedir(os.path.join(dst, animal))
    dst = makedir(os.path.join(dst, animal, 'days')) #dst='Y:\\sstcre_analysis\\fmats'
    if weekdir: weekdir = os.path.join(weekdir, animal)
    # get only days, not week fmats
    #dont copy weeks if not specified
    # if not weeks:
    #     weeks = [xx for xx in os.listdir(src) if  "week" in xx and "ref" not in xx]
    # if days==False: days = []
    days = list(days)
    days.sort()
    # move all converted fmats to separate folder
    for i in days:     
        print(i)   
        # pth = os.path.join(src, str(i))
        pth = os.path.join(src, str(i))
        imgfl = [os.path.join(pth, xx) for xx in os.listdir(pth) if "000" in xx][0]
        # imgfl = pth
        if combined:
            mat = os.path.join(imgfl, "suite2p", "combined", "Fall.mat") 
            if os.path.exists(mat):
                copypth = os.path.join(dst, f"{animal}_day{int(i):03d}_plane0_Fall.mat")
                if os.path.exists(copypth) and overwrite==False:
                    print(f"*********Fall for day {i} already exists in {dst}*********")    
                else:
                    shutil.copy(mat, copypth)            
                    print(f"*********Copied day {i} Fall to {dst}*********")
        else:
            for plane in planes:
                mat = os.path.join(imgfl, "suite2p", f"plane{plane}", "Fall.mat") 
                if os.path.exists(mat):
                    copypth = os.path.join(dst, f"{animal}_day{int(i):03d}_plane{plane}_Fall.mat")
                    if os.path.exists(copypth) and overwrite==False:
                        print(f"*********Fall for day {i} already exists in {dst}*********")    
                    else:
                        shutil.copy(mat, copypth)            
                        print(f"*********Copied day {i} Fall to {dst}*********")

    if weeks:
        for w in weeks:            
            if not weekdir: 
                imgfl = os.path.join(src, f'week{w:02d}')
            else:
                imgfl = os.path.join(weekdir, f'week{w:02d}')
            for plane in planes:
                mat = os.path.join(imgfl, "suite2p", f"plane{plane}", "Fall.mat") 
                copypth = os.path.join(dst, f"{animal}_week{w:02d}_plane{plane}_Fall.mat")
                if os.path.exists(copypth) and overwrite==False:
                    print(f"*********Fall for week {w} already exists in {dst}*********")    
                else:
                    shutil.copy(mat, copypth)        
                    print(f"*********Copied week {w} Fall to {dst}*********")
    return 

def copydrdfldstruct(src, dst, days, overwrite=False):
    """Useful for sharing interneuron pipeline data."""
    makedir(dst)
    # days = listdir(src)

    for day in days:  
        day = os.path.join(src,str(day))
        dst_day = os.path.join(dst, os.path.basename(day))
        makedir(dst_day)
        imgfl = [os.path.join(day, xx) for xx in os.listdir(day) if "000" in xx][0]
        imgfl1 = [os.path.join(imgfl, xx) for xx in os.listdir(imgfl) if "suite2p" in xx][0]
        planes = range(len([xx for xx in os.listdir(imgfl1) if "plane" in xx]))
        # find all aligned files
        files_to_copy = [
            f for f in os.listdir(imgfl1)
            if (f.endswith('.mat')  and 'reg' in f)
        ]
        # copy 1
        for file in files_to_copy:
            source_file = os.path.join(imgfl1, file)
            copypth = os.path.join(dst_day, os.path.basename(imgfl), 'suite2p')

            if not os.path.exists(copypth):
                os.makedirs(copypth)

            if os.path.exists(os.path.join(copypth, file)) and not overwrite:
                print(f"*********File {file} for day {day} already exists in {dst}*********")
            else:
                shutil.copy(source_file, copypth)
                print(f"*********Copied {file} from {day} to {dst_day}*********")

        for plane in planes:
            reg_tif_folder = os.path.join(imgfl1, f"plane{plane}")
            if os.path.exists(reg_tif_folder):
                # Find all .mat files ending with 'roibyclick_F' and starting with 'E', and all .jpg files
                files_to_copy=[
                    f for f in os.listdir(reg_tif_folder)
                    if (f.endswith('roibyclick_F.mat') or f.endswith('.mat') or f.startswith('E')) or f.endswith('.jpg')
                ]

                for file in files_to_copy:
                    source_file = os.path.join(reg_tif_folder, file)
                    copypth = os.path.join(dst_day, os.path.basename(imgfl), "suite2p", f"plane{plane}")

                    if not os.path.exists(copypth):
                        os.makedirs(copypth)

                    if os.path.exists(os.path.join(copypth, file)) and not overwrite:
                        print(f"*********File {file} for day {day} already exists in {dst}*********")
                    else:
                        shutil.copy(source_file, copypth)
                        print(f"*********Copied {file} from {day} to {dst_day}*********")

def deletetifs(src,fls=False,keyword='*.tif'):
    """deletes tifs
    useful after you've checked for motion correction

    Args:
        src (str): path to animal folder containing processed data
        keyword (str, optional): folder name. Defaults to 'reg_tif'.
    """
    #src = 'Z:\sstcre_imaging\e201'
    if not fls:
        fls = listdir(src)
    for fl in fls:
        from pathlib import Path
        for path in Path(src).rglob(keyword):
            # deletes reg_tif directory and all its contents
            if 'red' not in str(path) and 'green' not in str(path):
                print(f"\n*** deleting {path}***")
                os.remove(path)
    return 

def deletebinaries(src,fls=False,keyword='data.bin'):
    """deletes tifs
    useful after you've checked for motion correction

    Args:
        src (str): path to animal folder containing processed data
        keyword (str, optional): folder name. Defaults to 'reg_tif'.
    """
    #src = 'Z:\sstcre_imaging\e201'
    if not fls:
        fls = listdir(src)
    for fl in fls:
        from pathlib import Path
        for path in Path(src).rglob(keyword):            
            print(f"\n*** deleting {path}***")
            os.remove(path)
    return 

def deleteregtif(src,fls=[],keyword='reg_tif'):
    """deletes reg_tif folder en masse
    useful after you've checked for motion correction

    Args:
        src (str): path to animal folder containing processed data
        keyword (str, optional): folder name. Defaults to 'reg_tif'.
    """
    #src = 'Z:\sstcre_imaging\e201'
    if len(fls)==0:
        fls = listdir(src)
    else:
        fls = [os.path.join(src,str(fl)) for fl in fls]
    for fl in fls:        
        for path in Path(fl).rglob(keyword):
            # deletes reg_tif directory and all its contents
            print(f"\n*** deleting {path}***")
            shutil.rmtree(path)
    return 

def get_motion_corrected_tifs_from_suite2p_binary(binarypth, dst, 
            Ly=512, Lx=629, chunk=1000):
    """converts suite2p binaries to motion corrected tifs

    Args:
        binarypth (_type_): path to data.bin
        dst (_type_): folder to store tifs
    """
    import suite2p
    # pth = 'Z:\sstcre_imaging\e201\week6\suite2p\plane0\data.bin'
    f_input2 = suite2p.io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=binarypth)
    # dst = 'X:\\week6_e201_motion_corrected'
    makedir(dst)
    for i in range(0,f_input2.shape[0],chunk):
        print(i) # make folder if it does not exist
        tifffile.imwrite(os.path.join(dst, f'file_{i:08d}.tif'), f_input2[i:i+chunk])

def make_tiles_from_scanbox_xyz_tile(sbxsrc,frames_per_plane=30, zsteps=3):
    """makes tiles per z plane to stich in imagej
    assumes row by row from FOV up in 2p
    Args:
        sbxsrc (_type_): _description_
        frames_per_plane (int, optional): _description_. Defaults to 30.
        zsteps (int, optional): _description_. Defaults to 3.
    """
    from sbxreader import sbx_memmap  
# src = r'Z:\sstcre_imaging\e201\0_ref_pln_day\230213_EH_DH_000_003\230213_EH_DH_000_003.sbx'
    dat = sbx_memmap(sbxsrc)
    dat=np.squeeze(dat)
    if len(dat.shape)>3:
        green = dat[:,0,:,:]
        red = dat[:,1,:,:]
    else:
        green = dat
    
def convert_zstack_sbx_to_tif(sbxsrc):
    """converts sbx from zstacks/opto tests to tifs

    Args:
        sbxsrc (_type_): path to sbx file
    """
    from sbxreader import sbx_memmap  
# src = r'Z:\sstcre_imaging\e201\0_ref_pln_day\230213_EH_DH_000_003\230213_EH_DH_000_003.sbx'
    dat = sbx_memmap(sbxsrc)
    dat=np.squeeze(dat)
    if len(dat.shape)>3:
        green = dat[:,0,:,:]
        red = dat[:,1,:,:]
        tifffile.imwrite(sbxsrc[:-4]+"_green.tif", green.astype("uint16"))
        tifffile.imwrite(sbxsrc[:-4]+"_red.tif", red.astype("uint16"))
    else:
        tifffile.imwrite(sbxsrc[:-4]+".tif", dat.astype("uint16"))
    return sbxsrc[:-4]+".tif"

def zip_file(source_file, output_zip):
    """
    Zips a single file.
    
    Parameters:
    source_file (str): The path to the file to be zipped.
    output_zip (str): The name/path of the output zip file.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(source_file, arcname=source_file.split('/')[-1])

def unzip_file(zip_file, output_dir):
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        zipf.extractall(output_dir)

def movesbx(src, dst, fls, fldkeyword='ZD'):
    """useful for moving sbx'es to another drive or to ris archive
        assumes your sbxs are saved within a folder made by scanbox: only true of the newer
        version > 2023
        if older versions of sbx, may need to manually modify based on folder structure
    Args:
        src (_type_): dir with day dir data
        dst (_type_): dest dir
        fldkeyword (str, optional): how your sbx is saved (e.g. 231107_ZD).
        it looks for the folder structure based on this. Defaults to 'ZD'.
    """
    for fl in fls:
        # take days and join paths
        fl = os.path.join(src,str(fl))
        try:
            imgfl = [xx for xx in listdir(fl) if fldkeyword in xx][0]
            sbxfl = [xx for xx in listdir(imgfl) if ".sbx" in xx][0]
            matfl = [xx for xx in listdir(imgfl) if ".mat" in xx][0]
            print(f"\n*** moving {sbxfl}***")
            shutil.move(sbxfl, dst)
            print(f"\n*** copying {matfl}***")
            shutil.copy(matfl, dst)
        except:
            print(f"\n*** no sbx in {fl}***")

def compresssbx_move_to_archive(sbxsrc, dst, compress=True):       
    for fl in fls:
        try:
            if compress:
                imgfl = [xx for xx in listdir(fl) if fldkeyword in xx][0]
                sbxfl = [xx for xx in listdir(imgfl) if ".sbx" in xx][0]
                matfl = [xx for xx in listdir(imgfl) if ".mat" in xx][0]
                # make move folder:
                mvfld =os.path.join(imgfl, 'sbx')
                if not os.path.exists(mvfld): os.mkdir(mvfld)
                shutil.move(sbxfl, mvfld)
                shutil.copy(matfl, mvfld)
                output_zip=os.path.join(imgfl, f'{sbxfl[:-4]}_zip')
                zip_file(mvfld, output_zip)
                print(f"\n*** moving {sbxfl}***")
                shutil.move(sbxfl, dst)
                print(f"\n*** copying {matfl}***")
                shutil.copy(matfl, dst)
            else:
                imgfl = [xx for xx in listdir(fl) if fldkeyword in xx][0]
                sbxfl = [xx for xx in listdir(imgfl) if ".sbx" in xx][0]
                matfl = [xx for xx in listdir(imgfl) if ".mat" in xx][0]
                print(f"\n*** moving {sbxfl}***")
                shutil.move(sbxfl, dst)
                print(f"\n*** copying {matfl}***")
                shutil.copy(matfl, dst)
        except:
            print(f"\n*** no sbx in {fl}***")
    
if __name__ == "__main__":
    usb = r"G:\2023-2026_ZD_VR"
    drives = [r'X:\vipcre', r'X:\vipcre',
              r'X:\vipcre',  r'X:\vipcre', r'X:\chrimson_snc_grabda', r'X:\chrimson_snc_grabda',
               r'X:\chrimson_snc_grabda', r'X:\chrimson_snc_grabda']
            # r'X:\lc_chr2_grabda', r'X:\lc_chr2_grabda', r'X:\lc_chr2_grabda']
    animals = ['z15', 'z17','z16','z14','e291','e292','e293','e294']#,
        # 'e278', 'e279', 'e280']
    for i,drive in enumerate(drives):
        copyvr(usb, drive, animals[i])
    # e.g.
    # src=r'X:\vipcre'
    # dst=r'Y:\analysis\fmats'
    # animal='z17'
    # copyfmats(src,dst,animal,days=[18],overwrite=True,combined=True)
    # animal='z15'
    # copyfmats(src,dst,animal,days=[13],overwrite=True)


