# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:06:02 2023

@author: Han
"""

import os, sys, shutil 
from datetime import datetime

def makedir(dr):
    if not os.path.exists(dr): os.mkdir(dr)
    return 

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
    

def copyvr(usb, drive, animal): #TODO: find a way to do the same for clampex
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
    days = listdir(os.path.join(drive, animal.lower())) # assumes drive > per animal folder structure
    days = [xx for xx in days if "week" not in xx] #excludes weeks
    dates = [];
    for day in days:
        fls = listdir(day)
        imgfl = [xx for xx in fls if "23" in xx or "ZD" in xx][0] # change conditional strings if you need!
        date = os.path.basename(imgfl)[:6]
        datetime_object = datetime.strptime(date, '%y%m%d')
        dates.append(str(datetime_object.date()))
    vrfls = listdir(usb,ifstring=animal.upper())
    # matches dates on vr files to imaging dates
    dates_vr = [str(datetime.strptime(os.path.basename(xx)[5:16], '%d_%b_%Y').date()) for xx in vrfls]
    for flnm,datevr in enumerate(dates_vr):
        if datevr in dates:
            ind = dates.index(datevr)
            dst = os.path.join(days[ind], "behavior", "vr") # relies on Zahra's folder structure
            shutil.copy(vrfls[flnm], dst)
            print(f"*******Copied {vrfls[flnm]} to {dst}*******\n")
    
    return


def copyfmats(src, dst, animal, overwrite=False):
    """useful for cell tracking, copies Fall to another location for each day in animal folder
    if you align to behavior can also use for further analysis 
    (run runVRalign.m in MATLAB, in projects > SST-cre inhibition)

    Args:
        src (str): drive with raw data and Fall.mat from suite2p, assumes animal folder exists inside it
        dst (str): drive to copy to, e.g.'Y:\\sstcre_imaging\\fmats'
        animal (str): e.g. e200
    """
    src = os.path.join(src, animal) #"Z:\sstcre_imaging"
    dst = os.path.join(dst, animal) #"dst='Y:\\sstcre_analysis\\fmats
    # get only days, not week fmats
    days = [int(xx) for xx in os.listdir(src) if  "week" not in xx and "ref" not in xx]
    weeks = [xx for xx in os.listdir(src) if  "week" in xx and "ref" not in xx]
    days.sort()
    # move all converted fmats to separate folder
    for i in days:        
        pth = os.path.join(src, str(i))
        imgfl = [os.path.join(pth, xx) for xx in os.listdir(pth) if "000" in xx][0]
        mat = os.path.join(imgfl, "suite2p", "plane0", "Fall.mat") 
        if os.path.exists(mat):
            copypth = os.path.join(dst, f"{animal}_day{int(i):03d}_Fall.mat")
            if os.path.exists(copypth) and overwrite==False:
                print(f"*********Fall for day {i} already exists in {dst}*********")    
            else:
                shutil.copy(mat, copypth)            
                print(f"*********Copied day {i} Fall to {dst}*********")

    if len(weeks)>0:
        for w in weeks:            
            imgfl = os.path.join(src, str(w))
            mat = os.path.join(imgfl, "suite2p", "plane0", "Fall.mat") 
            copypth = os.path.join(dst, f"{animal}_week{int(w[4:]):02d}_Fall.mat")
            if os.path.exists(copypth) and overwrite==False:
                print(f"*********Fall for day {i} already exists in {dst}*********")    
            else:
                shutil.copy(mat, copypth)        
                print(f"*********Copied {w} Fall to {dst}*********")
    return 
