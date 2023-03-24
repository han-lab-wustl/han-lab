# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:59:23 2023

@author: Zahra
"""

import os, shutil
animal = "e201"
src = os.path.join("Z:\sstcre_imaging", animal)
dst = r"Y:\sstcre_analysis\fmats"
# get only days, not week fmats
days = [int(xx) for xx in os.listdir(src) if  "week" not in xx and "ref" not in xx]
weeks = [xx for xx in os.listdir(src) if  "week" in xx and "ref" not in xx]
days.sort()
# move all converted fmats to separate folder
for i in days:
    print(i)
    pth = os.path.join(src, str(i))
    imgfl = [os.path.join(pth, xx) for xx in os.listdir(pth) if "000" in xx][0]
    mat = os.path.join(imgfl, "suite2p", "plane0", "Fall.mat") 
    if os.path.exists(mat):
        shutil.copy(mat, os.path.join(dst, f"{animal}_day{int(i):03d}_Fall.mat"))

if len(weeks)>0:
    for w in weeks:
        print(w)
        imgfl = os.path.join(src, str(w))
        mat = os.path.join(imgfl, "suite2p", "plane0", "Fall.mat") 
        shutil.copy(mat, os.path.join(dst, f"{animal}_week{int(w[4:]):02d}_Fall.mat"))        