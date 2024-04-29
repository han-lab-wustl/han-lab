import os, shutil, glob

def copyvrfl_matching_pickle(picklesrc, vrsrc):
    for fl in os.listdir(picklesrc):
        if fl[-2:]=='.p':
            picklefl = os.path.join(picklesrc,fl)
            vrfl = glob.glob(picklefl[:-15]+'*.mat')
            if len(vrfl)==0:
                vrflsearch = fl[:-15]+'*.mat'
                vrflsr = glob.glob(os.path.join(vrsrc, vrflsearch))[0]
                shutil.copy(vrflsr, picklesrc)
    print('\n ************ done copying vr files! ************')
