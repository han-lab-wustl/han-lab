    """dlc preprocessing scripts
    relies on han-lab repo
    """

import os, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')

def preprocess(step,vrdir, dlcfls):
    if step == 0:
    #vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    #dlcfls = r'G:\dlc_mixedmodel2' # h5 and csv files from dlc
        mouse_df = preprocessing.copyvr_dlc(vrdir, dlcfls)    
    return mouse_df

if __name__ == "__main__":
    vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\DLC\dlc_mixedmodel2' # h5 and csv files from dlc
    step=0
    df = preprocess(step,vrdir,dlcfls)
    # now need to fix vr mat files separately in matlab, lol
    # will not work otherwise!!!
    for i,mouse in enumerate(df.index):
        dlcflss, vrfls = df.iloc[0]
        for j,dlcfl in enumerate(dlcflss):
            print(mouse,j)
            preprocessing.VRalign(os.path.join(dlcfls,os.path.basename(vrfls[j])),
                     dlcfl)
