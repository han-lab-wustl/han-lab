import os, sys, pickle
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
import preprocessing
#analyze videos and copy vr files before this step
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#TODO: find a way to append df to add more mice

def preprocess(step,vrdir, dlcfls):
    if step == 0:
    #vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    #dlcfls = r'G:\dlc_mixedmodel2' # h5 and csv files from dlc
        mouse_df = preprocessing.copyvr_dlc(vrdir, dlcfls)  
        mouse_df.to_pickle(os.path.join(dlcfls,"mouse_df.p"))  
    return mouse_df

if __name__ == "__main__":
    vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\DLC\dlc_mixedmodel2' # h5 and csv files from dlc
    step=0
    # df = preprocess(step,vrdir,dlcfls)
    # now need to fix vr mat files separately in matlab, lol
    # will not work otherwise!!!
    with open(os.path.join(dlcfls,"mouse_df.p"), "rb") as fp: #unpickle
        df = pickle.load(fp)
    for i,mouse in enumerate(df.index):
        if i>8:
            dlcflss, vrfls = df.iloc[i] # get vr and dlc paired files
            for j,dlcfl in enumerate(dlcflss):                
                # reassign since you copied over the vr files to the dlc files folder
                vrflv2 = os.path.join(dlcfls,os.path.basename(vrfls[j]))
                print(mouse, vrflv2, dlcfl)
                preprocessing.VRalign(vrflv2, dlcfl)
