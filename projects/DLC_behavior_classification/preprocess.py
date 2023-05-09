    """dlc preprocessing scripts
    relies on han-lab repo
    """

import os, sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils import utils 
#analyze videos and copy vr files before this step

def preprocess(step,vrdir, dlcfls):
    if step == 0:
    #vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    #dlcfls = r'G:\dlc_mixedmodel2' # h5 and csv files from dlc
        mouse_df = utils.copyvr_dlc(vrdir, dlcfls)

if __name__ == "__main__":
    vrdir =  r'I:\VR_data' # copy of vr data, curated to remove badly labeled files
    dlcfls = r'Y:\dlc_mixedmodel2' # h5 and csv files from dlc
    step=0
    preprocess(step,vrdir,dlcfls)