
import os , numpy as np, tifffile, SimpleITK as sitk, sys
from math import ceil
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
imagingflnm = r"\\storage1.ris.wustl.edu\ebhan\Active\DopamineData\E181\Day_02"

import suite2p
ops = suite2p.default_ops() # populates ops with the default options
#edit ops if needed, based on user input
ops["reg_tif"]=True
ops["nplanes"]=4
ops["delete_bin"]=True #False
ops["save_mat"]=True

# provide an h5 path in 'h5py' or a tiff path in 'data_path'
# db overwrites any ops (allows for experiment specific settings)
db = {
    'h5py': [], # a single h5 file path
    'h5py_key': 'data',
    'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
    'data_path': [imagingflnm], # a list of folders with tiffs 
                                            # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                                        
    'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
    # 'fast_disk': 'C:/BIN', # string which specifies where the binary file will be stored (should be an SSD)
    }

# run one experiment
opsEnd = suite2p.run_s2p(ops=ops, db=db)