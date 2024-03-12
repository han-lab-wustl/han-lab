# -*- coding: utf-8 -*- 
"""
Created on Fri Feb 24 15:45:37 2023

@author: Zahra
"""

import os, sys, shutil, tifffile, ast, time
import argparse   
import pandas as pd, numpy as np
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom your clone
from utils.utils import makedir 
from utils import preprocessing

def main(**args):
    
    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)    
    if args["stepid"] == 0:
        ###############################MAKE FOLDERS#############################
        #check to see if day directory exists
        preprocessing.makeflds(params["datadir"], params["mouse_name"], params["day"])
        ##########################TRANSFER (COPY) DATA##########################
        preprocessing.copy_folder(params["transferdir"], os.path.join(params["datadir"], params["mouse_name"], params["day"]))

    elif args["stepid"] == 1:
        ####CHECK TO SEE IF FILES ARE TRANSFERRED AND MAKE TIFS/RUN SUITE2P####
        #args should be the info you need to specify the params
        # for a given experiment, but only params should be used below        
        
        print(params)
        #check to see if imaging files are transferred
        imagingfl=[xx for xx in os.listdir(os.path.join(params["datadir"],
                                        params["mouse_name"], params["day"])) if "000" in xx][0]
        imagingflnm=os.path.join(params["datadir"], params["mouse_name"], params["day"], 
                imagingfl)
        if not params["cell_detect_only"]: 
            # if cell detect only not specified, make tifs, else skip
            if len(imagingfl)!=0:           
                print(imagingfl)
                if params["crop_opto"]:
                    imagingflnm = preprocessing.maketifs(imagingflnm,89,512,89,718)
                else:
                    imagingflnm = preprocessing.maketifs(imagingflnm,0,512,89,718)            
                print(imagingflnm)

        #do suite2p after tifs are made
        # set your options for running
        import suite2p
        ops = suite2p.default_ops() # populates ops with the default options
        #edit ops if needed, based on user input
        ops = preprocessing.fillops(ops, params)
        ops["threshold_scaling"]=1 #TODO: make modular
        ops["max_iterations"]=30
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
        save_params(params, imagingflnm)

    elif args["stepid"] == 2:
        print(params["days_of_week"])
        ########################WEEKLY CONCATENATED SUTIE2P RUN########################
        dayflds = [os.path.join(params["datadir"],params["mouse_name"], str(day)) for day in params["days_of_week"]]
        imgpths = [os.path.join(dayfld, xx) for dayfld in dayflds for xx in os.listdir(dayfld) if "000" in xx]
        if not params["cell_detect_only"]: 
                # if cell detect only not specified, make tifs, else skip
            for imgpth in imgpths:
                #assumes that these tifs have already been made in step 1
                tifspth = [os.path.join(imgpth, xx) for xx in os.listdir(imgpth) if ".tif" in xx]
                if len(tifspth)==0:
                    print("\n writing tifs....")     
                    if params["crop_opto"]:
                        preprocessing.maketifs(imgpth,89,512,89,718)
                    else:
                        preprocessing.maketifs(imgpth,0,512,89,718)
                    print(imgpth)
        # otherwise concat all tifs
        tifspths = [os.path.join(imgpth, xx) for imgpth in imgpths for xx in os.listdir(imgpth) if ".tif" in xx]
        tifspths.sort(); print(tifspths)
        #savedir
        params["savedir"] = "X:\weekly_concat" #TODO: make var
        weekdir = os.path.join(params["savedir"],params["mouse_name"], \
            "week"+str(params["week"])); makedir(weekdir)    
        #do suite2p after tifs are made
        # set your options for running
        import suite2p
        ops = suite2p.default_ops() # populates ops with the default options
        #edit ops if needed, based on user input
        ops = preprocessing.fillops(ops, params)
        ops['nimg_init'] = 2000
        # ops['th_badframes'] = 0 # prevents cropping of fov, see s2p docs, useful for weekly concatenated videos 
        # bc you might have one day with bad motion corr but it shouldn't drastically affect the rois?
        # provide an h5 path in 'h5py' or a tiff path in 'data_path'
        # db overwrites any ops (allows for experiment specific settings)
        db = {
            'h5py': [], # a single h5 file path
            'h5py_key': 'data',
            'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
            'data_path': imgpths, # a list of folders with tiffs 
                                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                                                
            'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
            'save_path0': weekdir
        }

        # run one experiment
        opsEnd = suite2p.run_s2p(ops=ops, db=db)
        save_params(params, weekdir)

    elif args["stepid"] == 3:
        #####################RUN DAY CELL DETECTION IN A LOOP#####################
        
        ####CHECK TO SEE IF FILES ARE TRANSFERRED AND MAKE TIFS/RUN SUITE2P####
        #args should be the info you need to specify the params
        # for a given experiment, but only params should be used below        
        print(params["days"])
        for day in params['days']: 
            # make sure fl exists on this day
            print(f"\n ************Processing day {day}************")
            imagingfl=[xx for xx in os.listdir(os.path.join(params["datadir"], 
                    params["mouse_name"], day)) if "000" in xx][0]
            imagingflnm=os.path.join(params["datadir"], params["mouse_name"], day, 
                                    imagingfl)
            
            if not params["cell_detect_only"]: 
                # if cell detect only not specified, make tifs, else skip
                if len(imagingfl)!=0:           
                    print(imagingfl)
                    if params["crop_opto"]:
                        imagingflnm = preprocessing.maketifs(imagingflnm,89,512,89,718)
                    else:
                        imagingflnm = preprocessing.maketifs(imagingflnm,0,512,89,718)            
                    print(imagingflnm)

            #do suite2p after tifs are made
            # set your options for running
            import suite2p
            ops = suite2p.default_ops() # populates ops with the default options
            #edit ops if needed, based on user input
            ops = preprocessing.fillops(ops, params)
            # temp
            ops["threshold_scaling"]=1 #TODO: make modular
            ops["max_iterations"]=30
            # test for e216
            # ops["allow_overlap"] = True
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
            save_params(params, imagingflnm)
        return False
        #save_params(params, )

def fill_params(mouse_name, day, days, datadir, transferdir, reg_tif, nplanes, delete_bin,
                move_bin, stepid, save_mat, crop_opto,
                days_of_week, week, cell_detect_only):

    params = {}

    #slurm params
    params["stepid"]        = stepid
    
    #experiment params
    params["datadir"]       = datadir           #main dir
    params["mouse_name"]    = mouse_name        #mouse name w/in main dir
    params["day"]           = day               #session no. w/in mouse name  
    params["transferdir"]   = transferdir       #name of external mounted on comp
    try: #TODO: fix error
        params["days_of_week"]  = days_of_week[0]   #days to put together for analysis of that week
    except:
        print("\n No days of week specified...\n")
    try: #TODO: fix error
        params["days"]  = days[0]   #days to put together for analysis of that week
    except:
        print("\n Multiple days not specified...\n")
    params["week"]          = week              #week np.
    #suite2p params
    params["reg_tif"]       = ast.literal_eval(reg_tif)
    params["nplanes"]       = nplanes
    params["delete_bin"]    = delete_bin
    params["move_bin"]      = move_bin
    params["save_mat"]      = save_mat
    params["crop_opto"]     = crop_opto
    params["cell_detect_only"] = cell_detect_only
        
    return params

def save_params(params, dst):
    """ 
    save params in parameter dictionary for reconstruction/postprocessing 
    """
    (pd.DataFrame.from_dict(data=params, orient="index").to_csv(os.path.join(dst, "param_dict.csv"),
                            header = False))
    sys.stdout.write("\nparameters saved in: {}".format(os.path.join(dst, "param_dict.csv"))); sys.stdout.flush()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("stepid", type=int,
                        help="Step ID to run folder name, suite2p processing, cell tracking")
    parser.add_argument("mouse_name",
                        help="e.g. E200")
    parser.add_argument("datadir", type=str,
                        help="Main directory with mouse names and days")
    parser.add_argument("--transferdir", type=str,
                        help="External where you are transfering data from, has to be full path to folder \n\
                            e.g. H:\imaging_backups\ 240306_ZD\ 240306_ZD_000_000")
    parser.add_argument("--day", type=str, default = '1',
                        help="day of imaging")
    parser.add_argument("--days", nargs="+", action = "append",
                        help="For step1, if you want to run registration in a loop \n\
                            e.g. 1 2 3")
    parser.add_argument("--days_of_week",  nargs="+", action = "append",
                        help="For step 2, if running weekly concatenated videos, \n\
                            specify days of the week (integers) \n\
                            e.g. 1 2 3")
    parser.add_argument("--week", type=int, default = '1',
                        help="For step 2, week no.")                        
    parser.add_argument("--reg_tif", default='False',
                        help="Whether or not to save move corrected imagings")
    parser.add_argument("--nplanes", default=1, type=int,
                        help="Number of planes imaged")
    parser.add_argument("--delete_bin", default=False,
                        help="Delete data.bin to run suite2p")
    parser.add_argument("--move_bin", default=False,
                        help="Move data.bin from fast disk")
    parser.add_argument("--save_mat", default=True,
                        help="Save Fall.mat (needed for cell tracking)")    
    parser.add_argument("--crop_opto", default=False,
                        help="Crop top band of opto to not mess up cell detection")    
    parser.add_argument("--cell_detect_only", default=False,
                        help="Skip making tifs and reg, only run cell detect")    
    
    
    args = parser.parse_args()
    
    main(**vars(args))