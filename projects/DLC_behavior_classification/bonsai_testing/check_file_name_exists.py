# -*- coding: utf-8 -*- 
"""
Created on Fri Feb 24 15:45:37 2023

@author: Zahra
"""

import os, sys, shutil, tifffile, ast, time
import argparse, numpy as np

def main(**args):
    
    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)    
    if os.path.exists(params['filename']):
        print('This file name already exists! Enter a different string name')

def fill_params(filename):

    params = {}

    #slurm params
    params["filename"]      = filename
    
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
    
    parser.add_argument("filename", type=str,
                        help="directory name")
    
    args = parser.parse_args()
    
    main(**vars(args))