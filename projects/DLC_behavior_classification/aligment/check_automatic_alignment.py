"""
zahra's check system for vralign
if doesn't look right, run manual alignment
"""
import pickle, os, sys
sys.path.append(r"C:\Users\workstation2\Documents\MATLAB\han-lab")
from utils.utils import listdir
src =  r"I:\vids_to_analyze\face_and_pupil\pupil"
picklepth = listdir(src, ifstring="_vr_dlc_align.p")
for pth in picklepth:
    with open(pth, "rb") as fp: #unpickle
            pdf = pickle.load(fp)
    
    print(os.path.basename(pth),pdf["start_stop"])

# checked alignment of all in src =  r"I:\vids_to_analyze\face_and_pupil\face"

########## NEED TO MANUAL ALIGN in src =  r"I:\vids_to_analyze\face_and_pupil\face" #################
# E216_20_Jan_2024_vr_dlc_align
# E218_01_Nov_2023_vr_dlc_align
# E218_15_Nov_2023_vr_dlc_align
# e218_16_Nov_2023_vr_dlc_align