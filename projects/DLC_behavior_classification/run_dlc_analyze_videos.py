"""
by zahra
additions - March 2024
gets cropped x y coordinates and adds it to config file
then runs dlc analyze and create vid
make sure a copy of the original config file exists!
"""

import deeplabcut, os, pandas as pd
from video_formating.get_crop import get_crop_and_edit_config_file
config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"D:\PupilTraining-Matt-2023-07-07\opto-vids\E218_controls\240219_E217.avi",
        r"D:\PupilTraining-Matt-2023-07-07\opto-vids\E218_controls\240220_E217.avi"]

# note that we have to run in the for loop like this since
# the config file needs to edited separately for each video
for vid in vids:
        # get cropping params and edit config file
        coords = get_crop_and_edit_config_file(vid,config_path)
        # how to use:
        # a new window will pop up of the mouse face
        # click and drag to draw a rectangle around eye
        # press """q""" when done drawing
        # run analyze videos (if needed)
        deeplabcut.analyze_videos(config_path, vid, shuffle=1, #rrcsv.video.values
                save_as_csv=True, gputouse=0)
        # create videos
        deeplabcut.create_labeled_video(config_path,vid, draw_skeleton=True, videotype='.avi',
        displaycropped=True)
        
        