# zahra
# additions - March 2024
# gets cropped x y coordinates and adds it to config file
# make sure a copy of the original config file exists!

import deeplabcut, os, pandas as pd
from video_formating.get_crop import get_crop_and_edit_config_file
config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"I:\vids_to_analyze\pupil_light_world\240202_E217.avi"]

for vid in vids:
        # get cropping params and edit config file
        x1,x2,y1,y2 = get_crop_and_edit_config_file(vid,config_path)
        # how to use:
        # draw a rectangle around eye
        # press q when done drawing
        # run analyze videos (if needed)
        deeplabcut.analyze_videos(config_path, vid, shuffle=1, #rrcsv.video.values
                save_as_csv=True, gputouse=0)
        # create videos
        deeplabcut.create_labeled_video(config_path,vid, 
        videotype='.avi',draw_skeleton=True,save_frames = True,
        displaycropped=True)
        