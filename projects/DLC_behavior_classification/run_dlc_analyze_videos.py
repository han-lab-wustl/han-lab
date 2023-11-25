# zahra
import deeplabcut, os, pandas as pd

config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"I:\eye_videos\230817_T10.avi",
        r"I:\eye_videos\230818_T10.avi",
        r"I:\eye_videos\230822_T10.avi",
        r"I:\eye_videos\230823_T10.avi"]
# run analyze videos (if needed)
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# create videos
# deeplabcut.create_labeled_video(config_path,vids[1], 
#         videotype='.avi',draw_skeleton=True,save_frames = True,displaycropped=True)
        