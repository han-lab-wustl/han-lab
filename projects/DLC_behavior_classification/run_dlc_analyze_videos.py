# zahra
import deeplabcut, os, pandas as pd

config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"D:\PupilTraining-Matt-2023-07-07\Pavlov vids\230323_E200.avi",
        r"D:\PupilTraining-Matt-2023-07-07\Pavlov vids\240223_E215.avi",
        r"D:\PupilTraining-Matt-2023-07-07\Multi reward vids\230321_E201.avi",
        r"D:\PupilTraining-Matt-2023-07-07\Multi reward vids\230406_E200.avi",
        r"D:\PupilTraining-Matt-2023-07-07\Multi reward vids\231020_E216.avi"]
# run analyze videos (if needed)
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# create videos
# deeplabcut.create_labeled_video(config_path,vids[1], 
#         videotype='.avi',draw_skeleton=True,save_frames = True,displaycropped=True)
        