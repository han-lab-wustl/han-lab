import deeplabcut, os, pandas as pd

config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"I:\eye_videos\230508_E200.avi",
        r"I:\eye_videos\230509_E200.avi",
        r"I:\eye_videos\230510_E200.avi",
        r"I:\eye_videos\230513_E200.avi"]
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# # crop=[168,305,70,158]
# vid = r"X:\PupilTraining-Matt-2023-07-07\videos\Adina's Videos\231017_E216.avi"
deeplabcut.create_labeled_video(config_path,vids[0],
        videotype='.avi',draw_skeleton=True,
        displaycropped=True)