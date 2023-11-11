# zahra
import deeplabcut, os, pandas as pd

config_path = r"F:\adina_model_202310\config.yaml"
# path to videos here
vids = [r'F:\temp_eye_videos\221222_E186.avi',
r'F:\temp_eye_videos\221223_E186.avi']
# run analyze videos (if needed)
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# create videos
# deeplabcut.create_labeled_video(config_path,vids[1],
#         videotype='.avi',draw_skeleton=True,save_frames = True,displaycropped=True)
        