import deeplabcut, os, pandas as pd

config_path = r"D:\Tail_Demo-Adina-2023-11-09"
# path to videos here
vids = r'I:\vids_to_analyze\tail'
vids = [os.path.join(vids,xx) for xx in os.listdir(vids)]
#vid = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\240120_E217.avi"]

deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# crop : # x1: 160
# x2: 250
# y1: 50
# y2: 170
#vid = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\240120_E228.avi"]
deeplabcut.create_labeled_video(config_path,vids,
        videotype='.avi',draw_skeleton=True,
        displaycropped=True)


config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# path to videos here
vids = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos"]
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# # crop=[168,305,70,158]
# vid = r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\231103_E218.avi"
deeplabcut.create_labeled_video(config_path,vids,
        videotype='.avi',draw_skeleton=True,
        displaycropped=True)
 
