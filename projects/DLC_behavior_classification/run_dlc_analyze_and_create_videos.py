import deeplabcut, os, pandas as pd
# face videos
config_path = r"D:\MixedMouse_trial_2\MixedModel_trial_2-Adina-2023-03-27\config.yaml"

# path to videos here - CHANGE
# iterates through directories
vids = r"\\storage1.ris.wustl.edu\ebhan\Active\Gerardo\DAvideoLickAnalysisEarlyDaysdays1and2"

deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# crop : # x1: 168
# x2: 305
# y1: 50
# y2: 200
#vid = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\240120_E228.avi"]
deeplabcut.create_labeled_video(config_path,vids,
        videotype='.avi',draw_skeleton=True,
        displaycropped=True)
# #%%
# import deeplabcut, os, pandas as pd

# config_path = r"D:\MixedMouse_trial_2\MixedModel_trial_2-Adina-2023-03-27\config.yaml"
# # path to videos here
# vids = r"D:\PupilTraining-Matt-2023-07-07\Pavlov vids"
# vids = [os.path.join(vids,xx) for xx in os.listdir(vids) if 'avi' in xx]

# deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
#         save_as_csv=True, gputouse=0)

# deeplabcut.create_labeled_video(config_path,vids[20],
#         videotype='.avi',draw_skeleton=True,
#         displaycropped=True)
# #%%
# import deeplabcut, os, pandas as pd
# config_path = r"D:\PupilTraining-Matt-2023-07-07\config.yaml"
# # path to videos here
# vids = r'D:\PupilTraining-Matt-2023-07-07\Pavlov vids'
# vids = [os.path.join(vids,xx) for xx in os.listdir(vids)]
# #vid = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\240120_E217.avi"]

# deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
#         save_as_csv=True, gputouse=0)
# # crop : # x1: 160
# # x2: 250
# # y1: 50
# # y2: 170
# #vid = [r"D:\PupilTraining-Matt-2023-07-07\videos\Adina Videos\240120_E228.avi"]
# deeplabcut.create_labeled_video(config_path,vids,
#         videotype='.avi',draw_skeleton=True,
#         displaycropped=True)


# ffmpeg -i D:\PupilTraining-Matt-2023-07-07\opto-vids\240201_E217.avi -c:v rawvideo D:\PupilTraining-Matt-2023-07-07\opto-vids\240201_E217_conv.avi