import deeplabcut, os, pandas as pd
config_path = r'X:\adina_model_202310\config.yaml'
# deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)
# #OPTIONAL: You can also plot the scoremaps, locref layers, and PAFs:

# deeplabcut.extract_save_all_maps(config_path, shuffle=1, Indices=[0, 5])
# https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html
# src = r'X:\eye_videos'
# # rrcsv = pd.read_csv(os.path.join(src,'hrz_videos.csv'))
# vids = [os.path.join(src,xx) for xx in os.listdir(src) if 'E216' in xx and 'avi' in xx]# analyze videos
# deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
#         save_as_csv=True, gputouse=0)
# extract outlier frames
# deeplabcut.extract_outlier_frames(config_path, 
#     videos,
#     extractionalgorithm='kmeans')

# deeplabcut.refine_labels(config_path)
# deeplabcut.merge_datasets(config_path)

config_path = r'X:\PupilTraining-Matt-2023-07-07\config.yaml'
src = r'X:\eye_videos'
# rrcsv = pd.read_csv(os.path.join(src,'hrz_videos.csv'))
vids = [os.path.join(src,xx) for xx in os.listdir(src) if 'E218' in xx and 'avi' in xx]# analyze videos
vids.sort()
vids = vids[2:]
deeplabcut.analyze_videos(config_path, vids, shuffle=1, #rrcsv.video.values
        save_as_csv=True, gputouse=0)
# crop=[168,305,70,158]
# vid = r"X:\PupilTraining-Matt-2023-07-07\videos\Adina's Videos\231017_E216.avi"
deeplabcut.create_labeled_video(config_path,vids,
        videotype='.avi',draw_skeleton=True,displaycropped=True)