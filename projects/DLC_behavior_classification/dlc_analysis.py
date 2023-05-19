import deeplabcut, os, pandas as pd
config_path = r'Y:\DLC\MixedModel_trial_2-Adina-2023-03-27\config.yaml'
# deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)
# #OPTIONAL: You can also plot the scoremaps, locref layers, and PAFs:

# deeplabcut.extract_save_all_maps(config_path, shuffle=1, Indices=[0, 5])
# https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html
src = r'Y:\DLC\VR_data\dlc'
rrcsv = pd.read_csv(os.path.join(src,'hrz_videos.csv'))
# analyze videos
deeplabcut.analyze_videos(config_path, rrcsv.video.values, shuffle=1,
        save_as_csv=True, gputouse=0)
# extract outlier frames
# deeplabcut.extract_outlier_frames(config_path, 
#     videos,
#     extractionalgorithm='kmeans')

# deeplabcut.refine_labels(config_path)
# deeplabcut.merge_datasets(config_path)
