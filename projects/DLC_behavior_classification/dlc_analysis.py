import deeplabcut,os
config_path = 'Y:\DLC\DLC_networks\pupil_licks_nose_paw-Zahra-2023-02-27\config.yaml'
# deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)
# #OPTIONAL: You can also plot the scoremaps, locref layers, and PAFs:

# deeplabcut.extract_save_all_maps(config_path, shuffle=1, Indices=[0, 5])
# https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html
src = r'Y:\DLC\DLC_networks\pupil_licks_nose_paw-Zahra-2023-02-27\videos' 
videos = ['230306_E200.avi', '230222_E200.avi']
videos = [os.path.join(src, xx) for xx in videos]
deeplabcut.extract_outlier_frames(config_path, 
    videos,
    extractionalgorithm='kmeans')

deeplabcut.refine_labels(config_path)
deeplabcut.merge_datasets(config_path)
