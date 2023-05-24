# Zahra
# mo-seq

import keypoint_moseq as kpms
import matplotlib
matplotlib.use('TkAgg') #might need for gui

project_dir = r'Y:\DLC\demo_project'
config = lambda: kpms.load_config(project_dir)

kpms.update_config(
        project_dir,
        fix_heading=True,
        video_dir='Y:\DLC\eye_videos',
        anterior_bodyparts=['NoseTip'],
        posterior_bodyparts=['BottomLip'],
        use_bodyparts=['EyeNorth','EyeNorthWest','EyeWest','EyeSouthWest',
        'EyeSouth','EyeSouthEast','EyeEast','EyeSouthEast','EyeEast',
        'EyeNorthEast','NoseTopPoint', 'NoseTip', 'NoseBottomPoint',
          'BottomLip', 'MiddleChin', 'WhiskerUpper1', 'WhiskerUpper', 
          'WhiskerUpper3', 'WhiskerLower1', 'WhiskerLower', 'WhiskerLower3', 
          'PawTop', 'PawMiddle', 'PawBottom', 'TongueTop', 'TongueTip',
            'TongueBottom'])
dlc_results = r'Y:\DLC\demo_project\h5s\hrz'
coordinates, confidences, bodyparts = kpms.load_deeplabcut_results(dlc_results)

# format data for modeling
data, labels = kpms.format_data(coordinates, confidences=confidences, 
                                **config())
#wtf is this
kpms.noise_calibration(project_dir, coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

# update latent dim to that explained by most componets
kpms.update_config(project_dir, latent_dim=4)

# optionally update kappa in the config before initializing 
# kpms.update_config(project_dir=project_dir, kappa=NUMBER)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())
