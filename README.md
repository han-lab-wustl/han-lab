# han-lab

Project-specific and general scripts from Han Lab @ WUSTL

**For a lot of analysis scripts involving pyramidal cells, you will need to add the entire directory as well as the directory of 'han-lab-archive' to path in MATLAB. This is being updated to allow depreciation of han-lab-archive scripts**

## suite2p install
This suite2p install is optimized to avoid errors compared to other suite2p installs we have tested.

In anaconda powershell:
```
conda create -n suite2p python=3.9
pip install suite2p==0.12.1 pyqtgraph pyqt5
notepad C:\Users\workstation2\anaconda3\envs\suite2p\lib\site-packages\suite2p\gui\visualize.py
```
edit `from rastermap.mapping` --> `from rastermap`

## video making install

To convert tifs from bonsai to lossless compression avis, following the instructions to install ffmpeg on Windows:
https://www.wikihow.com/Install-FFmpeg-on-Windows

Make sure this is added to path!

Then, install ffmpeg-python & other dependencies in your conda environment: `pip install ffmpeg-python SimpleITK`

Follow the scripts in `projects/DlC_behavior_formatting/video_formatting` for the pipeline
1. `curate_tif_to_video_conversion.py` --> gets # of files per folder to make sure they are aligned with imaging (manually check and remove duplicates)
2. `convert_tif_to_avi` --> takes tif folder, makes memory mapped array, and converts to avi

## general lab pipeline 

### dopamine release imaging
make sure the repo is added to your MATLAB path with subfolders!

- make tifs: `han-lab\utils\preprocessing_2p_images\runVideosTiff_EH_new_sbx1.m` OR `han-lab\utils\preprocessing_2p_images\opto_correction\runVideosTiff_Opto.m`
if using `runVideosTiff`, change the MATLAB path in `loadVideoTiffNoSplit...` to your MATLAB directory:
e.g. from 
```
javaaddpath 'C:\Program Files\MATLAB\R2017b\java\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2017b\java\ij.jar'
```
to 
```
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\ij.jar'
```
- run suite2p (and set appropriate settings for planes, frame rate, saving tifs, etc.)
```
conda activate suite2p
suite2p
```
- run dopamineROI pipeline: `"han-lab\dopaminepipeline\preprocessing\batch_make_roi_and_dff_dopamine_multiplane.m`
  - pick day files
  - draw ROIs or recall from a previous day's ROIs
  - calculate $\Delta$ F/F
- align to behavior
- do downstream analysis with align `params.mat` file in each plane

### dopamine receptor imaging
make sure the repo is added to your MATLAB path with subfolders!

- make tifs: step 1 of `suite2p_processing_pipeline\run_suite2p_drd.py` OR `han-lab\utils\preprocessing_2p_images\runVideosTiff_EH_new_sbx1.m`
if using `runVideosTiff`, change the MATLAB path in `loadVideoTiffNoSplit_EH2_new_sbx_uint16.m` to your MATLAB directory:
e.g. from 
```
javaaddpath 'C:\Program Files\MATLAB\R2017b\java\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2017b\java\ij.jar'
```
to 
```
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\ij.jar'
```
- run suite2p (and set appropriate settings for planes, frame rate, saving tifs, etc.)
```
conda activate suite2p
suite2p
```
- run InterneuronPipeline: `projects\dopamine_receptor\pipeline`
  - pick day files
  - make .mat of movies
  - click rois
- align to behavior
- do downstream analysis with align `F.mat` file

## behavior_analysis

General scripts for plotting behavioral variables in Pavlovian conditioning/HRZ. Mostly used for monitoring behavior in Pavlovian conditioning.

`COMgeneralview_multiple_EBGMEH1602.m`: Plots HRZ behavior using multiple VR files across multiple days. Relies on `COMgeneralanalysis`.
` 
`Comgeneralview_EBGMEH1602`: does all the graphs we've ever made for behavior to view
  
`correct_artifact_licks`: not sure if needed, should remove licks in the dark
 
`COMgeneralanalysis`: All of our actual analysis, used in most scripts. Separated for the sake of space and convenience for debugging.
 
`COMgeneralPlotMini`: just behavior subplot
 
`COMgeneralviewJustAbsolutelickDist`: Plots behavior with just the absolute lick distance ttest
 
`COMsmallversion`: (note! uses specific days I know are good to test out, can select others but will post those if you want to use the same) attempts all the different versions and shows how the statistical tests compare accross each day, skips behavior plots
 
`mtit`: if you have matlab earlier than 2018(?) will use this function to put a title on a subplotted figure

## hidden_reward_zone_task

`MasterHRZ.m`

Eleonora's HRZ (and remapping) analysis

Filters pyramidal cells and plots tuning curves, calculates cosine similarity, and makes epoch tables.

## suite2p_processing_pipeline

Zahra's wrappers around suite2p to make tifs, run motion corr, get ROIs

## place_cell_pipeline

Zahra's scripts to get place cells in HRZ, based off Suyash and Tank lab code 

## projects > cell_tracking

Goal is to track pyramidal cells detected by Suite2p across weeks/days

Procedure:
1. Use day `Fall.mat` output to run CellReg

Relies on installation of [CellReg](https://github.com/zivlab/CellReg)

`format_conversion_Suite2p_CNMF_e.m` to convert Suite2p's output `Fall.mat` into spatial footprints

I would suggest copying `Fall.mat` of all your daysyou want to track into one place for this analysis. I use `copy_fmats.py`.

Uses non-rigid transform!

## projects > dopamine

`Batch_SP_SO_SR_days_GM3_new.m`

Munni's original code. Allows drawing/selection of a ROI and extraction of dFF.

`VRdarkrewards_init_analysis5v2_addrewNew.m`

Munni's original code that aligns behavioral events to dFF. Modifications for labels/titles made by ZD.

`pre_post_diff_anat_darkreward_single_rew.m`

Plots a heatmap of CS-triggered averages across days in the Pavlovian task.

## utils

`runVideosTiff_EH_new_sbx_2channel.m`, `loadVideoTiffNoSplit_EH2_new_sbx_2channel.m`

Visualize 2 channel sbx files in MATLAB. Useful for optogenetics or tracing experiments where 2 channels of fluorophores exist in the images.

`viewgreenandredzstackssbx.m`

`MacroconverttoAVI.ijm`

Converts tiffs to avi in ImageJ. Useful for converting behavior videos into avis for DLC. Use in regular mode.
`input` = directory with directories of tifs 
`output` = directory to store avis
`name` = regex to filter by animal

### utils > utils.py

Contains functions helpful to navigate thru directories, copy `Fall.mat` en masse, and copy VR files
