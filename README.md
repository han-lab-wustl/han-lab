# han-lab

Project-specific and general scripts from Han Lab @ WUSTL

## hidden_reward_zone_task

`MasterHRZ.m`

Filters pyramidal cells and plots tuning curves, calculates cosine similarity, and makes epoch tables.

## projects > cell_tracking

Goal is to track pyramidal cells detected by Suite2p across weeks/days

Procedure:
1. Concatenate videos of all days per week and run motion registration / ROI detection in Suite2p (done before this step)
2. Use day and week `Fall.mat` output to run CellReg

Relies on installation of [CellReg](https://github.com/zivlab/CellReg)

`format_conversion_Suite2p_CNMF_e.m` to convert Suite2p's output `Fall.mat` into spatial footprints

I would suggest copying `Fall.mat` of all your days/week you want to track into one place for this analysis. I use `copy_fmats.py`.

`run_cellreg_week2week.m`

Cell track across weeks (**map 1**), first block contains all file/folder specific parameters you may need to change

Uses non-rigid transform

Uses the last week (last file in the list) as a reference image, but you may want to change it to a week that has the most cells, etc.

`get_tracked_cells_per_week.m` 

Uses results of the previous run file to get cell indices and plot them on the mean image per day

NOTE: **this relies on a *specific* folder structure**

`run_cellreg_week2day.m`

Cell track cells detected in the weekly concatenated movies across days (**map 2**)

`get_tracked_cells_week2day.m`

Map cells tracked per week back to at least 1 day that week, plot them on the mean image per day, and align to behavior

## projects > axonal-GCamp_dopamine

How to run preprocessing and motion correction on axonal-GCamp images from two-photon

NOTE: **most of Zahra's run scripts take command line arguments**

Relies on some dependencies in Python as well as a downloaded version of [Suite2p](https://github.com/MouseLand/suite2p) in your environment
```
pip install tifffile matplotlib numpy pandas
```

`run_axonal_dopamine_motion_reg.py`

On the command line (on Windows, Anaconda Powershell Prompt), navigate to the `axonal-GCamp_dopamine` folder

Type `python run_axonal_dopamine_motion_reg.py -h` for description of input arguments

To make folder structure:
```
python .\run_axonal_dopamine_motion_reg.py 0 e194 X:\dopamine_imaging\ --day 6
```

0 = step (making folder structure)

e194 = mouse name

X:\dopamine_imaging = drive containing mouse folder and imaging day subfolders within it

6 = day (optional argument); this is what the folder will be named

I suggest having a lookup table of day folder to experiment, imaging notes, camera acquisition etc. in a separate spreadsheet
## utils > utils.py

Contains functions helpful to navigate thru directories, copy `Fall.mat` en masse, and copy VR files