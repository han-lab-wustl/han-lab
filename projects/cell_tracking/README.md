## projects > cell_tracking

Goal is to track pyramidal cells detected by Suite2p across weeks/days

Procedure:
1. Concatenate videos of all days per week and run motion registration / ROI detection in Suite2p (done before this step)

- I suggest using `run_suite2p_celltrack.py` in projects > SST-Cre_inhibition to easily concatenate weekly videos without copy pasting the tifs into 1 directory for suite2p
- I also suggest cropping tifs (if you need to crop) of all days in an identical way to ensure the dimensions are the same across tracking
- suite2p params are set in `run_suite2p_celltrack.py` as a threshold of 0.5 and max_iter of 200 (see suite2p docs for details). These are more permissive than the normal params, and worked well for my data with sparser and dimmer labeling. I would suggest doing a parameter sweep if you are not happy with the results for these params
- `run_suite2p_celltrack.py` relies on a folder structure of scanbox folder > scanbox sbx and tif. It finds the tif assuming this hierarchy. If you have a different folder structure, I would suggest modifying it to this so you don't have to change the paths in the source code. 

4. Use day and week `Fall.mat` output to run CellReg

Relies on installation of [CellReg](https://github.com/zivlab/CellReg)

`format_conversion_Suite2p_CNMF_e.m` to convert Suite2p's output `Fall.mat` into spatial footprints

I would suggest copying `Fall.mat` of all your days/week you want to track into one place for this analysis. I use `copy_fmats.py`.

`run_cellreg_week2week.m`

Cell track across weeks (**map 1**), first block contains all file/folder specific parameters you may need to change

Uses non-rigid transform

Uses the last week (last file in the list) as a reference image, but you may want to change it to a week that has the most cells, etc.

`get_tracked_cells_per_week.m` 

Uses results of the previous run file to get cell indices and plot them on the mean image per day

- Here you can choose whether you want to keep cells tracked across all weeks, cells tracked across >2 weeks, etc. by changing the parameter in the script.

NOTE: **this relies on a *specific* folder structure**

`run_cellreg_week2day.m`

Cell track cells detected in the weekly concatenated movies across days (**map 2**)

`get_tracked_cells_week2day.m`

Map cells tracked per week back to at least 1 day that week (this can be changed in the script), plot them on the mean image per day, and align to behavior
