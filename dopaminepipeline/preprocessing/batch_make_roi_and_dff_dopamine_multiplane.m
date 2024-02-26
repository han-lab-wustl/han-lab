%%%%% batch processing
%%ROI selection and the extraction of raw flourescence
% zahra's notes
% uses primarily for GRAB-DA recordings in 2p

%% step 1 - pick files
clear all
close all
pr_dir=uipickfiles;
days_check=1:length(pr_dir);
fprintf("******* Picking references frames (for one day) is necessary to setup the reference polygons to use on another day *******\n")
fprintf("******* Pick references if you are processing data for an experiment for the first time *******\n")
ref_exists=input('Hit ENTER to skip reference planes setup or any letter/number to setup reference planes: '); %%% if reference image hase been already choosen
if isempty(ref_exists)
    pr_dirref=uipickfiles;%%% chose reference day here day1
else
    pr_dirref=[];
end
%% - step 2 - draw or align ROIs to reference
set_reference_polygons_dopamine(pr_dir, pr_dirref, days_check, ref_exists)
%% - step 3 - calculate dff 
close all
% extract base mean from all rois selected before
% if you wanna pick files again, run the commented out line below
% pr_dir=uipickfiles;
% run dff
[params] = extract_dff_from_ROI_dopamine(pr_dir);
