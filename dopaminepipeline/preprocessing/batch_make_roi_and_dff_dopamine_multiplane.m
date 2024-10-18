%%%%% batch processing
%%ROI selection and the extraction of raw flourescence
% zahra's notes
% use: primarily for GRAB-DA or other GRAB imaging in 2p

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
% otherwise, runs on dir select for roi selection above
% pr_dir=uipickfiles;
% run dff
base_window=200; % s window over which to calculate baseline
pctile = 0.08; % pctile for baseline calc
[params] = extract_dff_from_ROI_dopamine(pr_dir, base_window, pctile);
%% - step 4 - align to behavior
% based on zahra's directory structure
% for dopamine
close all

for dy=1:length(pr_dir)
    src = pr_dir{dy};
    daypth = dir(fullfile(src, "**\behavior\vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, '**\params.mat')); 
    savepthfmat = VRalign_dopamine_w_opto_events(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end