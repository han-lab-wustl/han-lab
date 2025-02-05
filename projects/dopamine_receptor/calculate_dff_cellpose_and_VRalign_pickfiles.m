% Zahra wrote this script to get cells from cell pose, calc dff, and 
% align to behavior
% also used to plot cells with behavior
%% step 1 - calc dff for only iscell + cells
% Zahra wrote this script to get cells from cell pose, calc dff, and 
% align to behavior
% also used to plot cells with behavior

%% step 1 - calc dff for only iscell + cells
clear all; clear all;

% Prompt the user to select the folder containing the 'plane' folders
src_path = uigetdir('', 'Select source directory for Fall.mat files');

% Find all Fall.mat files in folders starting with 'plane'
Fmat = dir(fullfile(src_path, '**', 'plane*', 'Fall.mat'));

% Calculate the number of planes based on the number of Fall.mat files
num_planes = length(Fmat);
Fs = 31.25 / num_planes;  % Adjust the sampling frequency based on planes
time = 300;               % ms

% Process each Fall.mat file
for i = 1:num_planes
    disp(Fmat(i).folder)
    [dff, f0] = redo_dFF_from_cellpose(fullfile(Fmat(i).folder, Fmat(i).name), Fs, time);
end
%% step 2 - align to behavior
% Prompt the user to select the behavior .mat files
[behav_file, behav_path] = uigetfile('*.mat', 'Select behavior vr .mat file');
fmatfl = Fmat;  % Assuming same Fall.mat files are used

savepthfmat = VRalign(fullfile(behav_path, behav_file), fmatfl, length(fmatfl));
disp(savepthfmat)
