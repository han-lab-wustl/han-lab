% Zahra wrote this script to get cells from cell pose, calc dff, and 
% align to behavior
% also used to plot cells with behavior
%% step 1 - calc dff for only iscell + cells
clear all; clear all;
src = 'Y:\drd2';
mouse_name = 'e256';
days = [3];
plns = [0:2];
Fs = 31.25/length(plns);
time = 300; % ms
for dy=days % days
    Fmat  = dir(fullfile(src, mouse_name, string(dy), '**', 'plane*', 'Fall.mat'));
    % iterate through all planes
    for i=1:length(Fmat)
        disp(Fmat(i).folder)
        [dff,f0] = redo_dFF_from_cellpose(fullfile(Fmat(i).folder, Fmat(i).name), ...
            Fs, time);    
    end
end
%% step 2 - align to behavior
for dy=days
    daypth = dir(fullfile(src, mouse_name, string(dy), "behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, mouse_name, string(dy), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end