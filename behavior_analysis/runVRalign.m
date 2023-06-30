%Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e200";
days = [88];
src = "Y:\sstcre_imaging";

for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, mouse_name, string(day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%% 
% for old data
mouse_name = "E145";
days = [9:12];
src = "F:\";
% add function path

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('Day%i',day), sprintf("%s*.mat", mouse_name)));    
    fmatfl = dir(fullfile(src, mouse_name, sprintf('Day%i',day), '**\plane*\Fall.mat'));     % finds all params files
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end