%Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e201";
days = [91,92];
src = "Z:\sstcre_imaging";

for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, mouse_name, string(day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fullfile(fmatfl.folder, fmatfl.name));
    disp(savepthfmat)
end
%% 
% for old data
mouse_name = "E135";
days = [1];
src = "F:\";
for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('Day%i',day), sprintf("%s*.mat", mouse_name)));
    planes = 3;
    fmatfl = dir(fullfile(src, mouse_name, sprintf('Day%i',day), '**\plane*\Fall.mat'));     % finds all params files
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), planes, fmatfl);
    disp(savepthfmat)
end