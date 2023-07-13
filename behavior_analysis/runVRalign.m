%Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e201";
days = [99];
src = "Z:\sstcre_imaging";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%% 
% for old data
mouse_name = "E145";
days = [4:7,9:12];
src = "F:\";
% add function path

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('Day%i',day), sprintf("%s*.mat", mouse_name)));    
    fmatfl = dir(fullfile(src, mouse_name, sprintf('Day%i',day), '**\plane*\Fall.mat'));     % finds all params files
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end