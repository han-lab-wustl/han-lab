%Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e200";
days = [29:35];
src = "Y:\sstcre_imaging";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%% 
% for aligned fmats

clear all;
mouse_name = "e201";
% days = [62:70,72:74,76,81:90];
days = [55:75];
src = "Z:\sstcre_imaging";
fmatsrc = "Y:\sstcre_analysis\fmats";
for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(fmatsrc, mouse_name, "days", sprintf('%s_day%03d*.mat',mouse_name, day))); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% for old data
mouse_name = "E145";
days = [4:7,9:12];
src = "X:\pyramidal_cell_data";
fmatsrc = 'Y:\sstcre_analysis\fmats';
% add function path

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), sprintf("%s*.mat", mouse_name)));    
    fmatfl = dir(fullfile(fmatsrc, mouse_name, 'days', sprintf('%s_day%03d_plane*', ...
    mouse_name, day)));     % finds all params files
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end