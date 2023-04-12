%Zahra
%run VR align
%based on Zahra's pipeline folder structure

mouse_name = "e201";
days = [32:34];
src = "Z:\sstcre_imaging";
% add function path
addpath(fullfile(pwd, "utils"));
for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, mouse_name, string(day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fullfile(fmatfl.folder, fmatfl.name));
    disp(savepthfmat)
end
%% 
% for dopamine mice
mouse_name = "e194";
days = [14:17];
src = "X:\dopamine_imaging";
% add function path
addpath(fullfile(pwd, "utils"));
for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior", "vr\*.mat"));
    planes = 3;
    fmatfl = dir(fullfile(src, mouse_name, string(day), '**\params.mat'));     % finds all params files
    savepthfmat = VRalign_dopamine(fullfile(daypth.folder, daypth.name),planes, fmatfl);
    disp(savepthfmat)
end