%Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e221";
days = [1];
src = 'Y:\opto_control_grabda_2m';

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior\vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\params.mat')); 
    savepthfmat = VRalign_dopamine_w_opto_events(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% example pick files
% for ziyi
[filename,path]=uigetfile('*.mat','pick your behavior file');
vrfl = fullfile(path, filename);
[Ffile,Ffilepath]=uigetfile('*.mat','pick the day you want to run (under which you should have params.mat files');
fmatfl = fullfile(Ffilepath, Ffile);
fmatfl = dir(fmatfl, '**\params.mat'); % length is number of plns
savepthfmat = VRalign_dopamine_w_opto_events(vrfl,fmatfl, length(fmatfl));
disp(savepthfmat)
%%
clear all;
mouse_name = "e242";
days = [61,62,63];
src = 'Y:\halo_grabda';

for day=days
    daypth = dir(fullfile(src, mouse_name, string(day), "behavior\vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, string(day), '**\params.mat')); 
    savepthfmat = VRalign_dopamine(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% 2 planes
% has to have combined cells
clear all;
mouse_name = "e217";
days = [12];
src = "X:\vipcre";
planes = 1;
for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\Fall.mat')); 
    fmatfl = fmatfl(1); % combined
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% for dopamine
clear all
mouse_name = "E231";
days = [22];
src = "Z:\";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "**\*time*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\params.mat')); 
    savepthfmat = VRalign_dopamine(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
clear all;
mouse_name = "grabda_sparse_hrz";
days = [9];
src = "\\storage1.ris.wustl.edu\ebhan\Active\dzahra";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\*roibyclick_F.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%% 
% for aligned fmats

clear all;
mouse_name = "e217";
days = [11];
% days = [55:75];
% src = "Z:\sstcre_imaging";
src = "X:\vipcre";
fmatsrc = "Y:\analysis\fmats";
for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(fmatsrc, mouse_name, "days", sprintf('%s_day%03d*.mat',mouse_name, day))); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% for old data
clear all
mouse_name = "e186";
days = [9:51];
src = "Y:\sstcre_analysis\behavior";
fmatsrc = 'Y:\sstcre_analysis\fmats';
% add function path

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), sprintf("%s*.mat", mouse_name)));    
    fmatfl = dir(fullfile(fmatsrc, mouse_name, 'days', sprintf('%s_day%03d_plane*', ...
    mouse_name, day)));     % finds all params files
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end

%%
% for old formatted data
% by day or 'D' folders
mouse_name = "E139";
days = [7:10];
src = "F:\";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('D%i',day), sprintf("%s*.mat", mouse_name)));    
    fmatfl = dir(fullfile(src, mouse_name, sprintf('D%i',day),'**\Fall.mat')) ;     % finds all params files
    % ignore combined fall for behavior alignment
    fmatfl = fmatfl(2:end);
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name), fmatfl, length(fmatfl));
    disp(savepthfmat)
end