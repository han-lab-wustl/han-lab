 %Zahra
%run VR align
%based on Zahra's pipeline folder structure
% run from han-lab dir
clear all;
mouse_name = "e217";
days = [47];
src = "X:\vipcre";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\Fall.mat')); 
    savepthfmat = VRalign(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
% for dopamine
clear all
mouse_name = "e232";
days = [74];
src = "Z:\chr2_grabda";

for day=days
    daypth = dir(fullfile(src, mouse_name, sprintf('%i',day), "behavior", "vr\*.mat"));
%     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
    fmatfl = dir(fullfile(src, mouse_name, sprintf('%i',day), '**\params.mat')); 
    savepthfmat = VRalign_dopamine(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
%%
clear all;
mouse_name = "e217";
days = [8:10];
src = "Y:\hrz_consolidation";

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
mouse_name = "e200";
days = [90];
% days = [55:75];
% src = "Z:\sstcre_imaging";
src = "Z:\sstcre_imaging";
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