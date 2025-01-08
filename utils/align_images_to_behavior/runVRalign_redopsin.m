% example pick files
% for ziyi
[filename,path]=uigetfile('*.mat','pick your behavior file');
vrfl = fullfile(path, filename);
[fmatfl]=uigetdir('pick the day folder you want to run (under which you should have params.mat files');
% fmatfl = fullfile(Ffilepath, Ffile);
fmatfl = dir(fullfile(fmatfl, '**\params.mat')); % length is number of plns
savepthfmat = VRalign_dopamine_w_opto_events(vrfl,fmatfl, length(fmatfl));
disp(savepthfmat)
