% run dff and fc3 calc for all cells
clear all;
% days = [62:67,69:70,72:74,76,81:85];
days = [17,18];
% falls = dir('Y:\sstcre_analysis\fmats\e200\days\*Fall.mat');
src = 'X:\vipcre';
% src = 'Z:\sstcre_imaging';
an = 'e218';
Fs = 31.25;
for f=1:length(days) %days, falls depending on format
    fall = dir(fullfile(src, an, sprintf('%i',days(f)), '**\*Fall.mat'));
    pth = fullfile(fall.folder,fall.name);
%     pth = fullfile(falls(f).folder,falls(f).name);    
    disp(pth)
    create_dff_fc3(pth, Fs)    
end