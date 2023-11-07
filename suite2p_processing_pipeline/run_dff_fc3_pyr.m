% run dff and fc3 calc for all cells
clear all;
days = [50:54];
% falls = dir('Y:\sstcre_analysis\fmats\e200\days\*Fall.mat');
% src = 'X:\vipcre\e218';
src = 'Z:\sstcre_imaging';
an = 'e201';
Fs = 31.25;
for f=1:length(days)
    fall = dir(fullfile(src, an, sprintf('%i',days(f)), '**\*Fall.mat'));
    pth = fullfile(fall.folder,fall.name);
    disp(pth)
    create_dff_fc3(pth, Fs)    
end