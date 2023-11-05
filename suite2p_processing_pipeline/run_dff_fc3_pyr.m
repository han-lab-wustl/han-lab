% run dff and fc3 calc for all cells
clear all;
days = [15,16];
% falls = dir('Y:\sstcre_analysis\fmats\e200\days\*Fall.mat');
Fs = 31.25;
parfor f=1:length(days)
    fall = dir(fullfile('X:\vipcre\e218', sprintf('%i',days(f)), '**\*Fall.mat'))
    pth = fullfile(fall.folder,fall.name)
    disp(pth)
    create_dff_fc3(pth, Fs)    
end