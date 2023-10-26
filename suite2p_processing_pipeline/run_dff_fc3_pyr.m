% run dff and fc3 calc for all cells
clear all;
falls = dir('Y:\sstcre_analysis\fmats\e200\days\*Fall.mat');
Fs = 31.25;
parfor f=1:length(falls)
    pth = fullfile(falls(f).folder,falls(f).name)
    disp(pth)
    create_dff_fc3(pth, Fs)    
end