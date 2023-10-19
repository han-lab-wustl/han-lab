% dys = [10:14,19,20];
clear all; clear all;
dys = dir(fullfile('Y:\sstcre_analysis\fmats\e201\days', '*Fall.mat'));
Fs = 31.25; % frame rate
for i=1:length(dys)
%     days = dir(fullfile('Y:\sstcre_imaging\e200\', sprintf('%i', i), '**\Fall.mat'));
    fallpth = fullfile(dys(i).folder,dys(i).name);
    fallpth = create_dff_fc3(fallpth, Fs)
end