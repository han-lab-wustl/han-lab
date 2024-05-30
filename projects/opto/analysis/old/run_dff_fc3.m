% ;
clear all; clear all;
dys = [33:35];
% dys = dir(fullfile('Y:\sstcre_analysis\fmats\e201\days', '*Fall.mat'));
Fs = 31.25; % frame rate
for i=1:length(dys)
    days = dir(fullfile('Y:\sstcre_imaging\e200\', sprintf('%i', dys(i)), '**\Fall.mat'));
%     fallpth = fullfile(days(i).folder,days(i).name);
    fallpth = fullfile(days.folder,days.name);
    fallpth = create_dff_fc3(fallpth, Fs)
end