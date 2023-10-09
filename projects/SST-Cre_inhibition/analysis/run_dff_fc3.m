days = dir('Y:\sstcre_analysis\fmats\e145\days\**\*_1_*.mat');
Fs = 31.25; % frame rate
for i=1:length(days)
    fallpth = fullfile(days(i).folder,days(i).name);
    fallpth = create_dff_fc3(fallpth, Fs)
end