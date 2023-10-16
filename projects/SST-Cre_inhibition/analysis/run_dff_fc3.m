dys = [10:14,19,20];
Fs = 31.25; % frame rate
for i=dys
    days = dir(fullfile('Y:\sstcre_imaging\e200\', sprintf('%i', i), '**\Fall.mat'));
    fallpth = fullfile(days(1).folder,days(1).name);
    fallpth = create_dff_fc3(fallpth, Fs)
end