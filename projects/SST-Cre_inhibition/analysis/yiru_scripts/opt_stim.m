clear all
close all
% run file to get opto frames, calculate behavioral vars and plot
% was like 5 diff scripts that used the same vars over and over

Settings.hrz_days = [55:62]; % get these days; ZD added for her folder structure
Settings.fmatpaths=dir('Z:\sstcre_imaging\e201\**\Fall.mat');
for i=1:length(Settings.fmatpaths)
    [parent,nm,~] = fileparts(fileparts(fileparts(fileparts(Settings.fmatpaths(i).folder))));
    dy{i} = str2num(nm);
end
dys = cell2mat(dy); % some days may have beahvior but not imaging and vice versa, tthus have to do this step for both files
% MESSY
Settings.hrz_days = [55:62]; % get days after this day; ZD added for her folder structure
Settings.fmatpaths = Settings.fmatpaths(ismember(dys,Settings.hrz_days)); % only certain days
Settings.vrpaths=dir('Z:\sstcre_imaging\e201\**\behavior\vr\*.mat');
for i=1:length(Settings.vrpaths)
    [parent,nm,~] = fileparts(fileparts(fileparts(Settings.vrpaths(i).folder)));
    dy{i} = str2num(nm);
end
dys = cell2mat(dy);
Settings.vrpaths = Settings.vrpaths(ismember(dys,Settings.hrz_days)); % only certain days

Settings.mouse_name = 'E201';
data = struct(); % if want to create a new dataset, comment line 6 and lines 14-16 (if ismember(day_cd ......)
% load('Y:\E186\E186\alldays_info.mat'); % contains a struct called data
rewrange = 10 *2/3;
optodays = [1 2 4 5 7 8]; % only run get opto frames on these days?
% assumes VR align is run already?
for fp=1:length(Settings.vrpaths)
    vrfl = fullfile(Settings.vrpaths(fp).folder,Settings.vrpaths(fp).name);
    fmatfl = fullfile(Settings.fmatpaths(fp).folder,Settings.fmatpaths(fp).name);
    if ismember(fp, optodays)
        vrfl = get_opto_frames(vrfl,fmatfl);
    end
    data = get_epoch_info(vrfl,fmatfl, data, fp, rewrange,Settings.mouse_name);      
end

save("Z:\hrz_info.mat","data")
data = load("Z:\hrz_info.mat");
plot_mean_speed(data)
plot_failures(data)














