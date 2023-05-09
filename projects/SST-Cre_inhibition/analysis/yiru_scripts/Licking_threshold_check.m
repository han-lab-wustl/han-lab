Settings.paths = dir("Y:\E186\E186\D68\Fall.mat");
Settings.level_mouse_name = 3;
Settings.level_day = 4;
for file =1:size(Settings.paths,1)
    file = fullfile(Settings.paths(file).folder,Settings.paths(file).name);
    l = load(file);
    directory = file;
    info = split(directory,'\');
    mouse_cd = string(info{Settings.level_mouse_name});
    day_cd = string(info{Settings.level_day});
    figure;plot(l.VR.lickVoltage);hold on;plot([0 length(l.VR.lickVoltage)],[l.VR.lickThreshold l.VR.lickThreshold]);plot(l.VR.lick,'Color','r')
    title(day_cd)
end

%% Lick threshold check for single VR data
%figure;plot(VR.lickVoltage);hold on;plot([0 length(VR.lickVoltage)],[VR.lickThreshold VR.lickThreshold]);plot(VR.lick,'Color','r')