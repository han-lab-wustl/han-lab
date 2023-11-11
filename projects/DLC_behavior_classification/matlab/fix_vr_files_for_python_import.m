clear all;
src = 'Y:\DLC\VR_data\dlc\convert';
fls = dir(fullfile(src, '*time*.mat'));

for i=1:length(fls)
    try
        load(fullfile(fls(i).folder,fls(i).name))
        save(fullfile(fls(i).folder,fls(i).name),'VR','-v7.3')
    catch
        disp(fls(i).name)
    end
    clear VR
end