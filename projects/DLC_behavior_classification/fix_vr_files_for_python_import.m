clear all;
src = 'Y:\DLC\dlc_mixedmodel2';
fls = dir(fullfile(src, '*time*.mat'));

for i=1:length(fls)
    load(fullfile(fls(i).folder,fls(i).name))
    save(fullfile(fls(i).folder,fls(i).name),'VR','-v7.3')
    clear VR
end