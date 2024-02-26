% zahra's fix to import mat files as h5py in python
clear all;
src = 'I:\vids_to_analyze\tail'; % path to vr files
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