clear all;
src = 'D:\PupilTraining-Matt-2023-07-07\opto-vids\Trial 2';
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
%%
clear all;
src = 'Y:\analysis\fmats\218';
fls = dir(fullfile(src, '*.mat'));

for i=1:length(fls)
    try
        load(fullfile(fls(i).folder,fls(i).name))
        save(fullfile(fls(i).folder,fls(i).name),'VR','-v7.3')
    catch
        disp(fls(i).name)
    end
    clear 
end