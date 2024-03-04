clear all;
src = 'I:\vids_to_analyze\face_and_pupil';
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