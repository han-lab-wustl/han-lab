clear all;
src = '\\storage1.ris.wustl.edu\ebhan\Active\calvin\E231\240410_CF';
fls = dir(fullfile(src, '*params*.mat'));

for i=1:length(fls)
    try
        load(fullfile(fls(i).folder,fls(i).name))
        save(fullfile(fls(i).folder,fls(i).name))
    catch
        disp(fls(i).name)
    end
    clearvars --except src fls i
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