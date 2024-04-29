clear all;
<<<<<<< HEAD
src2 = 'I:\vip_inhibition';
fls2 = dir(fullfile(src2, '*time*.mat'));

for i=1:length(fls2)    
    load(fullfile(fls2(i).folder,fls2(i).name))    
    save(fullfile(fls2(i).folder,fls2(i).name),'VR','-v7.3')
    disp(fls2(i).name)
%     disp(src)
%     disp(fls)
    clearvars -except src2 fls2 i
=======
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
>>>>>>> c4f7a267d2a6d9dc55e3bd4e196eb9797a585d54
end