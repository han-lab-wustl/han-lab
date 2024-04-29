clear all;
src2 = 'I:\vip_inhibition';
fls2 = dir(fullfile(src2, '*time*.mat'));

for i=1:length(fls2)    
    load(fullfile(fls2(i).folder,fls2(i).name))    
    save(fullfile(fls2(i).folder,fls2(i).name),'VR','-v7.3')
    disp(fls2(i).name)
%     disp(src)
%     disp(fls)
    clearvars -except src2 fls2 i
end