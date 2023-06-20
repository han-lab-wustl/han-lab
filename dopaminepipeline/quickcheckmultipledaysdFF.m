% clear all
% close all
mouse_id=195;
% pr_dir=uipickfiles;
days_check=1:length(pr_dir);

 dir_s2p = struct2cell(dir([pr_dir{1} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
regions = {'SR','SP','SO'};
figure;
for allplanes=1:size(planefolders,2)
    for days=days_check
        dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        
        cd(pr_dir2)
        
        load('params.mat')
        dffplane{allplanes}(:,days) = smoothdata(params.roibasemean3{1},'gaussian',3);

    end
    subplot(size(planefolders,2),1,size(planefolders,2)-allplanes+1)
   stackedplot(dffplane{allplanes},'PlotScaling','normalize')
   title(regions{allplanes})
end
