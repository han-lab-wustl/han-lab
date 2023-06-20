% clear all
% close all
mouse_id=195;
% pr_dir=uipickfiles;
days_check=1:length(pr_dir);
meanspeedtemp = NaN(1,length(days_check));
timelicking = NaN(1,length(days_check));
for days=days_check
    dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
figure;
for allplanes=1:size(planefolders,2)
    
        dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        
        cd(pr_dir2)
        
        load('params.mat')
        sb(allplanes) = subplot(size(planefolders,2)+1,1,size(planefolders,2)-allplanes+1);
%         timedFF = 1:length(params.roibasemean3{1});
        if length(timedFF)>length(params.roibasemean3{1})
            timedFF = timedFF(1:end-1);
        end
        plot(timedFF,params.roibasemean3{1})
        hold on
        ylims = ylim;
        plot(utimedFF(1:end-1),rescale(stims,ylims(1),ylims(2)))
       
end
    title(['Day ' num2str(days)])
sb(allplanes+1)= subplot(size(planefolders,2)+1,1,allplanes+1);
temp = smoothdata(forwardvelALL,'gaussian',10);
meanspeedtemp(days) = nanmean(temp);
timelicking(days) = length(find(licksALL))/length(licksALL);
tempt = 1:length(forwardvelALL);
plot(utimedFF,temp)
hold on; plot(utimedFF(find(licksALL)),temp(find(licksALL)),'r.')
ylims = ylim;
        plot(utimedFF(1:end-1),rescale(stims,ylims(1),ylims(2)))
linkaxes(sb,'x')
end