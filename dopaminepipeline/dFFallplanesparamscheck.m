clear all
close all

day_dir = uipickfiles('Prompt','Please Select all the folders for the previous popup');
%%
figure;
ROI_labels = {'SLM','SR','SP','SO'};
for d = 1:length(day_dir)
    dir_s2p = struct2cell(dir([day_dir{d} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));

    Fs = 31.25/size(planefolders,2);


for allplanes=1:size(planefolders,2)

        dir_s2p = struct2cell(dir([day_dir{d} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        load([pr_dir2 'params.mat'])
        subtightplot(size(planefolders,2)+1,length(day_dir),d+(length(day_dir)-allplanes+1)*length(day_dir)-length(day_dir))
        plot(timedFF,smoothdata(params.roibasemean3{1},'gaussian',10)/nanmean(params.roibasemean3{1}))
        hold on
        if d == 1
        ylabel(ROI_labels{allplanes})
        end
        
        xlim([5 utimedFF(end)])
        if allplanes == 4
            title(['Day ' num2str(d)])
        end
        if allplanes == 1
            
            gauss_win = 10;
        speed_smth_1=smoothdata(forwardvelALL,'gaussian',gauss_win)';
        subtightplot(size(planefolders,2)+1,length(day_dir),d+(length(day_dir)+1)*length(day_dir)-length(day_dir))
            plot(utimedFF,rescale(speed_smth_1),'k-')
             xlim([5 utimedFF(end)])
             if d == 1
                 ylabel('Speed')
             end
        end
        
end
end