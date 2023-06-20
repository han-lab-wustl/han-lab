pr_dir = uipickfiles;

figure;
for days = 1:length(pr_dir)
    subplot(1,length(pr_dir),days)
    for allplanes = 1:4
     pr_dir1=strcat(pr_dir{days},'\')
        %%%s2p directory
        pr_dir2=strcat(pr_dir1,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
        load([pr_dir2 'params.mat'])
        if allplanes == 1
        plot(rescale(forwardvel,-1,0),'k-')
        end
        hold on
        plot(rescale(params.base_mean,allplanes-1,allplanes))
    end
end