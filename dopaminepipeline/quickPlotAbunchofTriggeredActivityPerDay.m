 ROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
 dopvarnames = {'roi_dop_alldays_planes_success_stop','roi_dop_alldays_planes_success_stop_no_reward','roi_dop_alldays_planes_success_stop_reward',...
     'roi_dop_alldays_planes_success_mov',...
     'roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_periCS'};
 roevarnames = {'roe_alldays_planes_success_stop','roe_alldays_planes_success_stop_no_reward','roe_alldays_planes_success_stop_reward',...
     'roe_alldays_planes_success_mov',...
     'roe_alldays_planes_perireward','roi_roe_alldays_planes_periCS'};
 for dop = 1:length(dopvarnames)

dopvar = eval(dopvarnames{dop});
roevar = eval(roevarnames{dop});
if dop <5
    dopvar = cellfun(@transpose,dopvar,'UniformOutput',0);
    roevar = cellfun(@transpose,roevar,'UniformOutput',0);
end

ndays = 26;
for d = 1:ndays
    find_figure([dopvarnames{dop} ' part ' num2str(ceil(d/10))]);
for r = 1:4
    dopx = linspace(-5,5,size(dopvar{d,r},1));
    roex = linspace(-5,5,size(roevar{d,r},1));
    subplot(4,10,d-floor(d/10.1)*10+((4-r+1)*10-10))
%     xax = 
    plot(dopx,dopvar{d,r})
    hold on
    plot(dopx,nanmean(dopvar{d,r},2),'k-','LineWidth',1.5)
    ylim([min(min(dopvar{d,r}))-(max(max(dopvar{d,r}))-min(min(dopvar{d,r}))) max(max(dopvar{d,r}))]) 
    ylabel(ROI_labels{r})
    title(['Day ' num2str(d) ' n = ' num2str(size(dopvar{d,r},2))])
    
             yyaxis right
             plot(roex,nanmean(roevar{d,r},2),'k-','LineWidth',1.5)
             ylim([min(nanmean(roevar{d,r},2)) max(nanmean(roevar{d,r},2))+range(nanmean(roevar{d,r},2))])
    
    
%     plot

end
end
 end
%%

 ROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
figure;
for d = 1:6
for r = 1:4
    subplot(4,6,d+((4-r+1)*6-6))
%     xax = 
    plot(roi_dop_alldays_planes_success_stop{d,r}')
    hold on
    plot(nanmean(roi_dop_alldays_planes_success_stop{d,r}',2),'k-','LineWidth',1.5)
    ylabel(ROI_labels{r})
    title(['Day ' num2str(d)])
    
%     plot

end
end
    