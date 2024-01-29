 ROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
 dopvarnames = {'ALL_dopnorm_allday_failed_post_rew','ALL_dopnorm_allday_failed_pre_rew_far','ALL_dopnorm_allday_failed_pre_rew_near',...
     'ALL_dopnorm_allday_probe_in_rew',...
     'ALL_dopnorm_allday_probe_post_rew','ALL_dopnorm_allday_probe_pre_rew_far','ALL_dopnorm_allday_probe_pre_rew_near',...
     'ALL_dopnorm_allday_success_in_rew_pre_us','ALL_dopnorm_allday_success_in_rew_post_us','ALL_dopnorm_allday_success_post_rew',...
     'ALL_dopnorm_allday_success_pre_rew_far','ALL_dopnorm_allday_success_pre_rew_near'};
 roevarnames = {'ALL_Spd_allday_failed_post_rew','ALL_Spd_allday_failed_pre_rew_far','ALL_Spd_allday_failed_pre_rew_near',...
     'ALL_Spd_allday_probe_in_rew',...
     'ALL_Spd_allday_probe_post_rew','ALL_Spd_allday_probe_pre_rew_far','ALL_Spd_allday_probe_pre_rew_near',...
     'ALL_Spd_allday_success_in_rew_pre_us','ALL_Spd_allday_success_in_rew_post_us','ALL_Spd_allday_success_post_rew',...
     'ALL_Spd_allday_success_pre_rew_far','ALL_Spd_allday_success_pre_rew_near'};
 for dop = 1:length(dopvarnames)

dopvar = eval(dopvarnames{dop});
roevar = eval(roevarnames{dop});
% if dop <5
%     dopvar = cellfun(@transpose,dopvar,'UniformOutput',0);
%     roevar = cellfun(@transpose,roevar,'UniformOutput',0);
% end

ndays = 10;
for d = 1:ndays
    find_figure([dopvarnames{dop} ' part ' num2str(ceil(d/10))]);
    if ~isempty(dopvar{d,1})
for r = 1:4
    dopx = linspace(-5,5,size(dopvar{d,1}(:,:,r),1));
    roex = linspace(-5,5,size(roevar{d,1}(:,:,r),1));
    subplot(4,10,d-floor(d/10.1)*10+((4-r+1)*10-10))
%     xax = 
    plot(dopx,dopvar{d,1}(:,:,r))
    hold on
    plot(dopx,nanmean(dopvar{d,1}(:,:,r),2),'k-','LineWidth',1.5)
    ylim([min(min(dopvar{d,1}(:,:,r)))-(max(max(dopvar{d,1}(:,:,r)))-min(min(dopvar{d,1}(:,:,r)))) max(max(dopvar{d,1}(:,:,r)))]) 
    ylabel(ROI_labels{r})
    title(['Day ' num2str(d) ' n = ' num2str(size(dopvar{d,1}(:,:,r),2))])
    
             yyaxis right
             plot(roex,nanmean(roevar{d,1}(:,:,r),2),'k-','LineWidth',1.5)
             ylim([min(nanmean(roevar{d,1}(:,:,r),2)) max(nanmean(roevar{d,1}(:,:,r),2))+range(nanmean(roevar{d,1}(:,:,r),2))])
    
    
%     plot

end
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
    