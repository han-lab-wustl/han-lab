 roerescale = [0.94 0.95];
    maxspeedlim = 25; %cm/s
 %
    
   realrois = [1 3 5];  %1:size(roi_dop_allsuc_mov,1) NOTE! I have several ROIS i ran to try to grab the stim with so I'm skipping those. If you don't have these just make it 1:size(roi_dop_allsuc_mov,1)
%% PeriStim    

for days = 1:size(roi_dop_alldays_planes_peristim,1)
    xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
    figure;
    for r = 1:length(realrois)
    subplot(2,2,r)
    plot(xax,roi_dop_alldays_planes_peristim{days,realrois(r)})
    hold on
    plot(xax,nanmean(roi_dop_alldays_planes_peristim{days,realrois(r)},2),'k','LineWidth',2)
    
      speedxax = frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
      plot(speedxax,roi_roe_alldays_planes_peristim{days,r}/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.5 0.5 0.5])
    plot(speedxax,nanmean(roi_roe_alldays_planes_peristim{days,r},2)/maxspeedlim*diff(roerescale)+roerescale(1),'k','LineWidth',2)
    
    plot(speedxax,roi_lick_alldays_planes_peristim{days,r}/100+0.93)
    end
    mtit(['Peri Stim Day ' num2str(days)])
%     plot(xax,nanmean(roe{days,realrois(r)},2)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
end

%% PeriStim  Stacked Plot
colors = distinguishable_colors(3);
for days = 1:size(roi_dop_alldays_planes_peristim,1)
    xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
    figure;
    for r = 1:length(realrois)
        for trial = 1:size(roi_dop_alldays_planes_peristim{days,realrois(r)},2)
    sb(trial*length(realrois)-length(realrois)+r) = subtightplot(size(roi_dop_alldays_planes_peristim{days,realrois(r)},2),length(realrois),trial*length(realrois)-length(realrois)+r);
    plot(xax,roi_dop_alldays_planes_peristim{days,realrois(r)}(:,trial),'Color',colors(rem(trial,3)+1,:))
    hold on
%     plot(xax,nanmean(roi_dop_alldays_planes_peristim{days,realrois(r)},2),'k','LineWidth',2)
    
      speedxax = frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
      plot(speedxax,roi_roe_alldays_planes_peristim{days,r}(:,trial)/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.5 0.5 0.5])
%     plot(speedxax,nanmean(roi_roe_alldays_planes_peristim{days,r},2)/maxspeedlim*diff(roerescale)+roerescale(1),'k','LineWidth',2)
    
    plot(speedxax,roi_lick_alldays_planes_peristim{days,r}(:,trial)/100+0.93,'k-')
    axis tight
    ylims = ylim;
    plot([0 0],ylims,'k--')
    if r>1
        yticks([])
    end
    end
    
%     plot(xax,nanmean(roe{days,realrois(r)},2)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
    end
    mtit(['Peri Stim Day ' num2str(days)])
    linkaxes(sb,'xy')
    clearvars sb
end

%% Stops

for days = 1:size(roi_dop_alldays_planes_success_stop,1)
    xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
    figure;
    for r = 1:length(realrois)
    subplot(2,2,r)
    plot(xax,roi_dop_alldays_planes_success_stop{days,realrois(r)})
    hold on
    plot(xax,nanmean(roi_dop_alldays_planes_success_stop{days,realrois(r)},1),'k','LineWidth',2)
    
    speedxax = frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
 plot(speedxax,roe_alldays_planes_success_stop{days,r}/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.5 0.5 0.5])
    plot(speedxax,nanmean(roe_alldays_planes_success_stop{days,r},1)/maxspeedlim*diff(roerescale)+roerescale(1),'k','LineWidth',2)
    end
    mtit(['Peri Stop Day ' num2str(days)])
end

%% Licks

for days = 1:size(roi_dop_alldays_planes_perinrlicks,1)
    xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
    figure;
    for r = 1:length(realrois)
    subplot(2,2,r)
    plot(xax,roi_dop_alldays_planes_perinrlicks{days,realrois(r)})
    hold on
    plot(xax,nanmean(roi_dop_alldays_planes_perinrlicks{days,realrois(r)},2),'k','LineWidth',2)
    
    speedxax = frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
     plot(speedxax,roi_roe_alldays_planes_perinrlicks{days,r}/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.5 0.5 0.5])
    plot(speedxax,nanmean(roi_roe_alldays_planes_perinrlicks{days,r},2)/maxspeedlim*diff(roerescale)+roerescale(1),'k','LineWidth',2)

    end
    mtit(['Peri Lick Day ' num2str(days)])
end

%% average speed
figure;
 for r = 1:length(realrois)
     subplot(2,2,r)
     plot(meanspeedtemp,percentsuccess(:,r),'b.','MarkerSize',25)
     title(['Plane ' num2str(r)])
     xlabel('Speed (cm)')
     ylabel('Percent Response per Day')
 end
 
 %%  time spent licking
 
figure;
 for r = 1:length(realrois)
     subplot(2,2,r)
     plot(timelicking,percentsuccess(:,r),'b.','MarkerSize',25)
     title(['Plane ' num2str(r)])
     xlabel('Percent Time Licking')
     ylabel('Percent Response per Day')
 end
 
 %% number of stops
 
 numstops = cellfun(@(x) size(x,1),roi_dop_alldays_planes_success_stop(:,1),'UniformOutput',1);
 figure;
 for r = 1:length(realrois)
     subplot(2,2,r)
     plot(numstops,percentsuccess(:,r),'b.','MarkerSize',25)
     title(['Plane ' num2str(r)])
     xlabel('Number of Stops')
     ylabel('Percent Response per Day')
 end