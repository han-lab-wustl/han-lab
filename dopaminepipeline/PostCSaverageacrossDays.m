 path='F:\workspaces_darkreward';%% WITH EARLIEST DAYS

    workspaces = {'156_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
        '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    

        
    %     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
    ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
    %     %157 roi labels
%     ROI_labels{2} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
    %     %158 roi
    ROI_labels{2} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     % fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8]};
    %158 roi
    ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    
    %     %%%168
    ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %171
    ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %170
    ROI_labels{6} = {'Plane 1 SR','Plane 2 SR_SP','Plane 2 SP','Plane 3 SP','Plane 3 SP_SO', 'Plane 4 SO'};
    %        fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8],[1:10],[1:8],[1:11]};
    %%%179
    ROI_labels{7} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    
    %%%181
    ROI_labels{8}{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    ROI_labels{8}{2} = {'Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %
    postavg = {};
    SOpostavg = {};
    notSOpostavg = {};
    figure;
       for currmouse = 1:length(workspaces)
        if ~iscell(workspaces{currmouse})
        load([path '\' workspaces{currmouse}])
        currROI_labels = ROI_labels{currmouse};
        currtitle = workspaces{currmouse}(1:3);
        
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
        psttime = find(xax>0&xax<=1.5);
        postavg{currmouse} = squeeze(nanmean(roi_dop_allsuc_perirewardCS(:,:,psttime),3));
        if currmouse == 1
            postavg{currmouse}(10,:) = [];
        end
        else
            load([path '\' workspaces{currmouse}{1}])
            currROI_labels = ROI_labels{currmouse}{1};
            currtitle = workspaces{currmouse}{1}(1:3);
            
            xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            psttime = find(xax>0&xax<=1.5);
        postavg{currmouse} = squeeze(nanmean(roi_dop_allsuc_perirewardCS(:,:,psttime),3));
        
        
        load([path '\' workspaces{currmouse}{2}])
            currtitle = workspaces{currmouse}{2}(1:3);
            
            xax = frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            psttime = find(xax>0&xax<=1.5);
        temp = squeeze(nanmean(roi_dop_allsuc_perirewardCS(:,:,psttime),3));
        
        postavg{currmouse} = padcatcell2mat([postavg(currmouse);{temp}],2,'left');
        
        end
        SOpostavg{currmouse} = postavg{currmouse}(:,end);
        notSOpostavg{currmouse} = postavg{currmouse}(:,1:end-1);
         planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),currROI_labels,'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
        subplot(1,length(workspaces),currmouse)
        pls = plot(postavg{currmouse});
        for jj = 1:length(pls)
        pls(jj).Color = color{jj};
        end
        if currmouse == 1
            ylabel('Average post dFF')
        end
        xlabel('Days')
        title(currtitle)
        axis square
       end
       
       %% combine mice
       SOcombined = padcatcell2mat(SOpostavg,1,'left');
       notSOcombined = padcatcell2mat(cellfun(@(x) nanmean(x,2),notSOpostavg,'UniformOutput',0),1,'left');
       
       figure; errorbar(nanmean(SOcombined(:,1:5),2),nanstd(SOcombined(:,1:5),[],2)/sqrt(5),'r-','Capsize',0,'LineWidth',1.5)
       hold on
       errorbar(nanmean(notSOcombined(:,1:5),2),nanstd(notSOcombined(:,1:5),[],2)/sqrt(5),'-','Color',[0.5 0.5 0.5],'Capsize',0,'LineWidth',1.5)
       title('GrabDa')
           ylim([0.99 1.03])
       yticks(0.99:.01:1.03)
       ylims = ylim;
       text(1:size(SOcombined,1),ones(1,size(SOcombined,1))*ylims(1)+0.001,num2str(sum(~isnan(SOcombined(:,1:5)),2)),'HorizontalAlignment','center')
       xlim([0 19])
       xticks(1:18)
   
       xticklabels(-17:0)
       
       figure; errorbar(nanmean(SOcombined(:,6:end),2),nanstd(SOcombined(:,6:end),[],2)/sqrt(3),'r-','Capsize',0,'LineWidth',1.5)
       hold on
       errorbar(nanmean(notSOcombined(:,6:end),2),nanstd(notSOcombined(:,6:end),[],2)/sqrt(3),'-','Color',[0.5 0.5 0.5],'Capsize',0,'LineWidth',1.5)
       title('GrabDa-mut')
       ylim([0.99 1.03])
       yticks(0.99:.01:1.03)
       ylims = ylim;
       text(1:size(SOcombined,1),ones(1,size(SOcombined,1))*ylims(1)+0.001,num2str(sum(~isnan(SOcombined(:,6:end)),2)),'HorizontalAlignment','center')
       xlim([0 19])
       xticks(1:18)
       
       xticklabels(-17:0)