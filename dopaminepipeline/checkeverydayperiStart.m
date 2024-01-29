 %% peri us with COM
    
    %plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','157_dark_reward_workspace','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace','221_dark_reward1_workspace'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
%     figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        name = workspace{ws}(1:3);
        load(workspace{ws})
%         means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         subplot(2,3,ws)
%         for p = 1:size(means,2)
%             errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
%             hold on
%         end
mCSwindowcomMean = cell(4,1);
for d = 1:size(roi_dop_alldays_planes_success_mov,1)
    xax = linspace(-5,5,size(roi_dop_alldays_planes_success_mov{d,1},2));
    windowind = find(xax>=-1&xax<=1);
    for pl = 1:4

         mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_success_mov{d,pl}(:,windowind),1)),1)*(xax(2)-xax(1))+xax(windowind(1));
    end
end
find_figure(['SO COM'])
subplot(3,3,ws)
plot(mCSwindowcomMean{4})
        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@(x) (nanmean(x,1)),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,1),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',1));
        subplot(3,3,ws)
        imagesc(xax,yax,test1)
        xlim([-3 3])
        hold on
        vline(0)
        plot(mCSwindowcomMean{pl},yax,'r.')
%         yticks(ytickax)
%         yticklabels(1:ytickax)
        yticks(yax)
        ylabel('Reward, Ticks Divide Days')
        xlabel('Seconds')
        colorbar()
        title(name)
        end
%         xlim([
        find_figure(['All PeriRewards notSOcombined'])
       test1 = [];
       for pl = 1:3
        test1 = cat(3,test1,cell2mat(cellfun(@(x) (nanmean(x,1)),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',0)));
       end
       test1 = nanmean(test1,3);
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,1),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',1));
        subplot(3,3,ws)
        imagesc(xax,yax,test1)
        xlim([-1 1])
        hold on
        vline(0)
%         yticks(ytickax)
%         yticklabels(1:ytickax)
        yticks(yax)
        ylabel('Reward, Ticks Divide Days')
        xlabel('Seconds')
        colorbar()
        title(name)
        mtit('All PeriReward')

    end
    for pl = 1:4
        find_figure(['All PeriRewards Plane ' num2str(pl)])
    mtit('All PeriReward')
    end
    
    %% Cross Correlation HeatMap with Speed
    
    %plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','157_dark_reward_workspace','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace','221_dark_reward1_workspace'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
%     figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        name = workspace{ws}(1:3);
        load(workspace{ws})
%         means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         subplot(2,3,ws)
%         for p = 1:size(means,2)
%             errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
%             hold on
%         end
% mCSwindowcomMean = cell(4,1);
% for d = 1:size(roi_dop_alldays_planes_success_mov,1)
%     xax = linspace(-5,5,size(roi_dop_alldays_planes_success_mov{d,1},2));
%     windowind = find(xax>=-1&xax<=1);
%     for pl = 1:4
% 
%          mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_success_mov{d,pl}(:,windowind),1)),1)*(xax(2)-xax(1))+xax(windowind(1));
%     end
% end

        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@(x) (nanmean(x,1)),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        test2 = cell2mat(cellfun(@(x) (nanmean(x,1)),roe_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        
        xax = linspace(-5,5,size(test1,2));
        speedxax = linspace(-5,5,size(test2,2));
        
        test3 = smoothdata(test2(:,1:4:end),'gaussian',2);
        test4 = [];
        for d = 1:size(test3,1)
        [test4(d,:),lagx] = xcorr(rescale(test1(d,1:end)),rescale(test3(d,:)),find(xax>-2,1),'coeff');
        end
        
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,1),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',1));
        subplot(3,3,ws)
        imagesc(lagx*(xax(2)-xax(1)),yax,test4)
%         colorbar(pink)
%         xlim([-3 3])
        hold on
        vline(0)
%         plot(mCSwindowcomMean{pl},yax,'r.')
%         yticks(ytickax)
%         yticklabels(1:ytickax)
        yticks(yax)
        ylabel('Reward, Ticks Divide Days')
        xlabel('Seconds')
        colorbar()
        title(name)
        end
%         xlim([
       

    end
    for pl = 1:4
        find_figure(['All PeriRewards Plane ' num2str(pl)])
    mtit('All PeriReward')
    end
    
     
    
    
     %% Cross Correlation HeatMap with Acc
    
    %plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','157_dark_reward_workspace','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace','221_dark_reward1_workspace'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
%     figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        name = workspace{ws}(1:3);
        load(workspace{ws})
%         means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         subplot(2,3,ws)
%         for p = 1:size(means,2)
%             errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
%             hold on
%         end
% mCSwindowcomMean = cell(4,1);
% for d = 1:size(roi_dop_alldays_planes_success_mov,1)
%     xax = linspace(-5,5,size(roi_dop_alldays_planes_success_mov{d,1},2));
%     windowind = find(xax>=-1&xax<=1);
%     for pl = 1:4
% 
%          mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_success_mov{d,pl}(:,windowind),1)),1)*(xax(2)-xax(1))+xax(windowind(1));
%     end
% end

        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@(x) (nanmean(x,1)),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        test2 = cell2mat(cellfun(@(x) (nanmean(x,1)),roe_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        
        xax = linspace(-5,5,size(test1,2));
        speedxax = linspace(-5,5,size(test2,2));
        
        test3 = smoothdata(diff(test2(:,1:4:end),[],2)/(xax(2)-xax(1)),'gaussian',2);
        test4 = [];
        for d = 1:size(test3,1)
        [test4(d,:),lagx] = xcorr(rescale(test1(d,1:end-1)),rescale(test3(d,:)),find(xax>-2,1),'coeff');
        end
        
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,1),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',1));
        subplot(3,3,ws)
        imagesc(lagx*(xax(2)-xax(1)),yax,test4)
        [maxcorrs,maxlags] = max(test4')
        hold on
        scatter(lagx(maxlags)*(xax(2)-xax(1)),yax,15,'r','filled')
%         colorbar(pink)
%         xlim([-3 3])
        hold on
        vline(0)
%         plot(mCSwindowcomMean{pl},yax,'r.')
%         yticks(ytickax)
%         yticklabels(1:ytickax)
        yticks(yax)
        ylabel('Reward, Ticks Divide Days')
        xlabel('Seconds')
        colorbar()
        title(name)
        end
%         xlim([
       

    end
    for pl = 1:4
        find_figure(['All PeriRewards Plane ' num2str(pl)])
    mtit('All PeriReward')
    end
    
    
       %% Every Day Line Plots per workspace
      
%load(INPUT wORKSPACE PATH HERE)

ROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
 dopvarnames = {'roi_dop_alldays_planes_success_mov','roi_dop_alldays_planes_peridoubleCS'};
 roevarnames = {'roe_alldays_planes_success_mov','roi_roe_alldays_planes_peridoubleCS'};
%  for dop = 1:length(dopvarnames)


% if dop <5
%     dopvar = cellfun(@transpose,dopvar,'UniformOutput',0);
%     roevar = cellfun(@transpose,roevar,'UniformOutput',0);
% end
    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
test = eval(dopvarnames{1});
ndays = size(test,1);
for d = 1:ndays
    find_figure([dopvarnames{1} ' part ' num2str(ceil(d/10))]);
for r = 1:4
    for dop = 1
        dopvar = eval(dopvarnames{dop});
roevar = eval(roevarnames{dop});
    dopx = linspace(-5,5,size(dopvar{d,r},1));
    roex = linspace(-5,5,size(roevar{d,r},1));
    if ~all(all(isnan(dopvar{d,r})))
    subplot(4,10,d-floor(d/10.1)*10+((4-r+1)*10-10))
%     xax = 
yyaxis left
 yax1 = nanmean(dopvar{d,r},1);
    seyax1 = nanstd(dopvar{d,r},[],1)/sqrt(size(dopvar{d,r},1));
    xax = linspace(-5,5,length(yax1));
    
%     clearvars h
hold on
yyaxis left
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r}/dop;
    h.patch.FaceColor = planecolors{r}/dop;
    h.edge(1).Color = planecolors{r}/dop;
    h.edge(2).Color = planecolors{r}/dop;
%     clearvars h
%     plot(dopx,dopvar{d,r})
    hold on
    l = vline(0);
    l.Color = [0 0 0];
%     plot(dopx,nanmean(dopvar{d,r},2),'k-','LineWidth',1.5)
    ylim([min(yax1-seyax1)-(max(yax1+seyax1)-min(yax1-seyax1)) max(yax1+seyax1)]) 
    ylabel(ROI_labels{r})
    title(['Day ' num2str(d) ' n = ' num2str(size(dopvar{d,r},2))])
    
           
    
%     plot
    end
    end
    for dop = 1
        roevar = eval(roevarnames{dop});
        if ~all(all(isnan(roevar{d,r})))
    dopx = linspace(-5,5,size(dopvar{d,r},1));
    roex = linspace(-5,5,size(roevar{d,r},2));

          yyaxis right
             if dop == 1
             plot(roex,nanmean(roevar{d,r},1),'-','Color',[0.5 0.5 0.5],'LineWidth',1.5)
             hold on
             accel = smoothdata(diff(smoothdata(nanmean(roevar{d,r},1),'gaussian',2))/(roex(2)-roex(1)),'gaussian',8);
             plot(roex(1:end-1),(accel - min(accel)-min(nanmean(roevar{d,r},1)))/5,'-','Color',[0.5 0 0.1],'LineWidth',1)
             else
                 plot(roex,nanmean(roevar{d,r},1),'k--','LineWidth',1.5)
             end
             ylim([min(nanmean(roevar{d,r},1)) max(nanmean(roevar{d,r},1))+range(nanmean(roevar{d,r},1))])
               yyaxis left
        end
    end
    xlim([-2 2])
end
end
%  end

%% Peak Peristart dop with peak acceleration
    
    %plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace2.mat','167_dark_reward_AllDays_CutPlanes_workspace2.mat','168Alldays_dark_reward_workspace_cut2.mat','169_dark_reward_AllDays_CutPlanes_workspace2.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace2'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
%     figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        name = workspace{ws}(1:3);
        load(workspace{ws})
%         means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         subplot(2,3,ws)
%         for p = 1:size(means,2)
%             errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
%             hold on
%         end
% mCSwindowcomMean = cell(4,1);
% for d = 1:size(roi_dop_alldays_planes_success_mov,1)
%     xax = linspace(-5,5,size(roi_dop_alldays_planes_success_mov{d,1},2));
%     windowind = find(xax>=-1&xax<=1);
%     for pl = 1:4
% 
%          mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_success_mov{d,pl}(:,windowind),1)),1)*(xax(2)-xax(1))+xax(windowind(1));
%     end
% end
   
        for pl = 1:4
             find_figure([name ' Plane ' num2str(pl) 'Dop vs Acc']);
%             find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = (cellfun(@(x) max(x,[],2),roi_dop_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        test2 = (cellfun(@(x) max(smoothdata(diff(smoothdata(x,2,'gaussian',20),[],2)/(10/size(x,2)),2,'gaussian',4),[],2),roe_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        
        for d = 1:length(test1)
            if isempty(test1{d})
                test1{d} = NaN(size(test2{d}));
            end
            subplot(ceil(sqrt(length(test1))),ceil(sqrt(length(test1))),d)
            scatter(test2{d},test1{d},15,'filled')
            xlabel('Acc (cm/s2)')
            ylabel('Dop (dFF)')
            r = corrcoef(test1{d},test2{d});
            title(['Day ' num2str(d) ' r:' num2str(r(1,2))])
        end
        
%         xax = linspace(-5,5,size(test1,2));
%         speedxax = linspace(-5,5,size(test2,2));
        
      
      
%         colorbar(pink)
%         xlim([-3 3])
       
%         plot(mCSwindowcomMean{pl},yax,'r.')
%         yticks(ytickax)
%         yticklabels(1:ytickax)
%         yticks(yax)
%         ylabel('Reward, Ticks Divide Days')
%         xlabel('Seconds')
%         colorbar()
%         title(name)
            
        end
%         xlim([
       

    end
%     for pl = 1:4
%         find_figure(['All PeriRewards Plane ' num2str(pl)])
%     mtit('All PeriReward')
%     end


%% Peak PeriCS dop with min acceleration
    
    %plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace2.mat','167_dark_reward_AllDays_CutPlanes_workspace2.mat','168Alldays_dark_reward_workspace_cut2.mat','169_dark_reward_AllDays_CutPlanes_workspace2.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace2'};% '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
%     figure;
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    
    for ws = 1:length(workspace)
        name = workspace{ws}(1:3);
        load(workspace{ws})
%         means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
%         subplot(2,3,ws)
%         for p = 1:size(means,2)
%             errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
%             hold on
%         end
% mCSwindowcomMean = cell(4,1);
% for d = 1:size(roi_dop_alldays_planes_success_mov,1)
%     xax = linspace(-5,5,size(roi_dop_alldays_planes_success_mov{d,1},2));
%     windowind = find(xax>=-1&xax<=1);
%     for pl = 1:4
% 
%          mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_success_mov{d,pl}(:,windowind),1)),1)*(xax(2)-xax(1))+xax(windowind(1));
%     end
% end
   
        for pl = 1:4
             find_figure([name ' Plane ' num2str(pl) 'Dop vs Acc']);
%             find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = (cellfun(@(x) max(x,[],2),roi_dop_alldays_planes_periCS(:,pl),'UniformOutput',0));
        test2 = (cellfun(@(x) max(smoothdata(diff(smoothdata(x,2,'gaussian',20),[],2)/(10/size(x,2)),2,'gaussian',4),[],2),roe_alldays_planes_success_mov(:,pl),'UniformOutput',0));
        
        for d = 1:length(test1)
            if isempty(test1{d})
                test1{d} = NaN(size(test2{d}));
            end
            subplot(ceil(sqrt(length(test1))),ceil(sqrt(length(test1))),d)
            scatter(test2{d},test1{d},15,'filled')
            xlabel('Acc (cm/s2)')
            ylabel('Dop (dFF)')
            r = corrcoef(test1{d},test2{d});
            title(['Day ' num2str(d) ' r:' num2str(r(1,2))])
        end
        
%         xax = linspace(-5,5,size(test1,2));
%         speedxax = linspace(-5,5,size(test2,2));
        
      
      
%         colorbar(pink)
%         xlim([-3 3])
       
%         plot(mCSwindowcomMean{pl},yax,'r.')
%         yticks(ytickax)
%         yticklabels(1:ytickax)
%         yticks(yax)
%         ylabel('Reward, Ticks Divide Days')
%         xlabel('Seconds')
%         colorbar()
%         title(name)
            
        end
%         xlim([
       

    end
%     for pl = 1:4
%         find_figure(['All PeriRewards Plane ' num2str(pl)])
%     mtit('All PeriReward')
%     end