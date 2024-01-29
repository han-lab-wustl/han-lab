%plot every perius as a heat map

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
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
        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@transpose,roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',0));
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,2),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',1));
        subplot(2,3,ws)
        imagesc(xax,yax,test1)
%         xlim([-1 1])
        hold on
        vline(0)
        yticks(ytickax)
        yticklabels(1:ytickax)
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
        
    %%
    
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
        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@(x) transpose(nanmean(x,2)),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',0));
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,2),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',1));
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
        end
%         xlim([
        find_figure(['All PeriRewards notSOcombined'])
       test1 = [];
       for pl = 1:3
        test1 = cat(3,test1,cell2mat(cellfun(@(x) transpose(nanmean(x,2)),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',0)));
       end
       test1 = nanmean(test1,3);
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,2),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',1));
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
for d = 1:size(roi_dop_alldays_planes_perireward,1)
    xax = linspace(-5,5,size(roi_dop_alldays_planes_perireward{d,1},1));
    windowind = find(xax>=-0.5&xax<=1);
    for pl = 1:4

         mCSwindowcomMean{pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_perireward{d,pl}(windowind,:),2)'),1)*(xax(2)-xax(1))+xax(windowind(1));
    end
end

        for pl = 1:4
            find_figure(['All PeriRewards Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@(x) transpose(nanmean(x,2)),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',0));
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,2),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',1));
        subplot(3,3,ws)
        imagesc(xax,yax,test1)
        xlim([-1 1])
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
        test1 = cat(3,test1,cell2mat(cellfun(@(x) transpose(nanmean(x,2)),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',0)));
       end
       test1 = nanmean(test1,3);
        xax = linspace(-5,5,size(test1,2));
        yax = 1:size(test1,1);
        ytickax = cumsum(cellfun(@(x) size(x,2),roi_dop_alldays_planes_periUS(:,pl),'UniformOutput',1));
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
    
       %% peak post CS activity
      

  path='D:\workspaces_darkreward';%% WITH EARLIEST DAYS
    cd(path)
    workspace = {'156_dark_reward_AllDays_CutPlanes_workspace.mat','167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
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
        for pl = 1:4
            find_figure(['All PeriRewards Latency Plane ' num2str(pl)])
        test1 = cell2mat(cellfun(@transpose,roi_dop_alldays_planes_periCS(:,pl),'UniformOutput',0));
        
        if pl<4
          [~,test2] = max(test1(:,40:end),[],2);

        else
          [~,test2] = max(test1(:,40:end),[],2);
        end
         test2 = test2*(xax(2)-xax(1))-(xax(2)-xax(1));
          subplot(3,3,ws)
          plot(test2,'o','Color',planecolors{pl})
          xlabel('Trials')
          if pl<4
              ylabel('Activity trough Latency (s after CS)')
          else
          ylabel('Activity Peak Latency (s after CS)')
          end
          title(name)
        end
    end