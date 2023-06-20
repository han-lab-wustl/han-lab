 path='F:\workspaces_darkreward';%% WITH EARLIEST DAYS

    workspaces = {'156_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
        '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    
    
    for currmouse = 1:length(workspaces)
        if ~iscell(workspaces{currmouse})
        load([path '\' workspaces{currmouse}])
        
        for days = 1:size(roi_dop_alldays_planes_peridoubleCS,1)
            for roiss = 1:size(roi_dop_alldays_planes_peridoubleCS,2)
                if ~isnan(nanmean(nanmean(roi_dop_alldays_planes_perireward_double{days,roiss})))
                roi_dop_alldays_planes_peridoubleCS{days,roiss} = roi_dop_alldays_planes_peridoubleCS{days,roiss}(:,1:size(roi_dop_alldays_planes_perireward_double{days,roiss},2));
                if ~isempty(nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2))
                roi_dop_allsuc_perireward_doubleCS(days,roiss,:) = nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2);
                end
                
                roi_roe_alldays_planes_peridoubleCS{days,roiss} = roi_roe_alldays_planes_peridoubleCS{days,roiss}(:,1:size(roi_dop_alldays_planes_perireward_double{days,roiss},2));
                if ~isempty(nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2))
                roi_roe_allsuc_perireward_doubleCS(days,roiss,:) = nanmean(roi_roe_alldays_planes_peridoubleCS{days,roiss},2);
                end
                end
           end
        end
        
        save(workspaces{currmouse},'roi_dop_alldays_planes_peridoubleCS','roi_dop_allsuc_perireward_doubleCS','roi_roe_alldays_planes_peridoubleCS','roi_roe_allsuc_perireward_doubleCS','-append')
        else
            for cc = 1:length(workspaces{currmouse})
            load([path '\' workspaces{currmouse}{cc}])
        
        for days = 1:size(roi_dop_alldays_planes_peridoubleCS,1)
            for roiss = 1:size(roi_dop_alldays_planes_peridoubleCS,2)
                 if ~isnan(nanmean(nanmean(roi_dop_alldays_planes_perireward_double{days,roiss})))
                roi_dop_alldays_planes_peridoubleCS{days,roiss} = roi_dop_alldays_planes_peridoubleCS{days,roiss}(:,1:size(roi_dop_alldays_planes_perireward_double{days,roiss},2));
                if ~isempty(nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2))
                roi_dop_allsuc_perireward_doubleCS(days,roiss,:) = nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2);
                end
                roi_roe_alldays_planes_peridoubleCS{days,roiss} = roi_roe_alldays_planes_peridoubleCS{days,roiss}(:,1:size(roi_dop_alldays_planes_perireward_double{days,roiss},2));
                if ~isempty(nanmean(roi_dop_alldays_planes_peridoubleCS{days,roiss},2))
                roi_roe_allsuc_perireward_doubleCS(days,roiss,:) = nanmean(roi_roe_alldays_planes_peridoubleCS{days,roiss},2);
                end
                end
            end
        end
        
        save(workspaces{currmouse}{cc},'roi_dop_alldays_planes_peridoubleCS','roi_dop_allsuc_perireward_doubleCS','roi_roe_alldays_planes_peridoubleCS','roi_roe_allsuc_perireward_doubleCS','-append')
            end
        end
    end