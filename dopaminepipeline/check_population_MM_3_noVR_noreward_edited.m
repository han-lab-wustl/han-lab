% Moving now from day activity to population activity. for noVR_noreward
% check_population_MM_3_noVR_noreward.m. 
% 
% categories:
% all stops
% stops > 4 sec
% stops<2 sec
% 
% allmotions
% motion after stopping for > 4 sec
% motions after stopping for <2sec


close all
clear all
saving=1;
savepath='G:\dark_reward\dark_reward';
timeforpost = [0 1];
for allcat=1:6%1:11
    % path='G:\'
    % path='G:\analysed\HRZ\'
    %     path='G:\dark_reward\';
    %     path='G:\dark_reward\solenoid_unrew';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_old\';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_new';
    % path='G:\dark_reward\solenoid_HRZ\after_vac';
    path='E:\noVRnoRewWorkspace';
    %     path='G:\dark_reward\HRZ_allmouse'
    % path = 'D:\munneworkspaces\';
    % workspaces = {'E149_workspace','E156_workspace.mat','E157_workspace.mat','E158_workspace.mat'};
    % workspaces = {'E148_RR_D1-8.mat','E149_RR_D1-7.mat','E156_RR_d5-7_9_14_F.mat',...
    %     'E157_RR_d5_7-12_14_F.mat','E158_RR_d5-12_14_F.mat'};
    % workspaces = {'E148_workspace_D1-12.mat','E149_workspace_D1-13.mat','E156_RR_F.mat',...
    %     'E157_RR_F.mat','E158_RR_F.mat'};
    % workspaces = {'E156_HRZ.mat','E157_HRZ.mat','E158_HRZ.mat'};
    
    
    %     workspaces = {'148_dark_reward_workspace_02.mat','149_dark_reward_workspace.mat','156_dark_reward_workspace.mat',...
    %         '157_dark_reward_workspace.mat','158_dark_reward_workspace.mat','168_dark_reward_workspace.mat','171_dark_reward_workspace_01.mat',...
    %         '170_dark_reward_workspace_04.mat'};
    
    
    workspaces = {'223_noVRnoRew_workspace.mat','224_noVRnoRew_workspace.mat','225_noVRnoRew_workspace'};
    
    %      workspaces = {'156_HRZ_workspace.mat','157_HRZ_workspace.mat','167_HRZ_workspace.mat','168_HRZ_workspace.mat','169_HRZ_workspace.mat',...
    %         '171_HRZ_workspace' '170_HRZ_workspace.mat'};
    
    % workspaces = {'167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
    %        '171_dark_reward_workspace' '170_dark_reward_workspace.mat'};
    
    
    %     % workspaces = {'148_dark_reward_workspace_02.mat','149_dark_reward_workspace.mat','156_dark_reward_workspace.mat',...
    %     %     '157_dark_reward_workspace.mat','158_dark_reward_workspace.mat'}
    %     % workspaces = {'168_dark_reward_workspace.mat','171_dark_reward_workspace_01.mat','170_dark_reward_workspace_04.mat'};
    %
    
    %%%%%
    % workspaces={'168_dark_reward_workspace_05.mat', '171_dark_reward_workspace_11.mat', '170_dark_reward_workspace_01.mat' }
    %%%168
    % ROI_labels{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %171
    % ROI_labels{2} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %170
    % ROI_labels{3} ={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};;
    %%%%%%
    
    %     %148 labels %%MM
    %     %%old batch
    %     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SR_SP','Plane 3 SP', 'Plane 4 SO'};%%MM
    %     %149 dark Rewards labels
    %     ROI_labels{2} = {'Plane 1 SR','Plane 1 SP','Plane 2 SP','Plane 2 SP_SO','Plane 3 SO','Plane 4 SO'};
    %     %156 roi labels
    %     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
    
    %     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
    %     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
    %     %     %157 roi labels
    %     ROI_labels{2} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
    %     %158 roi
%     ROI_labels{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SO_SP','Plane 3 SP','Plane 4 SO'};
    %     % fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8]};
    %158 roi
     ROI_labels{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    ROI_labels{2} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
     ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %%%168
%     ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %     %171
%     ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %     %170
%     %       ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SO_SP','Plane 3 SP','Plane 4 SO'};
%     %        fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8],[1:10],[1:8],[1:11]};
%     %%%179
%     ROI_labels{6} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     
%     %%%181
%     ROI_labels{7} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %     %
%     
    
    %
    
    %     %%%167
    %     ROI_labels{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %168
    %     ROI_labels{2} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %
    %     %169
    %     ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %
    %     %      ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %171
    %     ROI_labels{4} =  {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %170
    %     ROI_labels{5} = {'Plane 1 SR','Plane 2 SR_SP','Plane 2 SP','Plane 3 SP','Plane 3 SP_SO', 'Plane 4 SO'};
    %      ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %      fileslist={[1:8],[1:8],[1:9],[1:8],[1:9]};
    %     fileslist={[1:10],[1:8],[1:10],[1:9],[1:8]};
    %    fileslist={[1:4],[1:4],[1:4],[1:4],[1:4]};
    
    %       fileslist={[1:4],[1:5],[1:3]};
    
    % new batch
    
    
    
    
    
    
    xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.02];
    allmouse_dop={}; allmouse_roe={};
    
    % fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
    
    
    
    mice = cellfun(@(x) x(1:4),workspaces,'UniformOutput',0);
    
    cats5={'roi_dop_allsuc_stop' 'roi_dop_allsuc_2s_stop' 'roi_dop_allsuc_4s_stop'  'roi_dop_allsuc_mov' 'roi_dop_allsuc_2s_mov' 'roi_dop_allsuc_4s_mov'}
    
    cats6={'roe_allsuc_stop' 'roe_allsuc_stop_2s' 'roe_allsuc_stop_4s' 'roe_allsuc_mov' 'roe_allsuc_mov_2s' 'roe_allsuc_mov_4s'}
    
    cats7={'roi_dop_alldays_planes_success_stop' 'roi_dop_alldays_planes_success_stop_reward' 'roi_dop_alldays_planes_success_no_reward'...
        'roi_dop_alldays_planes_success_mov'}
    
    
    dopvariablename = cats5{allcat};
    roevariablename = cats6{allcat};
    
    savepath = 'D:\munneworkspaces\HRZfigures\StartTriggered\summaryfigure\';
    saving = 0;
    
    setylimmanual2=[0.985 1.02]
    setylimmanual = [0.985 1.02];
    roerescale = [0.986 0.995];
    maxspeedlim = 25; %cm/s
    setxlimmanualsec = [-5 5];
    
    
    
    Pcolor = {[0 0 1],[0 1 0],[0 1 1],[1 1 0],[204 164 61]/256,[231 84 128]/256};
    
    % mousestyle =
    pstmouse = {};
    % cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    % cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    % cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    % cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    % cats5={ 'roi_dop_allsuc_perireward'};
    % cats6={'roi_roe_allsuc_perireward'};
    %    tdays=length(pr_dir0);
    %     earlydays = [1 2 3 4 ];
    %     latedays = [5 6 7 8];
    
    
    
    %
    %     earlydays = [1 2 ];
    %     latedays = [3 4];
    
    
    
    for currmouse = 1:length(workspaces)
        load([path '\' workspaces{currmouse}])
        %     if currmouse == 1
        %         close
        %     end
        tdays=length(pr_dir0);
        earlydays=[1:3];
        %         latedays=tdays-3:tdays;
        
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w )
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)
        %%% change for 313/79
        
        
        %         roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_success_mov(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        %         roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        %
        
        %          files=1:size(roi_dop_alldays_planes_perireward,1);
        
        files=1:size(roi_dop_alldays_planes_success_mov,1);
        
        %         if currmouse==1
        %             files=[1:9 11];
        % %             files = fileslist{currmouse};
        %         else
        %
        %             files=1:size(roi_dop_alldays_planes_perireward,1);
        %         end
        %         files=1:size(roi_dop_alldays_planes_perireward,1);
        find_figure(strcat('early and late days allmouse',cats5{allcat}))
        
        %     cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
        %     cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
        %     cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
        %     cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
        %     cats5={ 'roi_dop_allsuc_perireward'};
        %     cats6={'roi_roe_allsuc_perireward'};
        if strcmp(roevariablename(end-1:end),'_0')
            %             for checkforemptyfiles = 1:size(roe_alldays_planes_perireward_0,1)
            %                 if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
            %                     roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
            %                     roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
            %                     roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
            %                     roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
            %
            %                     dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
            %                     dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
            %                     dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
            %                     dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
            %                 end
            %             end
            %
            
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            
            
            
            
            
            roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
            
        else
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
        end
        %     color = Pcolor;
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s',cats5{allcat}))%%%20 days dop
            %             subplot(2,length(workspaces),currmouse)
            
            %              ax(jj)=subplot(3,length(workspaces),currmouse), hold on
            %               imagesc(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)))
            %
            %
            %             subplot(3,length(workspaces),currmouse+length(workspaces))
            %             imagesc(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)))
            %
            %
            subplot(2,length(workspaces),length(workspaces)+currmouse)
            ylims = ylim;
            
            
            %                         xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
            
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            [x,y]=find((squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:))==0))
            
            yax=nanmean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1);
            
            %%
            
            %%
            
            
            
            
            
            if jj == 1
                minyax = min(yax);
                maxyax = max(yax);
                %             patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
                %             patchy.EdgeAlpha = 0;
                %             patchy.FaceAlpha - 0.5;
                %             hold on
                %             plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
            else
                minyax = min(min(yax),minyax);
                maxyax = max(max(yax),maxyax);
            end
            se_yax=nanstd(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1))
            hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                text(xt(jj),yt(jj),ROI_labels{currmouse}{jj},'Color',color{jj})
            end
            title(workspaces{currmouse}(1:3))
            ylim(setylimmanual);
            %     yticks([])  ylim(setylimmanual);
            
            %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
            %     hp=plot(yax,'Color',color{jj})
            %     legend(hp,strcat('plane',num2str(jj)));hold on
            %     xlabel('Time(s)');
            xlim(setxlimmanualsec)
            %     set(gca,'ylim',[0.99 1.01])
            if jj == 1
                
                if size(roe_success_peristop,2)==79
                    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
                else
                    xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                end
                plot(xax,nanmean(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
            end
            ylims = ylim;
            if jj == size(dopvariable,2)
                ylims = ylim;
                pls = plot([0 0],ylims,'--k','Linewidth',1);
                ylim(ylims)
                pls.Color(4) = 0.5;
            end
            if currmouse == 1
                ylabel('Early Days')
            end
            
            %%%% compute mean
            lt_days=fliplr(length(files)-earlydays+1);
            pst=nanmean(squeeze(dopvariable(lt_days,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
            pre=nanmean(squeeze(dopvariable(lt_days,jj,40+ceil(timeforpost(1)-timeforpost(2)/5*40)-1:40+ceil(timeforpost(1)/5*40)-1)),2)
            mean_pre_pst=nanmean(pst);
            meanpstmouse{currmouse,jj} = mean_pre_pst;
            se_pre_pst=nanstd(pst)./sqrt(size(pst,1));
            
            pstmouse{currmouse,jj}(2,:) = pst; premouse{currmouse,jj}(2,:) = pre;
            roi_idx{currmouse,jj}=ROI_labels{currmouse}{jj};
            %             corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:3)
            pstmouse_allcat{allcat,2} = pstmouse;
            premouse_allcat{allcat,2} = premouse;
            
            allmouse_dop{2}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{2}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            %%%%%%%%%%%
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            %%%% compute area under the curve
            data=squeeze(dopvariable(lt_days,jj,40+ceil(0/5*40):40+ceil(4.5/5*40)));
            clear pst_ac
            for auc=1:size(data,1)
                pst_ac(auc,:)=trapz(1:size(data,2),data(auc,:));%%% 0.64-2.56
            end
            mean_ac_pre_pst=nanmean(pst_ac);
            meanacpstmouse{currmouse,jj} = mean_ac_pre_pst;
            se_ac_pre_pst=nanstd(pst_ac)./sqrt(size(pst_ac,1));
            mean_ac_pre_pst=nanmean(pst);
            ac_pstmouse{currmouse,jj} = pst_ac;
            
            
            
            
            
            
            %         title(workspaces{currmouse}(1:4))
            
            %         pst=nanmean(squeeze(data1(files,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
            %         mean_pre_pst(1,2)=nanmean(pst);
            %         meanpstmouse{currmouse,jj} = mean_pre_pst(1,2);
            %         se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
            %         pstmouse{currmouse,jj} = pst;
            %         corrcoef(pstmouse{currmouse,jj},pst)
            %             subplot(2,3,3+currmouse)
            %             imagesc(squeeze(data1(:,jj,:)))
            %             colormap(fake_parula)
            
            
            
            
        end
        currmouse
    end
    currmouse
    
    %%%%%%
    
    %     find_figure('mean_resp_per_mouse')
    %     for jj = 1:length(workspaces)
    %         for currmouse = 1:length(workspaces)
    %             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
    %             scatter(ones(size(pstmouse{currmouse,jj}))*jj*scaling+spacescale*(currmouse-1),...
    %                 pstmouse{currmouse,jj},20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.1)
    %             hold on
    %             scatter(jj*scaling+spacescale*(currmouse-1),meanpstmouse{currmouse,jj},100,'k','s','LineWidth',2)
    %             ylims = ylim;
    %             xtickl = [xtickl, workspaces{currmouse}(1:4)];
    %             if currmouse<length(workspaces)
    %                 [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(pstmouse{currmouse,jj},pstmouse{length(workspaces),jj});
    %
    %             end
    %         end
    %
    %     end
    %
    
    roi=unique([ROI_labels{:}]);
    
    %     for bb=1:4%%plane wise
    %         %     for fkfk=1:length(workspaces)
    %         ispresent = cellfun(@(s) ~isempty(strfind (s,num2str(bb))), roi_idx);
    %         [x y]=find(ispresent)
    %         for fk=1:size(x,1)
    %             mean_vals{bb,fk}=mean(pstmouse{x(fk),y(fk)});
    %             roi_sel{bb,fk}=roi_idx{x(fk),y(fk)};
    %             mouse_sel{bb,fk}=id_mouse{x(fk),y(fk)};
    %         end
    %     end
    
    clear comb_vals mean_comb_vals mean_vals comb_roi_vals ac_vals comb_ac_vals
    
    %%% for mean amplitude and
    for bb=1:4
        
        for fkfk=1:size(roi_idx,1)
            ispresent = cellfun(@(s) ~isempty(strfind(s,num2str(bb))),roi_idx(fkfk,:))
            [x y]=find(ispresent)
            vals=[];mean_vals=[];roi_vals=[];ac_vals=[];mean_ac_vals=[];
            for fk=1:size(x,2)
                vals=[vals (pstmouse{x(fk)+fkfk-1,y(fk)})];
                roi_vals{fk}=roi_idx{x(fk)+fkfk-1,y(fk)}(9:end);
                ac_vals=[ac_vals (ac_pstmouse{x(fk)+fkfk-1,y(fk)})];
                
            end
            comb_vals{bb,fkfk}= vals;
            mean_comb_vals{bb,fkfk}=mean(vals);
            comb_roi_vals{bb,fkfk}=roi_vals;
            comb_ac_vals{bb,fkfk}=ac_vals;
        end
    end
    
    
    %      subplot(2,length(workspaces),currmouse+length(workspaces))
    %     subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
    % % %     scaling = 2;
    % % %     spacescale = 0.2;
    % % %     xtickm = [];
    % % %     xtickl = {};
    % % %     ylims= [0.985 1.02];
    % % %     ptest = [];
    % % %     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    % % %     for jj = 1:4
    % % %         for currmouse = 1:length(workspaces)
    % % %             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
    % % %             %
    % % %             for roin=1: size(comb_vals{jj,currmouse},2)
    % % %                 if roin==1
    % % %                 scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % % %                     comb_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
    % % %                 else
    % % %                      scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % % %                     comb_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
    % % %                 end
    % % %
    % % %             end
    % % %             hold on
    % % %             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
    % % %
    % % %              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
    % % %             ylims = ylim;
    % % %             xtickl = [xtickl, workspaces{currmouse}(1:4)];
    % % %
    % % %             if currmouse<length(workspaces)
    % % %                 [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
    % % %
    % % %             end
    % % %             ylims= [0.985 1.025];
    % % %             set(gca,'ylim',ylims)
    % % %         end
    % % %
    % % %     end
    % % %
    % % %     realhs = h;
    % % %     realhs = reshape(realhs,size(ptest,1),size(ptest,2));
    % % %
    % % %     for jj = 1:size(realhs,2)
    % % %         for kk = 1:size(realhs,1)
    % % %             if realhs(kk,jj) == 1
    % % %                 plot([jj*scaling+spacescale*(kk-1) jj*scaling+spacescale*(length(workspaces)-1)],[1.006+0.0019-kk*0.0005 1.006+0.0019-kk*0.0005],'k-')
    % % %             end
    % % %         end
    % % %     end
    % % %
    % % %
    % % %     xlim([1.5 10])
    % % %     xticks(xtickm)
    % % %     xticklabels(xtickl)
    % % %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % %  find_figure(strcat('mean_response_Auc',cats5{allcat}));
    % % %    subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
    % % %     scaling = 2;
    % % %     spacescale = 0.2;
    % % %     xtickm = [];
    % % %     xtickl = {};
    % % % %     ylims= [0.985 1.02];
    % % %     ptest = [];
    % % %     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    % % %     for jj = 1:4
    % % %         for currmouse = 1:length(workspaces)
    % % %             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
    % % %             %
    % % %             for roin=1: size(comb_ac_vals{jj,currmouse},2)
    % % %                 if roin==1
    % % %                 scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % % %                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
    % % %                 else
    % % %                      scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % % %                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
    % % %                 end
    % % %
    % % %             end
    % % %             hold on
    % % % %             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
    % % %
    % % % %              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
    % % % %             ylims = ylim;
    % % %             xtickl = [xtickl, workspaces{currmouse}(1:4)];
    % % % %                     if currmouse<length(workspaces)
    % % % %                         [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
    % % % %
    % % % %                     end
    % % %             ylims= [0.985 1.025];
    % % % %             set(gca,'ylim',ylims)
    % % %         end
    % % %
    % % %     end
    % % %
    %%%%%%%%%%%%
    
    
    
    
    
    %%%%%%%%%%%%EARLY DAYS
    
    %     for currmouse = 1:length(workspaces)
    %         load([path '\' workspaces{currmouse}])
    %         %     if currmouse == 1
    %         %         close
    %         %     end
    % %         files = fileslist{currmouse};
    %
    %         %     find_figure(strcat('allmouse',cats5{allcat}))
    %         find_figure(strcat('early and late days allmouse',cats5{allcat}))
    %
    %         %     cats5={'roi_dop_allsuc_perirewardCS' 'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
    %         %         'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov'}
    %         %     cats6={'roi_roe_allsuc_perirewardCS' 'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
    %         %         'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov'  }
    % %         roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
    % %         roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
    %
    %         planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    %         roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
    %         [v, w] = unique( roiplaneidx, 'stable' );
    %         duplicate_indices = setdiff( 1:numel(roiplaneidx), w )
    %         color = planecolors(roiplaneidx);
    %         color(duplicate_indices)=cellfun(@(x) x/10 ,color(duplicate_indices) ,'UniformOutput' ,false)
    %
    %
    %         %     cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    %         %     cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    %         %     cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    %         %     cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    %         %     cats5={ 'roi_dop_allsuc_perireward'};
    %         %     cats6={'roi_roe_allsuc_perireward'};
    %         if strcmp(roevariablename(end-1:end),'_0')
    %             for checkforemptyfiles = 1:size(roe_alldays_planes_perireward_0,1)
    %                 if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
    %                     roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                     roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                     roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                     roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %
    %                     dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                     dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                     dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                     dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                 end
    %             end
    %
    %
    %             dopvariable = eval(dopvariablename);
    %             roevariable = eval(roevariablename);
    %
    %
    %             roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
    %
    %         else
    %             dopvariable = eval(dopvariablename);
    %             roevariable = eval(roevariablename);
    %         end
    %         %     color = Pcolor;
    %         for jj = 1:size(dopvariable,2)
    %             eval(sprintf('data1=%s',cats5{allcat}))%%%20 days dop
    %             subplot(2,length(workspaces),currmouse)
    %             ylim(setylimmanual);
    %             ylims = ylim;
    %             %         xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    %             xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
    %             yax=mean(squeeze(dopvariable(files(earlydays),jj,:)),1);
    %             if jj == 1
    %                 minyax = min(yax);
    %                 maxyax = max(yax);
    %                 %             patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
    %                 %             patchy.EdgeAlpha = 0;
    %                 %             patchy.FaceAlpha - 0.5;
    %                 %             hold on
    %                 %             plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
    %             else
    %                 minyax = min(min(yax),minyax);
    %                 maxyax = max(max(yax),maxyax);
    %             end
    %             se_yax=nanstd(squeeze(dopvariable(files(earlydays),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(earlydays),jj,:)),1))
    %             hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
    %             if sum(isnan(se_yax))~=length(se_yax)
    %                 if ~isempty (h10.patch)
    %                 h10.patch.FaceColor = color{jj};
    %                 end
    %                 h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
    %                 h10.edge(2).Color=color{jj};
    %             end
    %             ylim(setylimmanual);
    %             %     yticks([])  ylim(setylimmanual);
    %
    %             %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
    %             %     hp=plot(yax,'Color',color{jj})
    %             %     legend(hp,strcat('plane',num2str(jj)));hold on
    %             %     xlabel('Time(s)');
    %             xlim(setxlimmanualsec)
    %             %     set(gca,'ylim',[0.99 1.01])
    %             if jj == 1
    %                 if size(roe_success_peristop,2)==79
    %                     xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    %                 else
    %                     xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
    %                 end
    %                 plot(xax,nanmean(squeeze(roevariable(files(earlydays),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
    %             end
    %             ylims = ylim;
    %             if jj == size(dopvariable,2)
    %                 ylims = ylim;
    %                 pls = plot([0 0],ylims,'--k','Linewidth',1);
    %                 ylim(ylims)
    %                 pls.Color(4) = 0.5;
    %             end
    %             if currmouse == 1
    %                 ylabel('Early Days')
    %             end
    %
    %             %         title(workspaces{currmouse}(1:4))
    %
    %             %         pst=nanmean(squeeze(data1(files,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
    %             %         mean_pre_pst(1,2)=nanmean(pst);
    %             %         meanpstmouse{currmouse,jj} = mean_pre_pst(1,2);
    %             %         se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
    %             %         pstmouse{currmouse,jj} = pst;
    %             %         corrcoef(pstmouse{currmouse,jj},pst)
    %             %             subplot(2,3,3+currmouse)
    %             %             imagesc(squeeze(data1(:,jj,:)))
    %             %             colormap(fake_parula)
    %
    %
    %             %%%% compute mean
    %             %         early_days=fliplr(length(files)-earlydays+1);
    %             pst=nanmean(squeeze(dopvariable(earlydays,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
    %              pre=nanmean(squeeze(dopvariable(earlydays,jj,40+ceil(timeforpost(1)-timeforpost(2)/5*40)-1:40+ceil(timeforpost(1)/5*40)-1)),2)
    %
    %
    %
    %              mean_pre_pst=nanmean(pst);
    %             meanpstmouse{currmouse,jj} = mean_pre_pst;
    %             se_pre_pst=nanstd(pst)./sqrt(size(pst,1));
    %             pstmouse{currmouse,jj}(1,:) = pst; premouse{currmouse,jj}(1,:) = pre;
    %             roi_idx{currmouse,jj}=ROI_labels{currmouse}{jj};
    % %             corrcoef(pstmouse{currmouse,jj},pst)
    %             id_mouse{currmouse,jj}=workspaces{currmouse}(1:3)
    %              pstmouse_allcat{allcat,1} = pstmouse;
    %              premouse_allcat{allcat,1} = premouse;
    %
    %
    %                allmouse_dop{1}{currmouse,jj}=squeeze(dopvariable(files(earlydays),jj,:));
    %              allmouse_roe{1}{currmouse,jj}=squeeze(roevariable(files(earlydays),1,:));
    %
    %             %%%% compute area under the curve
    %             data=squeeze(dopvariable(earlydays,jj,40+ceil(0/5*40):40+ceil(4.5/5*40)));
    %             clear pst_ac
    %             for auc=1:size(data,1)
    %                 pst_ac(auc,:)=trapz(1:size(data,2),data(auc,:));%%% 0.64-2.56
    %             end
    %             mean_ac_pre_pst=nanmean(pst_ac);
    %             meanacpstmouse{currmouse,jj} = mean_ac_pre_pst;
    %             se_ac_pre_pst=nanstd(pst_ac)./sqrt(size(pst_ac,1));
    %             mean_ac_pre_pst=nanmean(pst);
    %             ac_pstmouse{currmouse,jj} = pst_ac;
    %
    %         end
    %     end
    
    %%% earlycombine all SO
    
    figHandles=gcf;
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    %             filename = 'allmousecombined_HRZ_allmouse_mearly_noVR_noreward_allsuc_4s_mov'
    filename = strcat('novr_noreward',cats5{allcat})
    filepath = 'C:\Users\workstation4\Desktop\03082023\dark_reward_figures\noVR_noreward'
    for i = 1:size(figHandles,1)
        fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%         exportgraphics(figHandles(i),fn,'ContentType','vector')
    end
    disp(['figure saved in: ' fn])
    
    
    find_figure(strcat('comballmouse',num2str(allcat)));clf
    
    %%%%%%
    for p=2%1:2
        SO_allmouse_dop=[cell2mat(allmouse_dop{1,p}(1,4)) ;cell2mat(allmouse_dop{1,p}(2:3,4))];%% mouse1-4
        SO_allmouse_mut=[NaN(2,length(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes))];
        
        SO_allmouse_dop_roe=[cell2mat(allmouse_roe{1,p}(1,4)) ;cell2mat(allmouse_roe{1,p}(2:3,4))];%% mouse1-6
        SO_allmouse_mut_roe=[NaN(2,length(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes))];
        
        
        find_figure(strcat('comballmouse',num2str(allcat)));subplot(2,2,(p-1)*2+1);
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
        yax=SO_allmouse_dop;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{4}; h10.mainLine.Color = color{4}; h10.edge(1).Color = color{4};
            h10.edge(2).Color=color{4};
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        if p==1
            ylabel('Late Days')
        else
            ylabel('Early Days')
        end
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
        yax=SO_allmouse_dop_roe;
        plot(xax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        
        
        %%%control
        subplot(2,2,((p-1)*2+2));
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
        yax=SO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{4}; h10.mainLine.Color = color{4}; h10.edge(1).Color = color{4};
            h10.edge(2).Color=color{4};
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5
        if p==1
            ylabel('Late Days')
        else
            ylabel('Early Days')
        end
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
        yax=SO_allmouse_mut_roe;
        plot(xax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        %%%%%%%%%%%%%%% combine all the rest except SO
        cc=allmouse_dop{1,p}(1:3,1:3);out1= cat(1,cc{:});
%         cc2=allmouse_dop{1,p}(1,1:4); out2=(cat(1,cc2{:}));
        withoutSO_allmouse_dop=[out1];%% mouse1-5
        
        subplot(2,2,(p-1)*2+1);
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
        yax= withoutSO_allmouse_dop;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{1}; h10.mainLine.Color = color{1}; h10.edge(1).Color = color{1};
            h10.edge(2).Color=color{1};
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        if p==1
            ylabel('Late Days')
        else
            ylabel('Early Days')
        end
        
        cc=allmouse_roe{1,p}(1:3,1:3);out1= cat(1,cc{:});
%         cc2=allmouse_roe{1,p}(1,1:4); out2=(cat(1,cc2{:}));
        withoutSO_allmouse_dop_roe=[out1];%% mouse1-6
        
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
        yax=withoutSO_allmouse_dop_roe;
        plot(xax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        title('GrabDA')
        
        %%% control
        
        %          cc=allmouse_dop{1,p}(2:6,1:3);out1= cat(1,cc{:});
        %         cc2=allmouse_dop{1,p}(1,1:5); out2=(cat(1,cc2{:}));
        
%         cc=allmouse_dop{1,p}(5:7,1:3);out1= cat(1,cc{:});
        
        withoutSO_allmouse_mut=[NaN(2,length(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes))];
        
        subplot(2,2,((p-1)*2+2));
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
        yax=withoutSO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax)
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{1}; h10.mainLine.Color = color{1}; h10.edge(1).Color = color{1};
            h10.edge(2).Color=color{1};
        end
        ylim(setylimmanual2);
        
        
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        if p==1
            ylabel('Late Days')
        else
            ylabel('Early Days')
        end
        title('GrabDA-mutant')
        
%         cc=allmouse_roe{1,p}(5:7,1:3);out1= cat(1,cc{:})
        
        withoutSO_allmouse_mut_roe=[NaN(2,length(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes))];
        
        xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
        yax=withoutSO_allmouse_mut_roe;
        plot(xax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        
        %         find_figure('stats');
        %         time_pts=[-2: 2];
        %         xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
        %
        %         resp_win=find(xax>=0 & xax<=1)
        %         [h,p]=ttest2(mean(SO_allmouse_dop(:,resp_win),2),mean(SO_allmouse_mut(:,resp_win),2))
        %          [h,p]=ttest2(mean(withoutSO_allmouse_dop(:,resp_win),2),mean(withoutSO_allmouse_mut(:,resp_win),2))
        
        
        %%% SO
        
        
        
        
        
        %%% ALLEXCEPT SO
        
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        %             filename = 'allmousecombined_HRZ_allmouse_mearly_noVR_noreward_allsuc_4s_mov'
        filename = strcat('comballmouse',cats5{allcat})
        filepath = 'C:\Users\workstation4\Desktop\03082023\dark_reward_figures\noVR_noreward'
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%             exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])
        
        
        
        
        
        
        
        
        
    end
    %%%%%%%%%%%%%%%%%
    
    xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes
    
    
    
    
    
    
    % %%%%%%%%%%%%%%
    % for currmouse = 1:length(workspaces)
    %     load([path '\' workspaces{currmouse}])
    %     if currmouse == 1
    %         close
    %     end
    %     files = fileslist{currmouse};
    %     find_figure('allmouse')
    %
    %     cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    %     cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    %     cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    %     cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    %     cats5={ 'dop_allsuc_perireward'};
    %     cats6={'roe_allsuc_perireward'};
    %     if strcmp(roevariablename(end-1:end),'_0')
    %         for checkforemptyfiles = 1:size(roe_alldays_planes_perireward_0,1)
    %             if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
    %                 roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                 roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                 roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %                 roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
    %
    %                 dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                 dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                 dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %                 dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
    %             end
    %         end
    %
    %
    %         dopvariable = eval(dopvariablename);
    %         roevariable = eval(roevariablename);
    %
    %
    %
    %
    %         roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
    %     else
    %         dopvariable = eval(dopvariablename);
    %         roevariable = eval(roevariablename);
    %     end
    %
    %
    %     color = Pcolor;
    %     for jj = 1:4
    %         eval(sprintf('data1=%s',cats5{1}))%%%20 days dop
    %         subplot(2,length(workspaces),currmouse)
    %         ylim(setylimmanual);
    %         ylims = ylim;
    %         xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    %         yax=mean(squeeze(dopvariable(files(earlydays),jj,:)),1);
    %         if jj == 1
    %             minyax = min(yax);
    %             maxyax = max(yax);
    %             %             patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
    %             %             patchy.EdgeAlpha = 0;
    %             %             patchy.FaceAlpha - 0.5;
    %             %             hold on
    %             %             plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
    %         else
    %             minyax = min(min(yax),minyax);
    %             maxyax = max(max(yax),maxyax);
    %         end
    %         se_yax=std(squeeze(dopvariable(files(earlydays),jj,:)),1)./sqrt(size(squeeze(roevariable(files(earlydays),jj,:)),1))
    %         hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
    %         h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
    %         h10.edge(2).Color=color{jj};
    %         ylim(setylimmanual);
    %         %     yticks([])  ylim(setylimmanual);
    %
    %         %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
    %         %     hp=plot(yax,'Color',color{jj})
    %         %     legend(hp,strcat('plane',num2str(jj)));hold on
    %         %     xlabel('Time(s)');
    %         xlim(setxlimmanualsec)
    %         %     set(gca,'ylim',[0.99 1.01])
    %         if jj == 1c
    %             plot(xax,mean(squeeze(roevariable(files(earlydays),jj,:)),1)/100*diff(roerescale)+roerescale(1),'k')
    %         end
    %         ylims = ylim;
    %         if jj == 4
    %             ylims = ylim;
    %             pls = plot([0 0],ylims,'--k','Linewidth',1);
    %             ylim(ylims)
    %             pls.Color(4) = 0.5;
    %         end
    %
    %         title(workspaces{currmouse}(1:4))
    %         if currmouse == 1
    %             ylabel('Early Days')
    %         end
    %
    %         pst=nanmean(squeeze(data1(files,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
    %         mean_pre_pst(1,2)=nanmean(pst);
    %         meanpstmouse{currmouse,jj} = mean_pre_pst(1,2);
    %         se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
    %         pstmouse{currmouse,jj} = pst;
    %         corrcoef(pstmouse{currmouse,jj},pst)
    %         %             subplot(2,3,3+currmouse)
    %         %             imagesc(squeeze(data1(:,jj,:)))
    %         %             colormap(fake_parula)
    %
    %     end
    % end
    
    
    
    % %  clear comb_vals mean_comb_vals mean_vals comb_roi_vals ac_vals comb_ac_vals
    % %
    % %  %%% for mean amplitude and
    % %     for bb=1:4
    % %
    % %         for fkfk=1:size(roi_idx,1)
    % %             ispresent = cellfun(@(s) ~isempty(strfind(s,num2str(bb))),roi_idx(fkfk,:))
    % %             [x y]=find(ispresent)
    % %             vals=[];mean_vals=[];roi_vals=[];ac_vals=[];mean_ac_vals=[];
    % %             for fk=1:size(x,2)
    % %                vals=[vals (pstmouse{x(fk)+fkfk-1,y(fk)})];
    % %                roi_vals{fk}=roi_idx{x(fk)+fkfk-1,y(fk)}(9:end);
    % %                 ac_vals=[ac_vals (ac_pstmouse{x(fk)+fkfk-1,y(fk)})];
    % %
    % %             end
    % %             comb_vals{bb,fkfk}= vals;
    % %             mean_comb_vals{bb,fkfk}=mean(vals);
    % %             comb_roi_vals{bb,fkfk}=roi_vals;
    % %             comb_ac_vals{bb,fkfk}=ac_vals;
    % %         end
    % %     end
    % %
    % %        find_figure(strcat('early days allmouse',cats5{allcat}))
    % %     subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
    % %
    % %     scaling = 2;
    % %     spacescale = 0.2;
    % %     xtickm = [];
    % %     xtickl = {};
    % %     ylims= [0.985 1.02];
    % %     ptest = [];
    % %     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    % %     for jj = 1:4
    % %         for currmouse = 1:length(workspaces)
    % %             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
    % %             %
    % %             for roin=1: size(comb_vals{jj,currmouse},2)
    % %                 if roin==1
    % %                 scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % %                     comb_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
    % %                 else
    % %                      scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % %                     comb_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
    % %                 end
    % %
    % %             end
    % %             hold on
    % %             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
    % %
    % %              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
    % %             ylims = ylim;
    % %             xtickl = [xtickl, workspaces{currmouse}(1:4)];
    % %                     if currmouse<length(workspaces)
    % %                         [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
    % %
    % %                     end
    % %             ylims= [0.985 1.025];
    % %             set(gca,'ylim',ylims)
    % %         end
    % %
    % %     end
    % %
    % %      realhs = h;
    % %     realhs = reshape(realhs,size(ptest,1),size(ptest,2));
    % %
    % %     for jj = 1:size(realhs,2)
    % %         for kk = 1:size(realhs,1)
    % %             if realhs(kk,jj) == 1
    % %                 plot([jj*scaling+spacescale*(kk-1) jj*scaling+spacescale*(length(workspaces)-1)],[1.006+0.0019-kk*0.0005 1.006+0.0019-kk*0.0005],'k-')
    % %             end
    % %         end
    % %     end
    % %
    % %
    % %     xlim([1.5 10])
    % % xticks(xtickm)
    % % xticklabels(xtickl)
    % %
    % %
    % %
    % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %  find_figure(strcat('mean_response_Auc',cats5{allcat}));
    % %     subplot(2,length(workspaces),1:length(workspaces)), cla()
    % %     scaling = 2;
    % %     spacescale = 0.2;
    % %     xtickm = [];
    % %     xtickl = {};
    % % %     ylims= [0.985 1.02];
    % %     ptest = [];
    % %     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
    % %     for jj = 1:4
    % %         for currmouse = 1:length(workspaces)
    % %             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
    % %             %
    % %             for roin=1: size(comb_ac_vals{jj,currmouse},2)
    % %                 if roin==1
    % %                 scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % %                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
    % %                 else
    % %                      scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
    % %                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
    % %                 end
    % %
    % %             end
    % %             hold on
    % % %             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
    % %
    % % %              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
    % % %             ylims = ylim;
    % %             xtickl = [xtickl, workspaces{currmouse}(1:4)];
    % % %                     if currmouse<length(workspaces)
    % % %                         [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
    % % %
    % % %                     end
    % %             ylims= [0.985 1.025];
    % % %             set(gca,'ylim',ylims)
    % %         end
    % %
    % %     end
    % %
    % % %      realhs = h;
    % % %     realhs = reshape(realhs,size(ptest,1),size(ptest,2));
    % % %
    % % %     for jj = 1:size(realhs,2)
    % % %         for kk = 1:size(realhs,1)
    % % %             if realhs(kk,jj) == 1
    % % %                 plot([jj*scaling+spacescale*(kk-1) jj*scaling+spacescale*(length(workspaces)-1)],[1.006+0.0019-kk*0.0005 1.006+0.0019-kk*0.0005],'k-')
    % % %             end
    % % %         end
    % % %     end
    % % %
    % %
    % %     xlim([1.5 10])
    % % xticks(xtickm)
    % % xticklabels(xtickl)
    
    
    %      if saving
    %         %     stripROIlabels = cellfun(@(x) )
    %         set(gcf,'units','normalized','outerposition',[0 0 1 1])
    %         saveas(gcf,[savepath '/' mouseId '_' variablelable{varsi} '_full_figure.svg'],'svg')
    %         hAx = findobj('type', 'axes');
    %         subtitles = {'_Late_Day','_Early_Day','AllDay_colorplot'};
    %         for iAx = 1:length(hAx)-1
    %             panel = mod(iAx,3);
    %             if panel == 0
    %                 panel = 3;
    %             end
    %             if (numROIs - ceil(iAx/3)+1)>0
    %                 planeidx = numROIs - ceil(iAx/3)+1;
    %
    %                 axtitle = [mouseId '_' ROI_labels{planeidx} subtitles{panel}];
    %             else
    %                 axtitle = [mouseId '_Speed' subtitles{panel} ];
    %             end
    %
    %             fNew = figure('units','normalized','outerposition',[0 0 1 1]);
    %             hNew = copyobj(hAx(iAx), fNew);
    %             set(hNew, 'pos', [0.23162 0.2233 0.72058 0.63107])
    %             set(gca,'fontsize', 18)
    %             set(gca,'FontName','Arial')
    %             %         InSet = get(hNew, 'TightInset');
    %             %         set(hNew, 'Position', [InSet(1:2)+0.05, 1-InSet(1)-InSet(3)-0.1, 1-InSet(2)-InSet(4)-0.05])
    %             if ~exist([savepath '/' variablelable{varsi}])
    %                 mkdir([savepath '/' variablelable{varsi}])
    %             end
    %             set(gca,'units','normalized','outerposition',[0.01 0.1 0.95 0.88])
    %             %         set(gcf, 'Renderer', 'Painters');
    %
    %             saveas(fNew,[savepath '/' variablelable{varsi} '/' axtitle '.svg'],'svg')
    %         end
    %         close all
    
    
    
end

% %%%signficance












%  % save all the current figures
% % %             figHandles = findall(0,'Type','figure')
%             figHandles=gcf;
% %             filename = 'allmousecombined_HRZ_allmouse_mearly_noVR_noreward_allsuc_4s_mov'
%             filename = 'allmousecombined_noVR_no_reward_4s_mov'
%             filepath = 'G:\dark_reward\noVR_noreward_allmouse_3'
%             for i = 1:size(figHandles,1)
%                 fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%                 exportgraphics(figHandles(i),fn,'ContentType','vector')
%             end
%             disp(['figure saved in: ' fn])


%
%




