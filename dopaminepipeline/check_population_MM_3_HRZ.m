
% MM_DA

% Input: workspaces folder
% Output:Figures
% single rew CS: First lick after reward
%  	single rew CS
% 	doubles: First lick after reward
% stopping success trials: all stops
% non rewarded stops
%    rewarded stops
% moving success trials: all motions
% moving rewarded
% moving unrewarded
% unrewarded stops with licks
% unrewarded stops without licks
% 
% Figure for combined planes (SO vs SP+SR+SLM) for GRABA and GRABDA-mutant
% 
% Figures for combined 
% significance analysis:
% comparison between GRABDA and GRABDA-mutant
% 3 strategies:
% All events combined early and late days
% 4 early and late days of each mouse
% Mean of 4early and late days each mouse
% 
% 
% Section 2:
% 
% draw comparison between the categories. Let’s say single vs double CS…
% for each mouse.


close all
clear all
saving=0;
% savepath='G:\dark_reward\dark_reward\figures ';
 filepath = 'C:\Users\workstation4\Desktop\04242023\HRZ_reward_figures'
timeforpost = [0 1];
timeforpre=[-1 0];
timeforpost=[0 1.5];
for allcat=1%7%1:11%1:11
    % path='G:\'
    % path='G:\analysed\HRZ\'
    %     path='G:\dark_reward\';
    %     path='G:\dark_reward\solenoid_unrew';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_old\';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_new';
    % path='G:\dark_reward\solenoid_HRZ\after_vac';
    %     path='G:\dark_reward\dark_reward_allmouse_2'; %% WITHOUT EARLIEST DAYS
%     path='G:\dark_reward\dark_reward_allmouse_3';%% WITH EARLIEST DAYS
        path='F:\HRZ_workspaces'
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
    
    
%     workspaces = {'156_dark_reward_workspace.mat','157_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
%         '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat','181_dark_reward_workspace.mat'};
    
         workspaces = {'156_HRZ_workspace.mat','157_HRZ_workspace.mat','167_HRZ_workspace.mat','168_HRZ_workspace.mat','169_HRZ_workspace.mat',...
            '171_HRZ_workspace' '170_HRZ_workspace.mat' '179_HRZ_workspace_d1-d5_d17-d21','158_HRZ_workspace.mat'};
    
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
    ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
    %     %157 roi labels
    ROI_labels{2} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
    %     %158 roi
    ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     % fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8]};
    %158 roi
    ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    
    %     %%%168
    ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %171
    ROI_labels{6} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    %     %170
    ROI_labels{7} = {'Plane 1 SR','Plane 2 SR_SP','Plane 2 SP','Plane 3 SP','Plane 3 SP_SO', 'Plane 4 SO'};
     %     %179
    ROI_labels{8} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
       %     %158
    ROI_labels{9} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
    
    
    %        fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8],[1:10],[1:8],[1:11]};
%     %%%179
%     ROI_labels{8} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     
%     %%%181
%     ROI_labels{9} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %     %
    
    
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
    
    cats5={'roi_dop_allsuc_perirewardCS' 'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
        'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov' 'roi_dop_allsuc_mov_reward' 'roi_dop_allsuc_mov_no_reward'...
        'roi_dop_allsuc_nolick_stop_no_reward' 'roi_dop_allsuc_lick_stop_no_reward'}
    cats6={'roi_roe_allsuc_perirewardCS' 'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
        'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov' 'roe_allsuc_mov_reward' 'roe_allsuc_mov_no_reward'...
        'roe_allsuc_nolick_stop_no_reward' 'roe_allsuc_lick_stop_no_reward'}
    cats7={'roi_dop_alldays_planes_periCS','roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_perireward_double','roi_dop_alldays_planes_success_stop'...
        'roi_dop_alldays_planes_success_stop_reward','roi_dop_alldays_planes_success_stop_no_reward','roi_dop_alldays_planes_success_mov','roi_dop_alldays_planes_success_mov_reward','roi_dop_alldays_planes_success_mov_no_reward'...
        'roi_dop_alldays_planes_success_nolick_stop_no_reward','roi_dop_alldays_planes_success_lick_stop_no_reward'};
    
    
    
    
    dopvariablename = cats5{allcat};
    dopvariablename_alldays=cats7{allcat};
    roevariablename = cats6{allcat};
    
    
    
    setylimmanual2=[0.976 1.08];
    setylimmanual = [0.976 1.09];
    roerescale = [0.983 0.992];
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
        earlydays=[1:4];
        %         latedays=tdays-3:tdays;
        
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)
        %%% change for 313/79
        
        
        roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        
        
        %          files=1:size(roi_dop_alldays_planes_perireward,1);
        
        
        
%         if currmouse==1
%             files=[1:9 11];
%             %             files = fileslist{currmouse};
%         else
%             
%             files=1:size(roi_dop_alldays_planes_perireward,1);
%         end
                files=1:size(roi_dop_alldays_planes_perireward,1);
        find_figure(strcat('early and late days allmouse',cats5{allcat}))
        
        %     cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
        %     cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
        %     cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
        %     cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
        %     cats5={ 'roi_dop_allsuc_perireward'};
        %     cats6={'roi_roe_allsuc_perireward'};
        if strcmp(roevariablename(end-1:end),'_0')
            for checkforemptyfiles = 1:size(roe_alldays_planes_perireward_0,1)
                if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
                    roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    
                    dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                end
            end
            
            
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            dopvariable_alldays=eval(dopvariablename_alldays);
            
            
            
            
            
            roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
            
        else
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            dopvariable_alldays=eval(dopvariablename_alldays);
        end
        %     color = Pcolor;
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s;',cats5{allcat}));%%%20 days dop
            %             subplot(2,length(workspaces),currmouse)
            subplot(2,length(workspaces),length(workspaces)+currmouse)
            ylims = ylim;
             xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
%             dopvariable(:,jj,:) = squeeze(dopvariable(:,jj,:))-repmat(nanmean(dopvariable(:,jj,find(xax>=timeforpre(1)&xax<timeforpre(2))),3),1,size(dopvariable,3))+1;
            
            %                         xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
            
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            [x,y]=find((squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:))==0));
            
            yax=nanmean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1);
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
            se_yax=nanstd(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1));
            hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                h10.patch.FaceAlpha = 0.07;
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
                text(xt(jj),yt(jj),ROI_labels{currmouse}{jj},'Color',color{jj})
            end
            title(workspaces{currmouse}(1:3))
            if currmouse==20
                ylim([0.98 1.07])
            else
                
                ylim(setylimmanual);
            end
            ;
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
                ylabel('Late Days')
            end
            
            %%%% compute mean
            lt_days=fliplr(length(files)-earlydays+1);
            pst=nanmean(squeeze(dopvariable(lt_days,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
            pre=nanmean(squeeze(dopvariable(lt_days,jj,40+ceil(timeforpost(1)-timeforpost(2)/5*40)-1:40+ceil(timeforpost(1)/5*40)-1)),2);
            mean_pre_pst=nanmean(pst);
            meanpstmouse{currmouse,jj} = mean_pre_pst;
            se_pre_pst=nanstd(pst)./sqrt(size(pst,1));
            
            pstmouse{currmouse,jj}(2,:) = pst; premouse{currmouse,jj}(2,:) = pre;
            roi_idx{currmouse,jj}=ROI_labels{currmouse}{jj};
            %             corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:3);
            pstmouse_allcat{allcat,2} = pstmouse;
            premouse_allcat{allcat,2} = premouse;
            
            allmouse_dop{2}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));           
            allmouse_roe{2}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            allmouse_time{2}{currmouse,jj} = xax;
            if allcat<=3
                allmouse_dop_alldays{2}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))'))';
            else
                allmouse_dop_alldays{2}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))));
            end
            
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
    
    %%%saving part
    
    
    
    
    
    
    
    
    
    
    
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
            ispresent = cellfun(@(s) ~isempty(strfind(s,num2str(bb))),roi_idx(fkfk,:));
            [x y]=find(ispresent);
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
    
    %     find_figure(strcat('late days allmouse',cats5{allcat}))
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
    
    
    
    
    
    
    
    for currmouse = 1:length(workspaces)
        load([path '\' workspaces{currmouse}])
        %     if currmouse == 1
        %         close
        %     end
        %         files = fileslist{currmouse};
        
        %     find_figure(strcat('allmouse',cats5{allcat}))
        find_figure(strcat('early and late days allmouse',cats5{allcat}))
        
        %     cats5={'roi_dop_allsuc_perirewardCS' 'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
        %         'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov'}
        %     cats6={'roi_roe_allsuc_perirewardCS' 'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
        %         'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov'  }
        roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
        
        %     cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
        %     cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
        %     cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
        %     cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
        %     cats5={ 'roi_dop_allsuc_perireward'};
        %     cats6={'roi_roe_allsuc_perireward'};
        if strcmp(roevariablename(end-1:end),'_0')
            for checkforemptyfiles = 1:size(roe_alldays_planes_perireward_0,1)
                if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
                    roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
                    
                    dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                    dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(dop_alldays_planes_perireward{files(1),1}));
                end
            end
            
            
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            dopvariable_alldays=eval(dopvariablename_alldays);
            
            
            
            
            roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
            
        else
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            dopvariable_alldays=eval(dopvariablename_alldays);
            
        end
        %     color = Pcolor;
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s;',cats5{allcat}));%%%20 days dop
            subplot(2,length(workspaces),currmouse)
            ylim(setylimmanual);
            ylims = ylim;
             xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
%             dopvariable(:,jj,:) = squeeze(dopvariable(:,jj,:))-repmat(nanmean(dopvariable(:,jj,find(xax>=timeforpre(1)&xax<timeforpre(2))),3),1,size(dopvariable,3))+1;
          
            %         xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            yax=mean(squeeze(dopvariable(files(earlydays),jj,:)),1);
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
            se_yax=nanstd(squeeze(dopvariable(files(earlydays),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(earlydays),jj,:)),1));
            hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                if ~isempty (h10.patch)
                    h10.patch.FaceColor = color{jj};
                end
                h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                h10.patch.FaceAlpha = 0.07;
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
            end
            if currmouse==20
            ylim([0.98 1.06])
            else
                
                ylim(setylimmanual);
            end
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
                plot(xax,nanmean(squeeze(roevariable(files(earlydays),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
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
            
            
            %%%% compute mean
            %         early_days=fliplr(length(files)-earlydays+1);
            pst=nanmean(squeeze(dopvariable(earlydays,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
            pre=nanmean(squeeze(dopvariable(earlydays,jj,40+ceil(timeforpost(1)-timeforpost(2)/5*40)-1:40+ceil(timeforpost(1)/5*40)-1)),2);
            
            
            
            mean_pre_pst=nanmean(pst);
            meanpstmouse{currmouse,jj} = mean_pre_pst;
            se_pre_pst=nanstd(pst)./sqrt(size(pst,1));
            pstmouse{currmouse,jj}(1,:) = pst; premouse{currmouse,jj}(1,:) = pre;
            roi_idx{currmouse,jj}=ROI_labels{currmouse}{jj};
            %             corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:3)
            pstmouse_allcat{allcat,1} = pstmouse;
            premouse_allcat{allcat,1} = premouse;
            
            
            allmouse_dop{1}{currmouse,jj}=squeeze(dopvariable(files(earlydays),jj,:));
            allmouse_roe{1}{currmouse,jj}=squeeze(roevariable(files(earlydays),1,:));
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            allmouse_time{1}{currmouse,jj} = xax;
            if allcat<=3
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(earlydays),jj))'))';
            else
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(earlydays),jj))));
            end
            
            
            %%%% compute area under the curve
            data=squeeze(dopvariable(earlydays,jj,40+ceil(0/5*40):40+ceil(4.5/5*40)));
            clear pst_ac
            for auc=1:size(data,1)
                pst_ac(auc,:)=trapz(1:size(data,2),data(auc,:));%%% 0.64-2.56
            end
            mean_ac_pre_pst=nanmean(pst_ac);
            meanacpstmouse{currmouse,jj} = mean_ac_pre_pst;
            se_ac_pre_pst=nanstd(pst_ac)./sqrt(size(pst_ac,1));
            mean_ac_pre_pst=nanmean(pst);
            ac_pstmouse{currmouse,jj} = pst_ac;
            
        end
    end
    
    
    if saving==1
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
        filename = [ 'dark_reward_'  dopvariablename];
    
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
            exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])

    end
    
    
    
      %%
    %%% earlycombine all SO
    timeforpost = [0 1];
    timeforpre=[-2 0];
    timeforpost=[0 1.5];
    
    find_figure(strcat('comballmouse',num2str(allcat)));clf
    xax = -4.9920:0.1280:4.9920;
    edges = xax(1)-nanmean(diff(xax))/2:nanmean(diff(xax)):xax(end)+nanmean(diff(xax))/2;
    roexax = -4.9920:0.032:4.9920;
    roeedges = roexax(1)-nanmean(diff(roexax))/2:nanmean(diff(roexax)):roexax(end)+nanmean(diff(roexax))/2;
    %%%%%%
    for p=1:2
        
%          cc=allmouse_dop{1,p}(2:6,1:3);out1= cat(1,cc{:});
%         cc2=allmouse_dop{1,p}(1,1:5); out2=(cat(1,cc2{:}));
%         withoutSO_allmouse_dop=[out2 ;out1];%% mouse1-6
        
%         cc=allmouse_roe{1,p}(2:6,1:3);out1= cat(1,cc{:});
%         cc2=allmouse_roe{1,p}(1,1:5); out2=(cat(1,cc2{:}));
%         withoutSO_allmouse_dop_roe=[out2 ;out1];%% mouse1-6
        
%          cc=allmouse_dop{1,p}(7:8,1:3);out1= cat(1,cc{:});
%         cc2=allmouse_dop{1,p}(6,1:5); out3=(cat(1,cc2{:}));
%         withoutSO_allmouse_mut=[out3; out1];
        
%         cc=allmouse_roe{1,p}(8:9,1:3);out1= cat(1,cc{:})
%         cc2=allmouse_roe{1,p}(7,1:5); out3=(cat(1,cc2{:}));
%         withoutSO_allmouse_mut_roe=[out3; out1];
        
        SO_allmouse_dop = [];
        SO_allmouse_mut = [];
        SO_allmouse_dop_alldays=[];
        SO_allmouse_mut_alldays=[];
        SO_allmouse_dop_roe=[];
        SO_allmouse_mut_roe=[];
        withoutSO_allmouse_dop = [];
        withoutSO_allmouse_mut = [];
        withoutSO_allmouse_mut_alldays=[];
        withoutSO_allmouse_dop_alldays= [];
        notSO_allmouse_idmut = [];
        notSO_allmouse_iddop = [];
        for ll = 1:length(workspaces)
            if~iscell(workspaces{ll})
                currroi = strcmp(ROI_labels{ll},'Plane 4 SO');
                
            else
                currroi = strcmp(ROI_labels{ll}{2},'Plane 4 SO');
            end
            withoutcurrroi = find(~currroi);
            currtime = allmouse_time{1,p}{ll,currroi};
            currindex = discretize(currtime,edges);
            
            if length(intersect(ll,[7:9]))>0
                notSO_allmouse_idmut = [notSO_allmouse_idmut withoutcurrroi];
                temp = [];
                for d = 1:4
                    temp(d,:) = accumarray(currindex',allmouse_dop{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
                SO_allmouse_mut=[SO_allmouse_mut; temp];
                temp = [];
                for d = 1:4
                    temp(d,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
        SO_allmouse_mut_alldays=[SO_allmouse_mut_alldays; temp];
        
        SO_allmouse_mut_roe=[SO_allmouse_mut_roe; allmouse_roe{1,p}{ll,currroi}];
        
          % for all the other planes
            temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:4
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_mut = [withoutSO_allmouse_mut;temp];
             
              temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:4
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_mut_alldays=[withoutSO_allmouse_mut_alldays;temp];
            %
            else
                
                
                % GRAB DA
                notSO_allmouse_iddop = [notSO_allmouse_iddop withoutcurrroi];
                           temp = [];
                for d = 1:4
                    temp(d,:) = accumarray(currindex',allmouse_dop{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
                SO_allmouse_dop=[SO_allmouse_dop; temp];
                temp = [];
                for d = 1:4
                    temp(d,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
        SO_allmouse_dop_alldays=[SO_allmouse_dop_alldays; temp];

        
        SO_allmouse_dop_roe=[SO_allmouse_dop_roe; allmouse_roe{1,p}{ll,currroi}(d,:)];

            % for all the other planes
            temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:4
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_dop = [withoutSO_allmouse_dop;temp];
            
             temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:4
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_dop_alldays=[withoutSO_allmouse_dop_alldays;temp];
            
            
            
            
            
            end
        end
        withoutSO_allmouse_mut_roe= SO_allmouse_mut_roe;
        withoutSO_allmouse_dop_roe= SO_allmouse_dop_roe;                
        
        
        find_figure(strcat('comballmouse',num2str(allcat)));subplot(2,2,(p-1)*2+1);
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
            ylabel('Early Days')
        else
            ylabel('Late Days')
        end
               yax=SO_allmouse_dop_roe;
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        
        
        
        %%%control
        subplot(2,2,((p-1)*2+2));
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
            ylabel('Early Days')
        else
            ylabel('Late Days')
        end
               yax=SO_allmouse_mut_roe;
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        %%%%%%%%%%%%%%% combine all the rest except SO
       
        
        subplot(2,2,(p-1)*2+1);
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
            ylabel('Early Days')
        else
            ylabel('Late Days')
        end
        
        
        
                yax=withoutSO_allmouse_dop_roe;
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        title('GrabDA')
        
        %%% control
        
        %          cc=allmouse_dop{1,p}(2:6,1:3);out1= cat(1,cc{:});
        %         cc2=allmouse_dop{1,p}(1,1:5); out2=(cat(1,cc2{:}));
        
       
        
        subplot(2,2,((p-1)*2+2));
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
            ylabel('Early Days')
            
        else
            ylabel('Late Days')
            xlabel('Time onset')
        end
        title('GrabDA-mutant')
        
        
        
        yax=withoutSO_allmouse_mut_roe;
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
        
        
        
        
        
        find_figure(strcat('casevscontrol allmouse',cats5{allcat}))
        
        
        subplot(4,2,(p-1)*4+1);
        yax=SO_allmouse_dop;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10= shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{end}; h10.mainLine.Color = color{end}; h10.edge(1).Color = color{end};
            h10.edge(2).Color=color{end};
        end
        
        
        
        
        yax=SO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h11 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h11.patch.FaceColor = color{end}/2; h11.mainLine.Color = color{end}/2; h11.edge(1).Color = color{end}/2;
            h11.edge(2).Color=color{end}/2; h11.mainLine.LineStyle='-'; h11.edge(2).LineStyle='-';
        end
        ylim(setylimmanual2);
        legend([h11.edge(2) h10.edge(2) ], 'SO-GRABDA-mut','SO-GRABDA','onset','Location','northwest')
        
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        if p==1
            ylabel('Early Days')
        else
            ylabel('Late Days')
            xlabel('Time onset')
        end
        
        
        subplot(4,2,((p-1)*4+2)); hold on
        yax=withoutSO_allmouse_dop;
        
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        h10 = shadedErrorBar(xax,yax',se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{1}; h10.mainLine.Color = color{1}; h10.edge(1).Color = color{1};
            h10.edge(2).Color=color{1};
        end
        
        
        yax=withoutSO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        h11 = shadedErrorBar(xax,yax',se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h11.patch.FaceColor = color{1}/2; h11.mainLine.Color = color{1}/2; h11.edge(1).Color = color{1}/2;
            h11.edge(2).Color=color{1}/2; h11.mainLine.LineStyle='-';h11.edge(2).LineStyle='-';
        end
        legend([h11.edge(2) h10.edge(2)], 'AllexceptSO-GRABDA-mut','AllexceptSO-GRABDA','onset','Location','northwest')
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        if p==1
            ylabel('Early Days')
        else
            ylabel('Late Days')
            xlabel('Time onset')
        end
        
        
%         find_figure(strcat('significance allmouse',cats5{allcat}))
        
         grabda_rows=[1 :size(workspaces,2)-3];
    grabdamut_rows=[grabda_rows(end)+1: size(workspaces,2)];
        
    pst=find(xax>timeforpre(1)&xax<=timeforpre(2)); pst=find(xax>timeforpost(1)&xax<=timeforpost(2));
    
%     pstE157 = find(xax>timeforpost(1)+0.5&xax<=timeforpost(2)+0.5);
    
    mtm_vals_mut=[];
    temp = [0 find(diff(notSO_allmouse_idmut)<=0) length(notSO_allmouse_idmut)]*4;%4=numdays
    for n = 1:length(temp)-1
        mtm_vals_mut(n) = mean(mean(withoutSO_allmouse_mut(temp(n)+1:temp(n+1),pst),2));
    end
    
    mtm_vals_dop=[];
    temp = [0 find(diff(notSO_allmouse_iddop)<=0) length(notSO_allmouse_iddop)]*4; %4=numdays
    for n = 1:length(temp)-1
%         if n == 2
%             mtm_vals_dop(n) = mean(mean(withoutSO_allmouse_dop(temp(n)+4:4:temp(n+1),pstE157),2));
%         else
        mtm_vals_dop(n) = mean(mean(withoutSO_allmouse_dop(temp(n)+1:temp(n+1),pst),2));
%         end
    end
    
    mtm_vals_SOdop = [];
    for n = 1:size(SO_allmouse_dop,1)/4
%         if n == 2
%             mtm_vals_SOdop(n) = mean(mean(SO_allmouse_dop(n*4,pstE157),2));
%         else
    mtm_vals_SOdop(n) = mean(mean(SO_allmouse_dop(n*4-3:n*4,pst),2));
%         end
    end
    
    mtm_vals_SOmut = [];
    for n = 1:size(SO_allmouse_mut,1)/4
    mtm_vals_SOmut(n) = mean(mean(SO_allmouse_mut(n*4-3:n*4,pst),2));
    end

    subplot(4,2,(p-1)*4+3);
    
    scatter(ones(size(mtm_vals_SOdop)),mtm_vals_SOdop,25,'r','filled','Jitter','on','Jitteramount',0.04)
    hold on
   
     
    xlim([0.5 2.5])
    xlims = xlim;
    [~,pl] = ttest(mtm_vals_SOdop-1);
    text(1,nanmean(mtm_vals_SOdop),['p:' num2str(pl)])
    scatter(ones(size(mtm_vals_dop))*2,mtm_vals_dop,25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
        [~,pl] = ttest(mtm_vals_dop-1);
    text(2,nanmean(mtm_vals_dop),['p:' num2str(pl)])
    plot(xlims,[1 1],'--','Color','k')
    
    plot([1 2],[nanmean(mtm_vals_SOdop) nanmean(mtm_vals_dop)],'ks')
%     errorbar([1 2],[nanmean(mtm_vals_SOdop) nanmean(mtm_vals_dop)],[nanstd(mtm_vals_SOdop) nanstd(mtm_vals_dop)]/sqrt(length(mtm_vals_dop)),[nanstd(mtm_vals_SOdop) nanstd(mtm_vals_dop)]/sqrt(length(mtm_vals_dop)),'k','Capsize',0,'LineStyle','none','LineWidth',0.5)
plot([1 1],nanmean(mtm_vals_SOdop)+ [nanstd(mtm_vals_SOdop) -nanstd(mtm_vals_SOdop)]/sqrt(length(mtm_vals_dop)),'k-','LineWidth',1.5)
    plot([2 2],nanmean(mtm_vals_dop)+ [nanstd(mtm_vals_dop) -nanstd(mtm_vals_dop)]/sqrt(length(mtm_vals_dop)),'k-','LineWidth',1.5)
       if p == 1
        ylabel('Early Days')
    else
        ylabel('Late Days')
       end
        xticks([1 2])
    xticklabels({'SO','otherPlanes'})
    title('GrabDA')
     ylim(setylimmanual2)
%     yticks(0.985:0.015:1.05)
    
    
    
    subplot(4,2,(p-1)*4+4);
    
    hold on
    scatter(ones(size(mtm_vals_mut))*2,mtm_vals_mut,25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
    
 scatter(ones(size(mtm_vals_SOmut)),mtm_vals_SOmut,25,[1 0 0],'filled','Jitter','on','Jitteramount',0.04)
    
        [~,pl] = ttest(mtm_vals_SOmut-1);
    text(1,nanmean(mtm_vals_SOmut),['p:' num2str(pl)])
    
    [~,pl] = ttest(mtm_vals_mut-1);
    text(2,nanmean(mtm_vals_mut),['p:' num2str(pl)])
    xlim([0.5 2.5])
     plot(xlims,[1 1],'--','Color','k')
     
      
    plot([1 2],[nanmean(mtm_vals_SOmut) nanmean(mtm_vals_mut)],'ks')
%     errorbar([1 2],[nanmean(mtm_vals_SOmut) nanmean(mtm_vals_mut)],[nanstd(mtm_vals_SOmut) nanstd(mtm_vals_mut)]/sqrt(length(mtm_vals_mut)),[nanstd(mtm_vals_SOdop) nanstd(mtm_vals_dop)]/sqrt(length(mtm_vals_dop)),'k','Capsize',0,'LineStyle','none','LineWidth',0.5)
    plot([1 1],nanmean(mtm_vals_SOmut)+ [nanstd(mtm_vals_SOmut) -nanstd(mtm_vals_SOmut)]/sqrt(length(mtm_vals_mut)),'k-','LineWidth',1.5)
    plot([2 2],nanmean(mtm_vals_mut)+ [nanstd(mtm_vals_mut) -nanstd(mtm_vals_mut)]/sqrt(length(mtm_vals_mut)),'k-','LineWidth',1.5)
   ylim(setylimmanual2)
%     yticks(0.985:0.015:1.05 )
    
    if p == 1
        ylabel('Early Days')
    else
        ylabel('Late Days')
    end
    xticks([1 2])
    xticklabels({'SO','otherPlanes'})
    title('GrabDA-mut')
    
        if saving
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
        filename = [ 'combmouse_dark_reward'  dopvariablename];
        
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
            exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])
        
        
       
        %%significance
        significance_plot_alldays_master %%%all events
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
        if saving
        filename = [ 'sig_dark_reward_allevents'  dopvariablename];
      
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
            exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])
        
        end
        significance_plot_days_master %% days
        
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
        if saving
        filename = [ 'sig_dark_reward_4days'  dopvariablename];
        
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
            exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])
        end
        
        significance_plot_meandays_master %%%mean days
        
        if saving
        figHandles=gcf;
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
        filename = [ 'sig_dark_reward_meandays'  dopvariablename];
       
        for i = 1:size(figHandles,1)
            fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
            exportgraphics(figHandles(i),fn,'ContentType','vector')
        end
        disp(['figure saved in: ' fn])
        end
        end
        
        
        
        
        
    end
        
        
    end
    
%     if saving==1
%         figHandles=gcf;
%         set(gcf,'units','normalized','outerposition',[0 0 1 1])
%         
%         filename = [ 'dark_reward_combineallmouse'  dopvariablename];
%         filepath = 'C:\Users\workstation4\Desktop\01162023\dark_reward_figures'
%         for i = 1:size(figHandles,1)
%             fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%             exportgraphics(figHandles(i),fn,'ContentType','vector')
%         end
%         disp(['figure saved in: ' fn])
%         
%         
%     end
    
    
    %%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%
    
    
    
    
    
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


% %%%signficance












% %  % save all the current figures
% % % %             figHandles = findall(0,'Type','figure')
%             figHandles=gcf;
%             filename = 'allmouse_HRZ_allmouse_mearly_mlate_success_norew_start'
%             filepath = 'C:\Users\workstation4\Desktop\1212022'
%             for i = 1:size(figHandles,1)
%                 fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%                 exportgraphics(figHandles(i),fn,'ContentType','vector')
%             end
%             disp(['figure saved in: ' fn])
%
%
% %
% %
%%
%%% draw comparison between the categories
cats7={'roi_dop_alldays_planes_perireward_double' }

cats8={'roi_dop_alldays_planes_perireward','roi_dop'}

cats9={ 'roe_allsuc_perireward_double'}

cats10={'roe_allsuc_perireward'}



cats7={'roi_dop_alldays_planes_peridoubleCS' }

cats8={'roi_dop_alldays_planes_periCS'}

cats9={'roi_roe_allsuc_perireward_doubleCS'}

cats10={'roi_roe_allsuc_perirewardCS'}


% cats7={' roi_dop_alldays_planes_success_stop_reward' }
% 
% cats8={'roi_dop_alldays_planes_success_stop_no_reward'}
% 
% cats9={'roe_allsuc_stop_reward'}
% 
% cats10={'roe_allsuc_stop_no_reward'}



cats7={' roi_dop_alldays_planes_success_mov_reward' }

cats8={'roi_dop_alldays_planes_success_mov_no_reward'}

cats9={'roe_allsuc_mov_reward'}

cats10={'roe_allsuc_mov_no_reward'}

% 
% cats7={'roi_dop_alldays_planes_success_lick_stop_no_reward'}
% 
% cats8={'roi_dop_alldays_planes_success_nolick_stop_no_reward'}
% 
% cats9={'roe_allsuc_lick_stop_no_reward'}
% 
% cats10={'roe_allsuc_nolick_stop_no_reward'}

path='F:\HRZ_workspaces'

workspaces = {'156_HRZ_workspace.mat','157_HRZ_workspace.mat','167_HRZ_workspace.mat','168_HRZ_workspace.mat','169_HRZ_workspace.mat',...
            '171_HRZ_workspace' '170_HRZ_workspace.mat'};



setylimmanual2=[0.985 1.02]
setylimmanual = [0.985 1.02];
roerescale = [0.986 0.995];
maxspeedlim = 25; %cm/s
setxlimmanualsec = [-5 5];


%     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
%     %157 roi labels
ROI_labels{2} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
%     %158 roi
ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     % fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8]};
%158 roi
ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

%     %%%168
ROI_labels{5} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %171
ROI_labels{6} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %170
ROI_labels{7} = {'Plane 1 SR','Plane 2 SR_SP','Plane 2 SP','Plane 3 SP','Plane 3 SP_SO', 'Plane 4 SO'};
%        fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8],[1:10],[1:8],[1:11]};
%%%179
ROI_labels{8} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

%%%181
ROI_labels{9} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %


figure;
setylimmanual = [0.985 1.02];
setxlimmanualsec = [-5 5];
% Pcolor = {[0 0 1],[0 1 0],[0 1 1],[1 1 0],[204 164 61]/256,[231 84 128]/256};



timeforpre=[-2 0];
timeforpost=[-0.5 0];  clear vals
% mousestyle =
pstmouse = {};
for allcat=1%%categories
    for currmouse=1:length(workspaces)%%%mouse
        
        dopvariablename_var1 = cats7{allcat};
        dopvariablename_var2 = cats8{allcat};
        
        
        roevariablename_var1 = cats9{allcat};
        roevariablename_var2 = cats10{allcat};
        
       
           
        
        
        
        
        
        
        load([path '\' workspaces{currmouse}])
        earlydays=[1:5];
        
        
        roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        
        
        dopvariable_1=eval(dopvariablename_var1);
        dopvariable_2=eval(dopvariablename_var2);
        
        dopvariable_1 = cellfun(@transpose,dopvariable_1,'UniformOutput',false);
        dopvariable_2 = cellfun(@transpose,dopvariable_2,'UniformOutput',false);
        
        
        
        roevariable_1=eval(roevariablename_var1);
        roevariable_2=eval(roevariablename_var2);
        
        
        files = 1:size(dopvariable_1,1);
        
        
        files1=find(sum(cellfun(@(x)size(x ,2)==79, dopvariable_1),2)==0)
        files2=files(find(sum(cellfun(@(x)size(x ,2)==0, dopvariable_1),2)==0))
        files=intersect(files1,files2);
        
        lt_days=files(fliplr((length(files))-earlydays+1))
        er_days=files(earlydays);
        
        
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
        
        
        
        for p=1:2%cat1 vs cat2
            if p==1
                planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
                color = planecolors(roiplaneidx);
                color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
                
            else
                planecolors={[0 0 0.25],[0 1 0],[204 164 61]/256,[150 60 80]/256};
                color = planecolors(roiplaneidx);
                color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
            end
            
            
            merge_myax=[];allmerge_myax1=[];   allmerge_myax2=[];
            
            
            
            for jj =1:size(dopvariable_1,2) %1:size(dopvariable_1,2)%%rois
                
                
                var_1=cell2mat(dopvariable_1(lt_days,jj)');
                var_2=cell2mat(dopvariable_2(lt_days,jj)');
                
                %                 roe_var_1=cell2mat(roevariable_1(lt_days,jj)');
                %                 roe_var_2=cell2mat(roevariable_2(lt_days,jj)');
                
                xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
                pre=find(xax>timeforpre(1)&xax<=timeforpre(2)); pst=find(xax>timeforpost(1)&xax<=timeforpost(2));
                m_var1=mean(var_1(pre,:),1); m_var2=mean(var_2(pre,:));
                
                
                %                 if jj == size(dopvariable_1,2)
                
                
                %                 end
                
                
                
                
                xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
                if p==1
                    yax=nanmean(var_2,2);   se_yax=std(var_2,[],2)./sqrt(size(var_2,2))
                    
                else
                    yax=nanmean(var_1,2);   se_yax=std(var_1,[],2)./sqrt(size(var_1,2))
                end
                
                
                
                if p==1
                    subplot(4,length(workspaces),currmouse)
                    hold on
                    ylim(setylimmanual);
                    ylims = ylim;
                    plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
                    
                    minyax = min(yax);
                    maxyax = max(yax);
                    patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
                    patchy.EdgeAlpha = 0;
                    patchy.FaceAlpha =0.1;
                    hold on
                    plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
                    
                    
                else
                    subplot(4,length(workspaces),2*length(workspaces)+currmouse)
                    hold on
                    ylim(setylimmanual);
                    ylims = ylim;
                    plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
                    minyax = min(yax);
                    maxyax = max(yax);
                    patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
                    patchy.EdgeAlpha = 0;
                    patchy.FaceAlpha=0.1;
                    hold on
                    plot(timeforpost,[ylims(2)-0.01*diff(ylims) ylims(2)-0.01*diff(ylims)],'k-')
                    
                end
                
                
                
                
                
                
                if jj<size(dopvariable_1,2)%%mergea all other regions
                    merge_myax=[merge_myax  yax];
                    
                    allmerge_myax1=[allmerge_myax1 var_1];
                    allmerge_myax2=[allmerge_myax2 var_2];
                end
                
                
                
                
                if jj==size(dopvariable_1,2)
                    subplot(4,length(workspaces),currmouse)
                    hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
                    h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                    h10.edge(2).Color=color{jj};
                    ylim(setylimmanual);
                    pls = plot([0 0],ylims,'--k','Linewidth',1);
                    ylim(ylims)
                    xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                    x1=plot(xax,nanmean(squeeze(roevariable_1(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k--')
                    %
                    x2=plot(xax,nanmean(squeeze(roevariable_2(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
                    
                    %                      xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                    %                     plot(xax,nanmean(squeeze(roevariable_1(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
                    
                    title(workspaces{1,currmouse}(1:3))
                    xlabel('first lick after reward')
                end
            end
            
            %%%plot the remaining planes
            subplot(4,length(workspaces),2*length(workspaces)+currmouse)
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            merge_syax=std(merge_myax,[],2)/sqrt(size(merge_myax,2))
            hold on, h11 = shadedErrorBar(xax,mean(merge_myax,2),se_yax,[],1);
            h11.mainLine.Color = color{1}; h11.edge(1).Color = color{1};
            h11.edge(2).Color=color{1};
            xlabel('first lick after reward')
            
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
            x1=plot(xax,nanmean(squeeze(roevariable_1(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k--')
            x2=plot(xax,nanmean(squeeze(roevariable_2(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
            
            legend([x1 x2],'reward','un-rew','Location','northwest')
            
            
            
            
            withso_vals{1,currmouse}=mean(var_1(pst,:),1);
            withso_vals{2,currmouse}=mean(var_2(pst,:),1);
            
            withoutso_vals{1,currmouse}=mean(allmerge_myax1(pst,:),1);
            withoutso_vals{2,currmouse}=mean(allmerge_myax2(pst,:),1);
            
            
        end
        
        
        %%%with so stats and plots
        subplot(4,length(workspaces),length(workspaces)+currmouse)
        ya=[withso_vals{1,currmouse} NaN(1,6) withso_vals{2,currmouse}];
        xa=[ones(size(withso_vals{1,currmouse})) NaN(1,6) 2*ones(size(withso_vals{2,currmouse}))];
        
        scatter(xa,ya,5,color{size(dopvariable_1,2)}, 'filled')
        set(gca,'xlim',[0 3]); hold on
        m_grp= [mean(withso_vals{1,currmouse}) mean(withso_vals{2,currmouse})]
        se_grp=[std(withso_vals{1,currmouse})/sqrt(size(withso_vals{1,currmouse},2)) std(withso_vals{2,currmouse}/sqrt(size(withso_vals{2,currmouse},2)))]
        errorbar(m_grp,se_grp)
        [h,p]=ttest2(withso_vals{1,currmouse},withso_vals{2,currmouse})
        text(1.5,1.025, num2str(p))
        set(gca,'ylim',[0.95 1.025])
        xticks([1 2])
        xticklabels({'reward','unrew'})
        
        
        %%%without so stats and plots
        subplot(4,length(workspaces),3*length(workspaces)+currmouse)
        ya=[withoutso_vals{1,currmouse} NaN(1,6) withoutso_vals{2,currmouse}];
        xa=[ones(size(withoutso_vals{1,currmouse})) NaN(1,6) 2*ones(size(withoutso_vals{2,currmouse}))];
        
        scatter(xa,ya,5,color{1}, 'filled')
        set(gca,'xlim',[0 3]); hold on
        m_grp= [mean(withoutso_vals{1,currmouse}) mean(withoutso_vals{2,currmouse})]
        se_grp=[std(withoutso_vals{1,currmouse})/sqrt(size(withoutso_vals{1,currmouse},2)) std(withoutso_vals{2,currmouse}/sqrt(size(withoutso_vals{2,currmouse},2)))]
        errorbar(m_grp,se_grp)
        [h,p]=ttest2(withoutso_vals{1,currmouse},withoutso_vals{2,currmouse})
        text(1.5,1.025, num2str(p))
        set(gca,'ylim',[0.95 1.025])
        xticks([1 2])
        xticklabels({'reward','unrew'})
        
        
        
    end
%     figHandles=gcf;
%     set(gcf,'units','normalized','outerposition',[0 0 1 1])
%     
%     filename = [ 'dark_reward_doubleallmouse_-0.5_3s'  dopvariablename_var1];
%     filepath = 'C:\Users\workstation4\Desktop\01162023\dark_reward_figures\double_reward_figures'
%     for i = 1:size(figHandles,1)
%         fn = fullfile(filepath,[filename '.pdf']);  %in this example, we'll save to a temp directory.
%         exportgraphics(figHandles(i),fn,'ContentType','vector')
%     end
%     disp(['figure saved in: ' fn])
    
    
    
    
    
end


%% case vs control



allmouse_RampMdl = {};
SOslopes = [];
SORsquared = [];
notSOslopes = [];
notSORsquared = [];

for currmouse = 1:size(allmouse_dop{2},1)
    figure;

xax = allmouse_time{2}{currmouse,1};
lastindex = find(~cellfun(@isempty,allmouse_dop{2}(currmouse,:)),1,'last');
yax = nanmean(allmouse_dop{2}{currmouse,lastindex});

mdl = fitlm(xax(1:find(xax<=0,1,'last')),yax(1:find(xax<=0,1,'last')));
allmouse_RampMdl{1,currmouse} = mdl;
plot(xax,yax)
hold on
xlims = xlim;
plot(xlims,xlims*mdl.Coefficients.Estimate(2)+mdl.Coefficients.Estimate(1))

SOslopes(currmouse) = mdl.Coefficients.Estimate(2);
SORsquared(currmouse) = mdl.Rsquared.Ordinary;

xax = allmouse_time{2}{currmouse,1};
lastindex = find(~cellfun(@isempty,allmouse_dop{2}(currmouse,:)),1,'last');
yax = nanmean(cell2mat(cellfun(@nanmean,allmouse_dop{2}(currmouse,1:lastindex-1)','UniformOutput',0)));

mdl = fitlm(xax(1:find(xax<=0,1,'last')),yax(1:find(xax<=0,1,'last')));
allmouse_RampMdl{2,currmouse} = mdl;
notSOslopes(currmouse) = mdl.Coefficients.Estimate(2);
notSORsquared(currmouse) = mdl.Rsquared.Ordinary;
plot(xax,yax)
hold on
xlims = xlim;
plot(xlims,xlims*mdl.Coefficients.Estimate(2)+mdl.Coefficients.Estimate(1))
end
% SOslopes(2) = NaN;
% notSOslopes(2) = NaN;




figure;
subplot(4,2,3)
scatter(ones(size(SOslopes(1:6))),SOslopes(1:6),25,'r','filled','Jitter','on','Jitteramount',0.04)
hold on
scatter(ones(size(SOslopes(7:end)))*2,SOslopes(7:end),25,'r','filled','Jitter','on','Jitteramount',0.04)

 plot([1 2],[nanmean(SOslopes(1:6)) nanmean(SOslopes(7:end))],'ks')
    errorbar([1 2],[nanmean(SOslopes(1:6)) nanmean(SOslopes(7:end))],[nanstd(SOslopes(1:6))/sqrt(length(SOslopes(1:6))) nanstd(SOslopes(7:end))/sqrt(length(SOslopes(7:end)))],'k','Capsize',0,'LineStyle','none','LineWidth',1.5)
   xlim([0.5 2.5])
   plot([0.5 2.5],[0 0],'k--')
   ylabel('Linear Fit Slope')
   xticks(1:2)
    title('SO')
   xticklabels({'GrabDa','GrabDa-Mut'})
   ylim([-4 2]*10^-3)
   yticks((-4:2:2)*10^-3)
%    [~,p] = ttest2(SOslopes(1:6),SOslopes(7:end));
%    text(1.5,nanmean(SOslopes),['p:' num2str(p)])
    [~,p] = ttest(SOslopes(1:6));
    text(1,nanmean(SOslopes(1:6)),['p:' num2str(p)])
    [~,p] = ttest(SOslopes(7:end));
    text(2,nanmean(SOslopes(7:end)),['p:' num2str(p)])
   
   
 subplot(4,2,4)

scatter(ones(size(notSOslopes(1:6))),notSOslopes(1:6),25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
hold on

scatter(ones(size(notSOslopes(7:end)))*2,notSOslopes(7:end),25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
 plot([1 2],[nanmean(notSOslopes(1:6)) nanmean(notSOslopes(7:end))],'ks')
    errorbar([1 2],[nanmean(notSOslopes(1:6)) nanmean(notSOslopes(7:end))],[nanstd(notSOslopes(1:6))/sqrt(length(SOslopes(1:6))) nanstd(notSOslopes(7:end))/sqrt(length(SOslopes(7:end)))],'k','Capsize',0,'LineStyle','none','LineWidth',1.5)
   xlim([0.5 2.5])
   plot([0.5 2.5],[0 0],'k--')
   ylabel('Linear Fit Slope')
   xticks(1:2)
  title('notSO')
   xticklabels({'GrabDa','GrabDa-Mut'})
   ylim([-4 2]*10^-3)
   yticks((-4:2:2)*10^-3)
%    [~,p] = ttest2(notSOslopes(1:6),notSOslopes(7:end));
%    text(1.5,nanmean(notSOslopes),['p:' num2str(p)])
[~,p] = ttest(notSOslopes(1:6));
    text(1,nanmean(notSOslopes(1:6)),['p:' num2str(p)])
    [~,p] = ttest(notSOslopes(7:end));
    text(2,nanmean(notSOslopes(7:end)),['p:' num2str(p)])
   
%
% figure;
subplot(4,2,7)
scatter(ones(size(SORsquared(1:6))),SORsquared(1:6),25,'r','filled','Jitter','on','Jitteramount',0.04)
hold on
scatter(ones(size(SORsquared(7:end)))*2,SORsquared(7:end),25,'r','filled','Jitter','on','Jitteramount',0.04)

 plot([1 2],[nanmean(SORsquared(1:6)) nanmean(SORsquared(7:end))],'ks')
    errorbar([1 2],[nanmean(SORsquared(1:6)) nanmean(SORsquared(7:end))],[nanstd(SORsquared(1:6))/sqrt(length(SORsquared(1:6))) nanstd(SORsquared(7:end))/sqrt(length(SORsquared(7:end)))],'k','Capsize',0,'LineStyle','none','LineWidth',1.5)
   xlim([0.5 2.5])
   plot([0.5 2.5],[0 0],'k--')
   ylabel('rsquared')
   xticks(1:2)
    title('SO')
   xticklabels({'GrabDa','GrabDa-Mut'})
%    ylim([-4 2]*10^-3)
%    yticks((-4:2:2)*10^-3)
%    [~,p] = ttest2(SORsquared(1:6),SORsquared(7:end));
%    text(1.5,nanmean(SORsquared),['p:' num2str(p)])
   
   
 subplot(4,2,8)

scatter(ones(size(notSORsquared(1:6))),notSORsquared(1:6),25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
hold on

scatter(ones(size(notSORsquared(7:end)))*2,notSORsquared(7:end),25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
 plot([1 2],[nanmean(notSORsquared(1:6)) nanmean(notSORsquared(7:end))],'ks')
    errorbar([1 2],[nanmean(notSORsquared(1:6)) nanmean(notSORsquared(7:end))],[nanstd(notSORsquared(1:6))/sqrt(length(SORsquared(1:6))) nanstd(notSORsquared(7:end))/sqrt(length(SORsquared(7:end)))],'k','Capsize',0,'LineStyle','none','LineWidth',1.5)
   xlim([0.5 2.5])
   plot([0.5 2.5],[0 0],'k--')
   ylabel('Rsqaured')
   xticks(1:2)
  title('notSO')
   xticklabels({'GrabDa','GrabDa-Mut'})
%    ylim([-3.5 1.5]*10^-3)
%    yticks((-3.5:1:1.5)*10^-3)
%    [~,p] = ttest2(notSORsquared(1:6),notSORsquared(7:end));
%    text(1.5,nanmean(notSORsquared),['p:' num2str(p)])

%%

for currmouse = 1:size(allmouse_dop{2},1)
    lastindex = find(~cellfun(@isempty,allmouse_dop{2}(currmouse,:)),1,'last');
    cell2mat(cellfun(@nanmean,allmouse_dop{2}(currmouse,1:lastindex-1)','UniformOutput',0));
end