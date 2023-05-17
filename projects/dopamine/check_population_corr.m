
% path = 'N:\Munni';
%%% MM DA
%%% population plots for early and late days for all the categories
%%%%%%Input: workspace folder 
%%%%%%output: 1) population mean 4 early and 4 late days actiivty of each
%%%%%%mouse individually
% % Input: workspaces folder
% % Output:Figures
% % single rew CS: First lick after reward
% %  	single rew CS
% % 	doubles: First lick after reward
% % stopping success trials: all stops
% % non rewarded stops
% %    rewarded stops
% % moving success trials: all motions
% % moving rewarded
% % moving unrewarded
% % unrewarded stops with licks
% % unrewarded stops without licks
% % 
% % Figure for combined planes (SO vs SP+SR+SLM) for GRABA and GRABDA-mutant
% % 
% % Figures for combined 
% % significance analysis:
% % comparison between GRABDA and GRABDA-mutant
% % 3 strategies:
% % All events combined early and late days
% % 4 early and late days of each mouse
% % Mean of 4early and late days each mouse
% % 
% % 
% % Section 2:
% % 
% % draw comparison between the categories. Let’s say single vs double CS…
% % for each mouse.




close all
clear all
saving=0;
savepath='X:\dopamine_analysis ';
filepath = 'X:\dopamine_analysis';
timeforpost = [0 1];
timeforpre=[-2 0];
timeforpost=[0 0.5];
for allcat=1%1:11%1:11%1:11%1:11
    
     
    % path='G:\'
    % path='G:\analysed\HRZ\'
    %     path='G:\dark_reward\';
    %     path='G:\dark_reward\solenoid_unrew';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_old\';
    % path='G:\dark_reward\solenoid_HRZ\before_vac_new';
    % path='G:\dark_reward\solenoid_HRZ\after_vac';
    %     path='G:\dark_reward\dark_reward_allmouse_2'; %% WITHOUT EARLIEST DAYS
    path='X:\dopamine_imaging';%% WITH EARLIEST DAYS
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
    
    
    workspaces = {'e193checkearly_dark_reward_workspace.mat',
        'e194checkearly_dark_reward_workspace.mat'};
    
    % ZD changed this for her 2 mice too
    ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO'};
    ROI_labels{2} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO'};
   
    
    xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.02];
    allmouse_dop={}; allmouse_roe={};
    
    % fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
    
    
    
    mice = cellfun(@(x) x(1:4),workspaces,'UniformOutput',0);
    
    cats5={'roi_dop_allsuc_perirewardCS' 'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
        'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov' 'roi_dop_allsuc_mov_reward' 'roi_dop_allsuc_mov_no_reward'...
        'roi_dop_allsuc_nolick_stop_no_reward' 'roi_dop_allsuc_lick_stop_no_reward'};
    cats6={'roi_roe_allsuc_perirewardCS' 'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
        'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov' 'roe_allsuc_mov_reward' 'roe_allsuc_mov_no_reward'...
        'roe_allsuc_nolick_stop_no_reward' 'roe_allsuc_lick_stop_no_reward'};
    cats7={'roi_dop_alldays_planes_periCS','roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_perireward_double','roi_dop_alldays_planes_success_stop'...
        'roi_dop_alldays_planes_success_stop_reward','roi_dop_alldays_planes_success_stop_no_reward','roi_dop_alldays_planes_success_mov','roi_dop_alldays_planes_success_mov_reward','roi_dop_alldays_planes_success_mov_no_reward'...
        'roi_dop_alldays_planes_success_nolick_stop_no_reward','roi_dop_alldays_planes_success_lick_stop_no_reward'};
    
    dopvariablename = cats5{allcat};
    dopvariablename_alldays=cats7{allcat};
    roevariablename = cats6{allcat};
    
    setylimmanual2=[0.985 1.02];
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
        
        
        % ZD: there is no explanation for why this was done and it gives an
        % error if you have less than 11 days
%         if currmouse==1
%             files=[1:9 11];
%             %             files = fileslist{currmouse};
%         else
            
        files=1:size(roi_dop_alldays_planes_perireward,1);

%                 files=1:size(roi_dop_alldays_planes_perireward,1);
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
        h1=[];
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s;',cats5{allcat}));%%%20 days dop
            %             subplot(2,length(workspaces),currmouse)
            subplot(2,length(workspaces),length(workspaces)+currmouse)
            ylims = ylim;
            
            
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
            hold on;
            h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                text(xt(jj),yt(jj),ROI_labels{currmouse}{jj},'Color',color{jj})
            end
            title(workspaces{currmouse}(1:4))
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
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:4);
            pstmouse_allcat{allcat,2} = pstmouse;
            premouse_allcat{allcat,2} = premouse;
            
            allmouse_dop{2}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{2}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            if allcat<=3
                allmouse_dop_alldays{2}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))'))';
            else
                allmouse_dop_alldays{2}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))));
            end                       
        end      
        
    end
    
    %%%saving part
     
    
    roi=unique([ROI_labels{:}]);
   
    
    clear comb_vals mean_comb_vals mean_vals comb_roi_vals ac_vals comb_ac_vals
    
  
    for currmouse = 1:length(workspaces)
        load([path '\' workspaces{currmouse}])
        
        allids=cellfun(@(x) strfind(x,'\'), pr_dir0, 'UniformOutput', false);
        check=unique( cellfun(@(c) c(end), allids))
        allidsn=cellfun(@(x) x(check:end), pr_dir0, 'UniformOutput', false);
        B = regexp(allidsn,'\d*','Match');
        day_labels= cellfun(@(x) (x(1)), B);
        
        %     if currmouse == 1
        %         close
        %     end
        %         files = fileslist{currmouse};
        
        %     find_figure(strcat('allmouse',cats5{allcat}))
        
        
%          if currmouse==1
%             files=[1:9 11];
%             %             files = fileslist{currmouse};
%         else
            
        files=1:size(roi_dop_alldays_planes_perireward,1);
%         end
        
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
        color(duplicate_indices)=cellfun(@(x) x/10 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
        
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
            end
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
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:4);
            pstmouse_allcat{allcat,1} = pstmouse;
            premouse_allcat{allcat,1} = premouse;
            
            
            allmouse_dop{1}{currmouse,jj}=squeeze(dopvariable(files(earlydays),jj,:));
            allmouse_roe{1}{currmouse,jj}=squeeze(roevariable(files(earlydays),1,:));
            
            if allcat<=3
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(earlydays),jj))'))';
            else
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(earlydays),jj))));
            end
            
            
            find_figure(strcat('compute correlation_',workspaces{currmouse}(1:4),cats5{allcat}));
            
            % ZD changed from this hard coded number 79. where did that
            % come from?
            dop_sig_alldays=squeeze(dopvariable(files,jj,:));
            roe_sig_alldays=imresize(squeeze(roevariable(files,1,:)),[size(dop_sig_alldays,1) size(dop_sig_alldays,2)]);
            
            dop_sig_early=squeeze(dopvariable(files(earlydays),jj,:));
            dop_sig_late=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            
            roe_sig_early=imresize(squeeze(roevariable(files(earlydays),1,:)),[1 size(dop_sig_alldays,2)]);
            roe_sig_late=imresize(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:)),[1 size(dop_sig_alldays,2)]);
            
            yax1 = [0.995 1.003];
            yax2 = [0.985 1.02];
            cax1 = [0.996 1.005];
            cax2 = [0.985 1.02];
            speedax = [-1 65];
            speedcax = [-15 55];
            
            ax(jj)=subplot(size(dopvariable,2)+1,2,((jj+1)-1)*2+1); imagesc(dop_sig_alldays); hold on;
            plot([length(xax(xax>=0)), length(xax(xax>=0))],[0 length(files)+0.5],'Linewidth',2); colormap(ax(jj), ...
                fake_parula); 
               caxis(cax2); colorbar('Location','westoutside')
            
            ylabel('Days');  xt=[10 70]; 
            title('dff all days')
            
            ax(jj)=subplot(size(dopvariable,2)+1,2,1); imagesc(roe_sig_alldays);hold on
            plot([ceil(size(roevariable,3)/2), ceil(size(roevariable,3)/2)],[0 length(files)+0.5],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
            colorbar('Location','westoutside')
            yticks([1:length(day_labels)])
            yticklabels(day_labels)
            ylabel('Days')
            title('roe all days')
           
            
            corr_mat=[];
            for co=1:size(roe_sig_alldays)
                [corr_val corr_p corr_up corr_low ]=corrcoef(dop_sig_alldays(co,:),roe_sig_alldays(co,:));
                corr_mat(co,:)=[corr_val(1,2) corr_p(1,2)];
            end
            
            
            allmouse_corr{jj,currmouse}=corr_mat;
          
            
            subplot(size(dopvariable,2)+1,2,2); hold on; plot(corr_mat(:,1),'Color',color{jj});
            xlims=xlim; h1(jj)=plot([xlims(1) xlims(2)],[0 0]); set(gca,'ylim',[-1 1]);
            ylabel('Days');  xlabel('r');
            %             text(0,max(corr_mat(:,1)), ROI_labels{currmouse}{jj},'FontSize',10,'Color',color{jj})
            lgnd=legend(h1,ROI_labels{currmouse},'Location','northwest')
            set(lgnd,'color','none');
            legend boxoff

            hb=subplot(size(dopvariable,2)+1,2,(jj)*2+2),b=bar(corr_mat(:,1));b(1).FaceColor=color{jj}; set(gca,'ylim',[-1 1]);
            hold on
            sig_corr=find(corr_mat(:,2)<0.05);
            scatter(sig_corr, corr_mat(sig_corr),'r*');
            
            text(size(dopvariable,1),max(corr_mat(:,1)), ROI_labels{currmouse}{jj})
            ylabel('R'); xlabel('days')
            %             text(0,max(corr_mat(:,1)), ROI_labels{currmouse}{jj},'FontSize',10,'Color',color{jj})
            lgnd=legend(hb,ROI_labels{currmouse}{jj},'Location','northwest')
            
            set(lgnd,'color','none');legend boxoff   
            corr_allmouse{currmouse}{jj,1}=corr_mat(:,1);   
            
            if saving==1
                figHandles=gcf;
                set(gcf,'units','normalized','outerposition',[0 0 1 1])
                
                filepathn=strcat(filepath ,'\corr_figures');
                mkdir (filepathn, dopvariablename)
                filepathn1=strcat(filepathn ,'\',dopvariablename);
                filename = [ 'corrvals_' workspaces{currmouse}(1:4) dopvariablename];
                
                for i = 1:size(figHandles,1)
                    fn = fullfile(filepathn1,[filename '.pdf']);  %in this example, we'll save to a temp directory.
                    exportgraphics(figHandles(i),fn,'ContentType','vector')
                end
                disp(['figure saved in: ' fn])
                
            end
            
            
            
            
            
            find_figure(strcat('early and late days allmouse',cats5{allcat}))
             
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
    
    %%%correlation mouse combined
    % ZD: this is coded to number of mice, ZD fixed but only for her mice

    
    for jj=1:size(corr_allmouse,2)
       
        lt_days_Corr=cellfun(@(x) x(size(x,1)-3:size(x,1),1),corr_allmouse{1,jj} ,'UniformOutput',false);
        SO_corr(jj,1)=mean(cell2mat(lt_days_Corr(size(lt_days_Corr,1),1)));
        without_SO_corr(jj,1)=mean(cell2mat(lt_days_Corr(1:size(lt_days_Corr,1)-1,1)));
    end
    % length(workspaces) = number of mice I have, all are experimental
    find_figure('corr_loc_latedays');clf
    clear xticks
    Grabda_mean=[SO_corr(1:length(workspaces),1)   without_SO_corr(1:length(workspaces),1)];
    Grabda=[SO_corr(1:length(workspaces),1)  ; without_SO_corr(1:length(workspaces),1)];
    Grabda_idx=[ones(1,length(workspaces))  2*ones(1,length(workspaces))];
    subplot(2,2,1), s=scatter(Grabda_idx,Grabda,'ko','filled');
    set(gca,'xlim',[0 3]); hold on;
    errorbar(mean(Grabda_mean),std(Grabda_mean)/sqrt( size(Grabda_mean,1)),'k-','Linewidth',2);
    plot([0 3],[0 0],'k--'); plot(Grabda_mean','Color',[0.7 0.7 0.7])
    set(gca,'ylim',[-1 1])
    xticks([1 2])
    xticklabels({'SO','allregions except SO'})
    s.MarkerFaceAlpha =0.4;
    
   
    set(gca,'ylim',[-1 1])
    xticks([1 2])
    xticklabels({'SO','allregions except SO'})
    s.MarkerFaceAlpha =0.4;
    
    
    
    
    
end
    
 
   
    %% all individual plots
    
    
    

    figure;
    
    
    for currmouse = 1:size(allmouse_corr,2)
        
        
        subplot(1,size(allmouse_corr,2),currmouse)
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
        for jj = 1:length(color)
            plot(allmouse_corr{jj,currmouse}(:,1),'Color',color{jj})
            hold on
        end
        axis square
        title(workspaces{currmouse}(1:4))
        sgtitle("correlation between dopamine and roe across days")
    end
    
    
    %% scatter of the average of last four days.
    
    
    figure; 
    grabdarows = length(workspaces);
    currmouse = length(workspaces);
    SOplane = find(~cellfun(@isempty,allmouse_corr(:,currmouse)),1,'last');
    SOloccor(currmouse) = nanmean(allmouse_corr{SOplane,currmouse}(end-3:end,1));
    notSOloccor(currmouse) = nanmean(cellfun(@(x) nanmean(x(end-3:end,1)),allmouse_corr(1:SOplane-1,currmouse),'UniformOutput',1));

    scatter(ones(size(grabdarows)),notSOloccor(grabdarows),25,'b','filled')
    hold on
    scatter(ones(size(grabdarows))*2,SOloccor(grabdarows),25,'r','filled')
    plot([1 2],[notSOloccor(grabdarows); SOloccor(grabdarows)],'Color',[0.55 0.55 0.55])
    errorbar([1 2],[nanmean(notSOloccor(grabdarows)) nanmean(SOloccor(grabdarows))],[nanstd(notSOloccor(grabdarows)) nanstd(SOloccor(grabdarows))]/sqrt(length(grabdarows)-sum(isnan(SOloccor(grabdarows)))),'k-','Capsize',0,'LineWidth',1.5)
    plot([1 2],[nanmean(notSOloccor(grabdarows)) nanmean(SOloccor(grabdarows))],'ks','LineWidth',2)
    xlim([0.5 2.5])
    ylabel('Loccomotion Correlation')
    xticks([1 2])
    xticklabels({'notSO','SO'})
    title('GrabDa')
    ylim([0 0.3])
%     [~,p] = 
    
%     
%     figure;
%     scatter(ones(size( mutrows)),notSOloccor( mutrows),25,[0 0 0.5],'filled')
%     hold on
%     scatter(ones(size( mutrows))*2,SOloccor( mutrows),25,[0.5 0 0],'filled')
%     plot([1 2],[notSOloccor( mutrows); SOloccor( mutrows)],'Color',[0.55 0.55 0.55])
%     errorbar([1 2],[nanmean(notSOloccor( mutrows)) nanmean(SOloccor( mutrows))],[nanstd(notSOloccor( mutrows)) nanstd(SOloccor( mutrows))]/sqrt(length( mutrows)-sum(isnan(SOloccor( mutrows)))),'k-','Capsize',0,'LineWidth',1.5)
%     plot([1 2],[nanmean(notSOloccor( mutrows)) nanmean(SOloccor( mutrows))],'ks','LineWidth',2)
%     xlim([0.5 2.5])
%     ylabel('Loccomotion Correlation')
%     xticks([1 2])
%     xticklabels({'notSO','SO'})
%     title('GrabDa-Mut')
%     yticks(-1:0.5:1)
%     