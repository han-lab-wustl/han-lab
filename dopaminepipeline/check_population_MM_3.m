
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




% close all
clear all
saving=0;
savepath='G:\dark_reward\dark_reward\figures ';
 filepath = 'F:\dark_reward_figures'
timeforpost = [0 1];
timeforpre=[-2 0];
timeforpost=[0 0.5];
%%
for allcat=1%10:11%1:11%1:11%1:11

    path='F:\workspaces_darkreward';%% WITH EARLIEST DAYS

    workspaces = {'156_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
        '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace.mat','181_dark_reward_latedays_workspace.mat'}};
    

    
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
    
   
    
    
    
    
    xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.02];
    allmouse_dop={}; allmouse_roe={}; allmouse_time = {};
    
    % fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
    
    
    singleworkspace = ~cellfun(@iscell,workspaces);
    mice(singleworkspace) = cellfun(@(x) x(1:4),workspaces(singleworkspace),'UniformOutput',0);
    mice(~singleworkspace) = cellfun(@(x) x{1}(1:4),workspaces(~singleworkspace),'UniformOutput',0);
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
        if ~iscell(workspaces{currmouse})
        load([path '\' workspaces{currmouse}])
        currROI_labels = ROI_labels{currmouse};
        currtitle = workspaces{currmouse}(1:3);
        else
            load([path '\' workspaces{currmouse}{2}])
            currROI_labels = ROI_labels{currmouse}{2};
            currtitle = workspaces{currmouse}{1}(1:3);
        end

        tdays=length(pr_dir0);
        earlydays=[1:4];
        
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),currROI_labels,'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)
        
        
        roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        
        
        
        if currmouse==1
            files=[1:9 11];
        else
            
            files=1:size(roi_dop_alldays_planes_perireward,1);
        end
        find_figure(strcat('early and late days allmouse',cats5{allcat}))
        

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
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s;',cats5{allcat}));%%%20 days dop

            subplot(2,length(workspaces),length(workspaces)+currmouse)
            ylims = ylim;
            
             
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            [x,y]=find((squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:))==0));
            
            yax=nanmean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1);
            if jj == 1
                minyax = min(yax);
                maxyax = max(yax);
            else
                minyax = min(min(yax),minyax);
                maxyax = max(max(yax),maxyax);
            end
            se_yax=nanstd(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1));
            hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
            if sum(isnan(se_yax))~=length(se_yax)
                h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                text(xt(jj),yt(jj),currROI_labels{jj},'Color',color{jj})
                h10.patch.FaceAlpha = 0.07;
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
%                 h10.edge(1).LineWidth =
            end
           
            title(currtitle)
            ylim(setylimmanual);

            xlim(setxlimmanualsec)

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
            roi_idx{currmouse,jj}=currROI_labels{jj};
            %             corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=currtitle;
            pstmouse_allcat{allcat,2} = pstmouse;
            premouse_allcat{allcat,2} = premouse;
            
            allmouse_dop{2}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{2}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
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
            
            
            
            

            
            
        end
        currmouse
        if currmouse == 1
        plot([-4 -4], [1.01 1.02],'k-')
        text(-1,1.03,'0.01dFF or 25cm/s')
        end
        yticks([])
        
    end
    %% early days
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
    
%     roi=unique([ROI_labels{:}]);
    
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
    
    
    
    
    
    
    
    for currmouse = 1:length(workspaces)
        
         if ~iscell(workspaces{currmouse})
        load([path '\' workspaces{currmouse}])
        currROI_labels = ROI_labels{currmouse};
        currtitle = workspaces{currmouse}(1:3);
        else
            load([path '\' workspaces{currmouse}{1}])
            currROI_labels = ROI_labels{currmouse}{1};
            currtitle = workspaces{currmouse}{1}(1:3);
        end
     
        find_figure(strcat('early and late days allmouse',cats5{allcat}))
            roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) nanmean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,size(roe_alldays_planes_perireward_0{1},1));
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),currROI_labels,'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/10 ,color(duplicate_indices) ,'UniformOutput' ,false);
        
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
                    h10.patch.FaceAlpha = 0.07;
                end
                h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                h10.edge(2).Color=color{jj};
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
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
            roi_idx{currmouse,jj}=currROI_labels;
            %             corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=currtitle;
            pstmouse_allcat{allcat,1} = pstmouse;
            premouse_allcat{allcat,1} = premouse;
            
            
            allmouse_dop{1}{currmouse,jj}=squeeze(dopvariable(files(earlydays),jj,:));
            allmouse_roe{1}{currmouse,jj}=squeeze(roevariable(files(earlydays),1,:));
            allmouse_time{1}{currmouse,jj} = xax;
            if allcat<=3
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(earlydays),jj))'))';
            else
                allmouse_dop_alldays{1}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(earlydays),jj))));
            end
            
            
            
            
%             corr_mat=[];
%             for co=1:size(dop_sig,1)
%                 [corr_coef corr_p corr_up corr_lo]=corrcoef(dop_sig(co,:),roe_sig(co,:))
%                 corr_mat(co,:)=[corr_coef(1,2) corr_p(1,2)];
%             end
%             
%             
            
            
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
        yticks([])
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
    timeforpost=[0 1];
    
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
        for ll = 1:length(workspaces)
            if~iscell(workspaces{ll})
                currroi = strcmp(ROI_labels{ll},'Plane 4 SO');
                
            else
                currroi = strcmp(ROI_labels{ll}{2},'Plane 4 SO');
            end
            withoutcurrroi = find(~currroi);
            currtime = allmouse_time{1,p}{ll,currroi};
            currindex = discretize(currtime,edges);
            if length(intersect(ll,[6:9]))>0
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
                    temp(gss,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_mut = [withoutSO_allmouse_mut;temp];
            %
            else
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
                    temp(gss,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_dop = [withoutSO_allmouse_dop;temp];
            %
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
%%
% %%%signficance


%%% draw comparison between the categories
cats7={'roi_dop_alldays_planes_perireward_double' }

cats8={'roi_dop_alldays_planes_perireward','roi_dop'}

cats9={ 'roe_allsuc_perireward_double'}

cats10={'roe_allsuc_perireward'}



cats7={'roi_dop_alldays_planes_peridoubleCS' }

cats8={'roi_dop_alldays_planes_periCS'}

cats9={'roi_roe_allsuc_perireward_doubleCS'}

cats10={'roi_roe_allsuc_perirewardCS'}


cats7={' roi_dop_alldays_planes_success_stop_reward' }

cats8={'roi_dop_alldays_planes_success_stop_no_reward'}

cats9={'roe_allsuc_stop_reward'}

cats10={'roe_allsuc_stop_no_reward'}



cats7={' roi_dop_alldays_planes_success_mov_reward' }

cats8={'roi_dop_alldays_planes_success_mov_no_reward'};

cats9={'roe_allsuc_mov_reward'}

cats10={'roe_allsuc_mov_no_reward'}


cats7={'roi_dop_alldays_planes_success_lick_stop_no_reward'}

cats8={'roi_dop_alldays_planes_success_nolick_stop_no_reward'}

cats9={'roe_allsuc_lick_stop_no_reward'}

cats10={'roe_allsuc_nolick_stop_no_reward'}




path='G:\dark_reward\dark_reward_allmouse_3'
workspaces = {'156_dark_reward_workspace.mat','157_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
    '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat','181_dark_reward_workspace.mat'};

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
timeforpost=[0 1];  clear vals
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
                    yax=nanmean(var_2,2);   se_yax=nanstd(var_2,[],2)./sqrt(size(var_2,2))
                    
                else
                    yax=nanmean(var_1,2);   se_yax=nanstd(var_1,[],2)./sqrt(size(var_1,2))
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
                    if ~isnan(yax)
                    h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
                    h10.edge(2).Color=color{jj};
                    end
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
%                     xlabel('first lick after reward')
                      xlabel('stop init')
                end
            end
            
            %%%plot the remaining planes
            subplot(4,length(workspaces),2*length(workspaces)+currmouse)
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            merge_syax=nanstd(merge_myax,[],2)/sqrt(size(merge_myax,2))
            hold on, h11 = shadedErrorBar(xax,nanmean(merge_myax,2),se_yax,[],1);
            h11.mainLine.Color = color{1}; h11.edge(1).Color = color{1};
            h11.edge(2).Color=color{1};
%             xlabel('first lick after reward')
            xlabel('stop init')
            
            
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
            x1=plot(xax,nanmean(squeeze(roevariable_1(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k--')
            x2=plot(xax,nanmean(squeeze(roevariable_2(lt_days,1,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
            
%             legend([x1 x2],'reward','un-rew','Location','northwest')
            
               legend([x1 x2],'lck','no-lck','Location','northwest')
            
            
            withso_vals{1,currmouse}=nanmean(var_1(pst,:),1);
            withso_vals{2,currmouse}=nanmean(var_2(pst,:),1);
            
            withoutso_vals{1,currmouse}=nanmean(allmerge_myax1(pst,:),1);
            withoutso_vals{2,currmouse}=nanmean(allmerge_myax2(pst,:),1);
            
            
        end
        
        
        %%%with so stats and plots
        subplot(4,length(workspaces),length(workspaces)+currmouse)
        ya=[withso_vals{1,currmouse} NaN(1,6) withso_vals{2,currmouse}];
        xa=[ones(size(withso_vals{1,currmouse})) NaN(1,6) 2*ones(size(withso_vals{2,currmouse}))];
        
        scatter(xa,ya,5,color{size(dopvariable_1,2)}, 'filled')
        set(gca,'xlim',[0 3]); hold on
        m_grp= [nanmean(withso_vals{1,currmouse}) nanmean(withso_vals{2,currmouse})]
        se_grp=[nanstd(withso_vals{1,currmouse})/sqrt(size(withso_vals{1,currmouse},2)) nanstd(withso_vals{2,currmouse}/sqrt(size(withso_vals{2,currmouse},2)))]
        errorbar(m_grp,se_grp)
        [h,p]=ttest2(withso_vals{1,currmouse},withso_vals{2,currmouse})
        text(1.5,1.025, num2str(p))
        set(gca,'ylim',[0.95 1.025])
        xticks([1 2])
%         xticklabels({'reward','unrew'})
        xticklabels({'lick','no-lick'})
        
        %%%without so stats and plots
        subplot(4,length(workspaces),3*length(workspaces)+currmouse)
        ya=[withoutso_vals{1,currmouse} NaN(1,6) withoutso_vals{2,currmouse}];
        xa=[ones(size(withoutso_vals{1,currmouse})) NaN(1,6) 2*ones(size(withoutso_vals{2,currmouse}))];
        
        scatter(xa,ya,5,color{1}, 'filled')
        set(gca,'xlim',[0 3]); hold on
        m_grp= [nanmean(withoutso_vals{1,currmouse}) nanmean(withoutso_vals{2,currmouse})]
        se_grp=[nanstd(withoutso_vals{1,currmouse})/sqrt(size(withoutso_vals{1,currmouse},2)) nanstd(withoutso_vals{2,currmouse}/sqrt(size(withoutso_vals{2,currmouse},2)))]
        errorbar(m_grp,se_grp)
        [h,p]=ttest2(withoutso_vals{1,currmouse},withoutso_vals{2,currmouse})
        text(1.5,1.025, num2str(p))
        set(gca,'ylim',[0.95 1.025])
        xticks([1 2])
%         xticklabels({'reward','unrew'})
        
        xticklabels({'lick','no-lick'})
        
        
        
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
















