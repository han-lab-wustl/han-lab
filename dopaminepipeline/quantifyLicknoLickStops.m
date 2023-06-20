
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
timeforpre=[-5 -1];
timeforpost=[0 0.5];
%%
extracount = 0;
figure;
meanalldays_acpstmouse = {};
meanalldays_speedpremouse = {};
meanacpstmouse = {};
allmouse_dop={}; allmouse_roe={}; allmouse_time = {};
for allcat=[10 11]%10:11%1:11%1:11%1:11
extracount = extracount +1;
    %% settings
    path='F:\workspaces_darkreward';%% WITH EARLIEST DAYS

    workspaces = {'156_dark_reward_workspace.mat','167_dark_reward_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_workspace.mat',...
        '171_dark_reward_workspace' '170_dark_reward_workspace.mat','179_dark_reward_workspace.mat',{'181_dark_reward_earlydays_workspace_00.mat','181_dark_reward_latedays_workspace_00.mat'}};
    

    
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
    
   %%
    
    
    
    
    xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.02];
    
    
    % fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
    
    
    singleworkspace = ~cellfun(@iscell,workspaces);
    mice(singleworkspace) = cellfun(@(x) x(1:4),workspaces(singleworkspace),'UniformOutput',0);
    mice(~singleworkspace) = cellfun(@(x) x{1}(1:4),workspaces(~singleworkspace),'UniformOutput',0);
    cats5={'roi_dop_allsuc_perirewardCS' 'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
        'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov' 'roi_dop_allsuc_mov_reward' 'roi_dop_allsuc_mov_no_reward'...
        'roi_dop_allsuc_nolick_stop_no_reward' 'roi_dop_allsuc_lick_stop_no_reward' 'roi_dop_allsuc_perireward_doubleCS'}
    cats6={'roi_roe_allsuc_perirewardCS' 'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
        'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov' 'roe_allsuc_mov_reward' 'roe_allsuc_mov_no_reward'...
        'roe_allsuc_nolick_stop_no_reward' 'roe_allsuc_lick_stop_no_reward' 'roi_roe_allsuc_perireward_doubleCS'}
    cats7={'roi_dop_alldays_planes_periCS','roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_perireward_double','roi_dop_alldays_planes_success_stop'...
        'roi_dop_alldays_planes_success_stop_reward','roi_dop_alldays_planes_success_stop_no_reward','roi_dop_alldays_planes_success_mov','roi_dop_alldays_planes_success_mov_reward','roi_dop_alldays_planes_success_mov_no_reward'...
        'roi_dop_alldays_planes_success_nolick_stop_no_reward','roi_dop_alldays_planes_success_lick_stop_no_reward','roi_dop_alldays_planes_peridoubleCS'};
    
    
    
    
    dopvariablename = cats5{allcat};
    dopvariablename_alldays=cats7{allcat};
    roevariablename = cats6{allcat};
    
    
    
    setylimmanual2=[0.980 1.06];
    setylimmanual = [0.980 1.02];
    roerescale = [0.982 0.992];
    maxspeedlim = 25; %cm/s
    setxlimmanualsec = [-5 5];
    
    
    
    Pcolor = {[0 0 1],[0 1 0],[0 1 1],[1 1 0],[204 164 61]/256,[231 84 128]/256};
    
    pstmouse = {};

    
    
    
    for currmouse = 1:length(workspaces)-1
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
            xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            allmouse_time{extracount}{currmouse,jj} = xax;

%                 if size(roe_success_peristop,2)==79
%                     xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
%                 else
%                     xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
%                 end
            yax = nanmean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)));
            subplot(2,length(workspaces),currmouse+length(workspaces)*extracount-length(workspaces))
            plot(xax,yax,'Color',color{jj})
            hold on
            ylim(setylimmanual2)
            
            if jj == 1
                
                if size(roe_success_peristop,2)==79
                    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames;
                else
                    xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                end
                plot(xax,nanmean(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
            end
            
            lt_days=files(fliplr(length(files)-earlydays+1));
            title(currtitle)
            
            allmouse_dop{extracount}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{extracount}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            
            if allcat<=3 || allcat == 12
                allmouse_dop_alldays{extracount}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))'))';
            else
                allmouse_dop_alldays{extracount}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))));
            end

            %%%% compute area under the curve
            pstwindow = find(allmouse_time{extracount}{currmouse,jj}>=timeforpost(1)&allmouse_time{extracount}{currmouse,jj}<=timeforpost(2));
            data=squeeze(dopvariable(lt_days,jj,pstwindow));
            meandata = nanmean(data);
            meanalldays_acpstmouse{extracount}{currmouse,jj} = nanmean(data,2);
            mean_ac_pre_pst=nanmean(meandata,2);
% mean_ac_pre_pst=trapz(allmouse_time{extracount}{currmouse,jj}(pstwindow),meandata);
            meanacpstmouse{extracount}{currmouse,jj} = mean_ac_pre_pst;
%             se_ac_pre_pst=nanstd(pst_ac)./sqrt(size(pst_ac,1));
%             mean_ac_pre_pst=nanmean(pst);
%             ac_pstmouse{currmouse,jj} = pst_ac;

        prewindow = find(xax>=timeforpre(1)&xax<=timeforpre(2));
        if nansum(squeeze(roevariable(lt_days,jj)))>0
        data = squeeze(roevariable(lt_days,1,prewindow));
        else
            data = NaN(length(lt_days),length(prewindow));
        end
        meanalldays_speedpremouse{extracount}{currmouse,jj} = nanmean(data,2);


        end
        currmouse
  
    end
end
  %%
    %%% earlycombine all SO
    timeforpost = [0 1];
    timeforpre=[-2 0];
    timeforpost=[0 1.5];
    setylimmanual2 = [0.98 1.03];
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
        for ll = 1:length(workspaces)-1
            if~iscell(workspaces{ll})
                currroi = strcmp(ROI_labels{ll},'Plane 4 SO');
                
            else
                currroi = strcmp(ROI_labels{ll}{2},'Plane 4 SO');
            end
            withoutcurrroi = find(~currroi);
            currtime = allmouse_time{1,p}{ll,currroi};
            currindex = discretize(currtime,edges);
            
            if length(intersect(ll,[6:9]))>0
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
        
        
        find_figure(strcat('comballmouse',num2str(allcat)));
        subplot(2,2,1);
                yax=SO_allmouse_dop;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{4}/(rem(p,2)+1); h10.mainLine.Color = color{4}/(rem(p,2)+1); h10.edge(1).Color = color{4}/(rem(p,2)+1);
            h10.edge(2).Color=color{4}/(rem(p,2)+1);
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
%         if p==1
%             ylabel('Early Days')
%         else
%             ylabel('Late Days')
%         end
            title('GrabDA-SO')
               yax=SO_allmouse_dop_roe;
               if p == 2
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
                    else
                    plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.55 0.55 0.55])
               end
        
        
        
        %%%control
        subplot(2,2,2);
                yax=SO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{4}/(rem(p,2)+1); h10.mainLine.Color = color{4}/(rem(p,2)+1); h10.edge(1).Color = color{4}/(rem(p,2)+1);
            h10.edge(2).Color=color{4}/(rem(p,2)+1);
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5
%         if p==1
%             ylabel('Early Days')
%         else
%             ylabel('Late Days')
%         end
        title('GrabDA-mutantSO')
               yax=SO_allmouse_mut_roe;
               if p == 2
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
               else
                    plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.55 0.55 0.55])
               end
        
        %%%%%%%%%%%%%%% combine all the rest except SO
       
        
        subplot(2,2,3);
                yax= withoutSO_allmouse_dop;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax);
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{1}/(rem(p,2)+1); h10.mainLine.Color = color{1}/(rem(p,2)+1); h10.edge(1).Color = color{1}/(rem(p,2)+1);
            h10.edge(2).Color=color{1}/(rem(p,2)+1);
        end
        ylim(setylimmanual2);
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
        title('GRABDA-withoutSO')
        
        
        
                yax=withoutSO_allmouse_dop_roe;
                if p == 2
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
           else
                    plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.55 0.55 0.55])
               end
%         title('GrabDA')
        
        %%% control
        
        %          cc=allmouse_dop{1,p}(2:6,1:3);out1= cat(1,cc{:});
        %         cc2=allmouse_dop{1,p}(1,1:5); out2=(cat(1,cc2{:}));
        
       
        
        subplot(2,2,4);
          yax=withoutSO_allmouse_mut;
        se_yax=nanstd(yax,1)./sqrt(size(yax,1));
        yax=nanmean(yax)
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        if sum(isnan(se_yax))~=length(se_yax)
            h10.patch.FaceColor = color{1}/(rem(p,2)+1); h10.mainLine.Color = color{1}/(rem(p,2)+1); h10.edge(1).Color = color{1}/(rem(p,2)+1);
            h10.edge(2).Color=color{1}/(rem(p,2)+1);
        end
        ylim(setylimmanual2);
        
        
        ylims = ylim;
        pls = plot([0 0],ylims,'--k','Linewidth',1);
        ylim(ylims)
        pls.Color(4) = 0.5;
%         if p==1
%             ylabel('Early Days')
%             
%         else
%             ylabel('Late Days')
%             xlabel('Time onset')
%         end
        title('GrabDA-mutant-withoutSO')
        
        
        
        yax=withoutSO_allmouse_mut_roe;
        if  p == 2
        plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
           else
                    plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'Color',[0.55 0.55 0.55])
               end
        
        
        
        
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
     ylim([0.99 1.01])
    yticks(0.99:0.005:1.01)
    
    
    
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
   ylim([0.99 1.01])
    yticks(0.99:0.005:1.01)
    
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



 %%   
figure;

SOdoubleArea = [];
notSOdoubleArea = [];
SOsingleArea = [];
notSOsingleArea = [];

for currmouse = 1:size(meanacpstmouse{1},1)
    temp = padcatcell2mat(meanacpstmouse{2});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last');
    if ~isempty(temp(currmouse,lastindex))
    SOdoubleArea = [SOdoubleArea temp(currmouse,lastindex)];
    notSOdoubleArea = [notSOdoubleArea nanmean(temp(currmouse,1:lastindex-1))];
    else
        SOdoubleArea = [SOdoubleArea NaN];
        notSOdoubleArea = [notSOdoubleArea NaN];
    end
    
    
    temp = padcatcell2mat(meanacpstmouse{1});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last');
    if ~isempty(temp(currmouse,lastindex))
    SOsingleArea = [SOsingleArea temp(currmouse,lastindex)];
    notSOsingleArea = [notSOsingleArea nanmean(temp(currmouse,1:lastindex-1))];
     else
        SOsingleArea = [SOsingleArea NaN];
        notSOsingleArea = [notSOsingleArea NaN];
    end
end
    
comparestatsplotspair(SOdoubleArea(1:5),SOsingleArea(1:5),'r',[0.5 0 0],'k',[1 2])
hold on
comparestatsplotspair(notSOdoubleArea(1:5),notSOsingleArea(1:5),[0.55 0.55 0.55],[0.25 0.25 0.25],'k',[4 5])
xlim ([0 6])
xticks([1 2 4 5])
xticklabels({'Lick','no Lick','Lick','no Lick'})
%%
figure;

SOdoubleArea = [];
notSOdoubleArea = [];
SOsingleArea = [];
notSOsingleArea = [];

for currmouse = 1:size(meanalldays_acpstmouse{1},1)
    temp = padcatcell2mat(meanacpstmouse{2});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last');
    temp = padcatcell2mat(meanalldays_acpstmouse{2});
    if ~isempty(temp(currmouse,lastindex))
    SOdoubleArea = [SOdoubleArea temp(currmouse*4-3:currmouse*4,lastindex)'];
    notSOdoubleArea = [notSOdoubleArea nanmean(temp(currmouse*4-3:currmouse*4,1:lastindex-1),2)'];
    else
        SOdoubleArea = [SOdoubleArea NaN(1,4)];
        notSOdoubleArea = [notSOdoubleArea NaN(1,4)];
    end
    
   temp = padcatcell2mat(meanacpstmouse{2});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last'); 
    temp = padcatcell2mat(meanalldays_acpstmouse{1});
    if ~isempty(temp(currmouse,lastindex))
    SOsingleArea = [SOsingleArea temp(currmouse*4-3:currmouse*4,lastindex)'];
    notSOsingleArea = [notSOsingleArea nanmean(temp(currmouse*4-3:currmouse*4,1:lastindex-1),2)'];
     else
        SOsingleArea = [SOsingleArea NaN(1,4)];
        notSOsingleArea = [notSOsingleArea NaN(1,4)];
    end
end
    
comparestatsplotspair(SOdoubleArea(1:5*4),SOsingleArea(1:5*4),'r',[0.5 0 0],'k',[1 2])
hold on
comparestatsplotspair(notSOdoubleArea(1:5*4),notSOsingleArea(1:5*4),[0.55 0.55 0.55],[0.25 0.25 0.25],'k',[4 5])
xlim ([0 6])
xticks([1 2 4 5])
xticklabels({'Lick','no Lick','Lick','no Lick'})


%%
figure;

SOdoubleArea = [];
notSOdoubleArea = [];
SOsingleArea = [];
notSOsingleArea = [];

for currmouse = 1:size(meanalldays_speedpremouse{1},1)
    temp = padcatcell2mat(meanacpstmouse{2});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last');
    temp = padcatcell2mat(meanalldays_speedpremouse{2});
    if ~isempty(temp(currmouse,lastindex))
    SOdoubleArea = [SOdoubleArea temp(currmouse*4-3:currmouse*4,lastindex)'];
    notSOdoubleArea = [notSOdoubleArea nanmean(temp(currmouse*4-3:currmouse*4,1:lastindex-1),2)'];
    else
        SOdoubleArea = [SOdoubleArea NaN(1,4)];
        notSOdoubleArea = [notSOdoubleArea NaN(1,4)];
    end
    
   temp = padcatcell2mat(meanacpstmouse{2});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last'); 
    temp = padcatcell2mat(meanalldays_speedpremouse{1});
    if ~isempty(temp(currmouse,lastindex))
    SOsingleArea = [SOsingleArea temp(currmouse*4-3:currmouse*4,lastindex)'];
    notSOsingleArea = [notSOsingleArea nanmean(temp(currmouse*4-3:currmouse*4,1:lastindex-1),2)'];
     else
        SOsingleArea = [SOsingleArea NaN(1,4)];
        notSOsingleArea = [notSOsingleArea NaN(1,4)];
    end
end
    
comparestatsplotspair(SOdoubleArea(1:5*4),SOsingleArea(1:5*4),'r',[0.5 0 0],'k',[1 2])
hold on
% comparestatsplotspair(notSOdoubleArea(1:5*4),notSOsingleArea(1:5*4),[0.55 0.55 0.55],[0.25 0.25 0.25],'k',[4 5])
xlim ([0 3])
xticks([1 2])
xticklabels({'Lick','no Lick'})
ylabel('Speed')