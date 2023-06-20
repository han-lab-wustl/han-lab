
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
timeforpost=[0 4.5];
%%
extracount = 0;
figure;
for allcat=[1 12]%10:11%1:11%1:11%1:11
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
    allmouse_dop={}; allmouse_roe={}; allmouse_time = {};
    
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
    
    
    
    setylimmanual2=[0.983 1.02];
    setylimmanual = [0.983 1.02];
    roerescale = [0.985 0.995];
    maxspeedlim = 25; %cm/s
    setxlimmanualsec = [-5 5];
    
    
    
    Pcolor = {[0 0 1],[0 1 0],[0 1 1],[1 1 0],[204 164 61]/256,[231 84 128]/256};
    
    pstmouse = {};

    
    
    
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
            ylim([0.985 1.06])

            lt_days=files(fliplr(length(files)-earlydays+1));
            
            
            allmouse_dop{extracount}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{extracount}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            
            if allcat<=3 || allcat == 12
                allmouse_dop_alldays{extracount}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))'))';
            else
                allmouse_dop_alldays{extracount}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))));
            end

            %%%% compute area under the curve
            pstwindow = find(allmouse_time{extracount}{currmouse,jj}>=timeforpost(1)&allmouse_time{extracount}{currmouse,jj}<=timeforpost(2));
            data=squeeze(dopvariable(lt_days,jj,pstwindow))-1;
            meandata = nanmean(data);
            mean_ac_pre_pst=trapz(allmouse_time{extracount}{currmouse,jj}(pstwindow),meandata);
            meanacpstmouse{extracount}{currmouse,jj} = mean_ac_pre_pst;
%             se_ac_pre_pst=nanstd(pst_ac)./sqrt(size(pst_ac,1));
%             mean_ac_pre_pst=nanmean(pst);
%             ac_pstmouse{currmouse,jj} = pst_ac;
 
        end
        currmouse
  
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
    SOdoubleArea = [SOdoubleArea temp(currmouse,lastindex)];
    notSOdoubleArea = [notSOdoubleArea nanmean(temp(currmouse,1:lastindex-1))];
    
    
    temp = padcatcell2mat(meanacpstmouse{1});
    lastindex = find(~isnan(temp(currmouse,:)),1,'last');
    SOsingleArea = [SOsingleArea temp(currmouse,lastindex)];
    notSOsingleArea = [notSOsingleArea nanmean(temp(currmouse,1:lastindex-1))];
end
    
comparestatsplotspair(SOdoubleArea(1:5),SOsingleArea(1:5),'r',[0.5 0 0],'k',[1 2])
hold on
comparestatsplotspair(notSOdoubleArea(1:5),notSOsingleArea(1:5),[0.55 0.55 0.55],[0.25 0.25 0.25],'k',[4 5])