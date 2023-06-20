% % % CS with US and  CS without activity
% % % Overlay of CS with US and  CS without activity with speed
% % % Difference in activity and speed behavior in CS with US and CS without activity conditions
% % % Significance and population summary of comparison between CS with US and  CS without activity
% % % 
% % % This has 2 categories:
% % % 1)	Comparison between CS with US and US without CS
% % % 2)	Comparison between CS with US and NEXT double reward if missed US without CS
% % % Change ids accordingly


close all
clear all
setylimmanual2 = [0.99 1.03];
cnt_2=0;
for allcat=[1 8]%1:8
    cnt_2=cnt_2+1;
    % path='G:\'
    % path='G:\analysed\HRZ\'
%     path='G:\dark_reward\';
%     path='G:\dark_reward\solenoid_unrew_3';
     path='F:\reward_addition';
    
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
    
%     % workspaces = {'148_dark_reward_workspace_02.mat','149_dark_reward_workspace.mat','156_dark_reward_workspace.mat',...
%     %     '157_dark_reward_workspace.mat','158_dark_reward_workspace.mat'}
%     % workspaces = {'168_dark_reward_workspace.mat','171_dark_reward_workspace_01.mat','170_dark_reward_workspace_04.mat'};
%     
    
%%%%%
% workspaces={'167_dark_reward_workspace.mat' '168_dark_reward_workspace.mat','169_dark_reward_workspace.mat', '171_dark_reward_workspace.mat', '170_dark_reward_workspace.mat' }
workspaces={'167_dark_reward_workspace.mat' '168_dark_reward_workspace.mat','169_dark_reward_workspace.mat' ,'171_dark_reward_workspace.mat' }




%%%168
ROI_labels{1} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%171
ROI_labels{2} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%%%
ROI_labels{3} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%171
ROI_labels{4} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
% %170
ROI_labels{5} ={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};;
%%%%%%
    
%     %148 labels %%MM
%     %%old batch
%     ROI_labels{1} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SR_SP','Plane 3 SP', 'Plane 4 SO'};%%MM
%     %149 dark Rewards labels
%     ROI_labels{2} = {'Plane 1 SR','Plane 1 SP','Plane 2 SP','Plane 2 SP_SO','Plane 3 SO','Plane 4 SO'};
%     %156 roi labels
%     ROI_labels{3} = {'Plane 1 SR','Plane 2 SP','Plane 2 SR_SP','Plane 3 SP_SO','Plane 3 SP','Plane 4 SO'};
%     %157 roi labels
%     ROI_labels{4} = {'Plane 1 SR','Plane 2 SP','Plane 3 SO','Plane 4 SO'};
%     %158 roi
%     ROI_labels{5} = { 'Plane 1 SR','Plane 2 SR_SP','Plane 3 SP', 'Plane 4 SP'};
%     % fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8]};
%     %%%168
%     ROI_labels{6} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %171
%     ROI_labels{7} = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
%     %170
%     ROI_labels{8} = {'Plane 1 SR','Plane 2 SR_SP','Plane 2 SP','Plane 3 SP','Plane 3 SP_SO', 'Plane 4 SO'};
%     
%     fileslist={[1:12],[1:12],[1:11 12 14],[1:13],[1:8],[1:10],[1:8],[1:11]};
    
%       fileslist={[1:5],[1:5],[1:5],[1:5],[1:4]};
%       fileslist={[1:2],[1:2],[1:2]};
 fileslist={[1:3],[1:3],[1:3],[1:2]};
    
    % new batch
    
    
    
    
    
    
    xt=[-3*ones(1,6)];
    yt=[1.005:0.003:1.04];
    
    
    % fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
    
    
    
    mice = cellfun(@(x) x(1:4),workspaces,'UniformOutput',0);
    
     cats5={'roi_dop_allsuc_perireward' 'roi_dop_allsuc_perireward_double' 'roi_dop_allsuc_stop'...
        'roi_dop_allsuc_stop_reward' 'roi_dop_allsuc_stop_no_reward' 'roi_dop_allsuc_mov'  'roi_dop_allsuc_perirewardCS' ,'roi_dop_allsuc_perireward_usonly'...
        'roi_dop_allsuc_rewusonly_single','roi_dop_allsuc_perirewardUS','roi_dop_allsuc_perirewarddoublemUS_CS'}
    cats6={'roe_allsuc_perireward' 'roe_allsuc_perireward_double'  'roe_allsuc_stop'...
        'roe_allsuc_stop_reward' 'roe_allsuc_stop_no_reward' 'roe_allsuc_mov'  'roi_roe_allsuc_perirewardCS', 'roi_roe_allsuc_rewusonly_lick'...
        'roi_roe_allsuc_rewusonly',' roi_roe_allsuc_perirewardUS','roi_roe_allsuc_perirewarddoublemUS_CS'}
    cats7={'roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_perireward_double','roi_dop_alldays_planes_success_stop'...
        'roi_dop_alldays_planes_success_stop_reward','roi_dop_alldays_planes_success_stop_no_reward','roi_dop_alldays_planes_success_mov', 'roi_dop_alldays_planes_unreward_single','roi_dop_alldays_planes_perireward_usonly'...
        'roi_dop_allsuc_rewusonly_single','roi_dop_allsuc_perirewardUS','roi_dop_allsuc_perirewarddoublemUS_CS'};
    dopvariablename = cats5{allcat};
    roevariablename = cats6{allcat};
    dopalldaysname = cats7{allcat};
    savepath = 'D:\munneworkspaces\HRZfigures\StartTriggered\summaryfigure\';
    saving = 0;
    
    
    setylimmanual = [0.97 1.06];
    roerescale = [0.976 0.985];
    maxspeedlim = 25; %cm/s
    setxlimmanualsec = [-5 5];
    
    
    
    Pcolor = {[0 0 1],[0 1 0],[0 1 1],[1 1 0],[204 164 61]/256,[231 84 128]/256};
    timeforpost = [0 2];
    % mousestyle =
    pstmouse = {};
    % cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    % cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    % cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    % cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    % cats5={ 'roi_dop_allsuc_perireward'};
    % cats6={'roi_roe_allsuc_perireward'};
  
    
    
    for currmouse = 1:length(workspaces)
        load([path '\' workspaces{currmouse}])
        %     if currmouse == 1
        %         close
        %     end
        
        planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels{currmouse},'UniformOutput',1);
        [v, w] = unique( roiplaneidx, 'stable' );
        duplicate_indices = setdiff( 1:numel(roiplaneidx), w )
        color = planecolors(roiplaneidx);
        color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)
        
        roe_allsuc_perireward = reshape(cell2mat(cellfun(@(x) mean(x,2)',roe_alldays_planes_perireward_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,313);
        roe_allsuc_perireward_double = reshape(cell2mat(cellfun(@(x) mean(x,2)',roe_alldays_planes_perireward_double_0(:,1),'UniformOutput',0)),size(roe_alldays_planes_perireward_0,1),1,313);
        
        files = fileslist{currmouse};
        find_figure(strcat('late days allmouse'))
          earlydays = files;
   
        
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
            dopvariable_alldays = eval(dopalldaysname);
            
            
            
            
            roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
            
        else
            dopvariable = eval(dopvariablename);
            roevariable = eval(roevariablename);
            dopvariable_alldays = eval(dopalldaysname);
        end
        %     color = Pcolor;
        for jj = 1:size(dopvariable,2)
            eval(sprintf('data1=%s',cats5{allcat}))%%%20 days dop
          subplot(2,length(workspaces),(cnt_2-1)*length(workspaces)+currmouse)
            ylim(setylimmanual);
            ylims = ylim;
            xax= frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
            yax=nanmean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1);
            yax
            df_f2(currmouse,jj,cnt_2,:)=yax;
            df_f2all{currmouse,cnt_2}=dopvariable;
             df_f2_CS{currmouse,2}=roi_dop_alldays_planes_perireward;
             df_f2_CS{currmouse,1}=roi_dop_alldays_planes_perireward_usonly;
             allmouse_time{cnt_2}{currmouse,jj} = xax;
             
             allmouse_dop{cnt_2}{currmouse,jj}=squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:));
            allmouse_roe{cnt_2}{currmouse,jj}=squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),1,:));
            
            if allcat<=3 || allcat == 8
                allmouse_dop_alldays{cnt_2}{currmouse,jj}=(cell2mat(cat(1,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))'))';
            else
                allmouse_dop_alldays{cnt_2}{currmouse,jj}=(cell2mat(cat(2,dopvariable_alldays(files(fliplr(length(files)-earlydays+1)),jj))));
            end
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
            se_yax=std(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)./sqrt(size(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1))
            if ~isnan(sum(se_yax))
                
                df_f2_se(currmouse,jj,cnt_2,:)=se_yax;
            else
                df_f2_se(currmouse,jj,cnt_2,:)=zeros(size(se_yax));
                se_yax=zeros(size(se_yax));
            end
                
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
            ylim(setylimmanual);
            %     yticks([])  ylim(setylimmanual);
            
            %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
            %     hp=plot(yax,'Color',color{jj})
            %     legend(hp,strcat('plane',num2str(jj)));hold on
            %     xlabel('Time(s)');
            xlim(setxlimmanualsec)
            %     set(gca,'ylim',[0.99 1.01])
            if jj == 1
                
                xaxn=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                plot(xaxn,nanmean(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
                
                roe_trace2(currmouse,jj,cnt_2,:)=nanmean(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)/maxspeedlim*diff(roerescale)+roerescale(1);
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
            
            if cnt_2==1
                title(strcat(workspaces{currmouse}(1:3),'US with CS'))
            else
                  title(strcat(workspaces{currmouse}(1:3),'US without CS'))
            end
                
            %%%% compute mean
            lt_days=fliplr(length(files)-earlydays+1);
            pst=nanmean(squeeze(dopvariable(lt_days,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
            mean_pre_pst=nanmean(pst);
            meanpstmouse{currmouse,jj} = mean_pre_pst;
            se_pre_pst=std(pst)./sqrt(size(pst,1));
            pstmouse{currmouse,jj} = pst;
            roi_idx{currmouse,jj}=ROI_labels{currmouse}{jj};
            corrcoef(pstmouse{currmouse,jj},pst)
            id_mouse{currmouse,jj}=workspaces{currmouse}(1:3)
            
             %%%% compute area under the curve
        data=squeeze(dopvariable(lt_days,jj,40+ceil(0/5*40):40+ceil(4.5/5*40)));
        clear pst_ac
        for auc=1:size(data,1)
            pst_ac(auc,:)=trapz(1:size(data,2),data(auc,:));%%% 0.64-2.56
        end
        mean_ac_pre_pst=nanmean(pst_ac);
        meanacpstmouse{currmouse,jj} = mean_ac_pre_pst;
        se_ac_pre_pst=std(pst_ac)./sqrt(size(pst_ac,1));
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
         if currmouse == 1
        plot([-4 -4], [1.01 1.02],'k-')
        text(-1,1.03,'0.01dFF or 25cm/s')
        end
        yticks([])
    end
    
    
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
    
%      find_figure(strcat('late days allmouse'))
% %      subplot(2,length(workspaces),currmouse+length(workspaces))
%     subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
%     scaling = 2;
%     spacescale = 0.2;
%     xtickm = [];
%     xtickl = {};
%     ylims= [0.985 1.02];
%     ptest = [];
%     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
%     for jj = 1:4
%         for currmouse = 1:length(workspaces)
%             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
%             %
%             for roin=1: size(comb_vals{jj,currmouse},2)
%                 if roin==1
%                 scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
%                     comb_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
%                 else
%                      scatter(ones(size(comb_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
%                     comb_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
%                 end
%                     
%             end
%             hold on
%             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
%             
%              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
%             ylims = ylim;
%             xtickl = [xtickl, workspaces{currmouse}(1:4)];
%             
%             if currmouse<length(workspaces)
%                 [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
%                 
%             end
%             ylims= [0.985 1.025];
%             set(gca,'ylim',ylims)
%         end
%         
%     end
%     
%     realhs = h;
%     realhs = reshape(realhs,size(ptest,1),size(ptest,2));
%     
%     for jj = 1:size(realhs,2)
%         for kk = 1:size(realhs,1)
%             if realhs(kk,jj) == 1
%                 plot([jj*scaling+spacescale*(kk-1) jj*scaling+spacescale*(length(workspaces)-1)],[1.006+0.0019-kk*0.0005 1.006+0.0019-kk*0.0005],'k-')
%             end
%         end
%     end
%     
%     
%     xlim([1.5 10])
%     xticks(xtickm)
%     xticklabels(xtickl)
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  find_figure(strcat('mean_response_Auc',cats5{allcat}));
%    subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
%     scaling = 2;
%     spacescale = 0.2;
%     xtickm = [];
%     xtickl = {};
% %     ylims= [0.985 1.02];
%     ptest = [];
%     color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
%     for jj = 1:4
%         for currmouse = 1:length(workspaces)
%             xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
%             %
%             for roin=1: size(comb_ac_vals{jj,currmouse},2)
%                 if roin==1
%                 scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
%                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.05)
%                 else
%                      scatter(ones(size(comb_ac_vals{jj,currmouse}(:,roin)))*jj*scaling+spacescale*(currmouse-1),...
%                     comb_ac_vals{jj,currmouse}(:,roin),20,color{jj}/2,'filled','Jitter','on', 'jitterAmount', 0.05)
%                 end
%                     
%             end
%             hold on
% %             scatter(jj*scaling+spacescale*(currmouse-1),mean_comb_vals{jj,currmouse},20,'k','s','LineWidth',2)
%             
% %              text(jj*scaling+spacescale*(currmouse-1)*ones(1,size(mean_comb_vals{jj,currmouse},2)),mean_comb_vals{jj,currmouse},comb_roi_vals{jj,currmouse})
% %             ylims = ylim;
%             xtickl = [xtickl, workspaces{currmouse}(1:4)];
% %                     if currmouse<length(workspaces)
% %                         [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(comb_vals{jj,currmouse}(1,:),comb_vals{jj,length(workspaces)}(1,:));
% %             
% %                     end
%             ylims= [0.985 1.025];
% %             set(gca,'ylim',ylims)
%         end
%         
%     end
    
    

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
                for d = 1:3
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_mut = [withoutSO_allmouse_mut;temp];
             
              temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:3
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
                for d = 1:size(allmouse_dop{1,p}{ll,currroi},1)
                    temp(d,:) = accumarray(currindex',allmouse_dop{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
                SO_allmouse_dop=[SO_allmouse_dop; temp];
                temp = [];
                for d = 1:size(allmouse_dop{1,p}{ll,currroi},1)
                    temp(d,:) = accumarray(currindex',allmouse_dop_alldays{1,p}{ll,currroi}(d,:)',[length(xax) 1],@mean);
                end
        SO_allmouse_dop_alldays=[SO_allmouse_dop_alldays; temp];

        
        SO_allmouse_dop_roe=[SO_allmouse_dop_roe; allmouse_roe{1,p}{ll,currroi}(d,:)];

            % for all the other planes
            temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:size(allmouse_dop{1,p}{ll,currroi},1)
                    gss = gss + 1;
                    temp(gss,:) = accumarray(currindex',allmouse_dop{1,p}{ll,withoutcurrroi(pss)}(d,:)',[length(xax) 1],@mean);
                end
            end
             withoutSO_allmouse_dop = [withoutSO_allmouse_dop;temp];
            
             temp = [];
            gss = 0;
            for pss = 1:length(withoutcurrroi)
                for d = 1:size(allmouse_dop{1,p}{ll,currroi},1)
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
%         subplot(2,2,((p-1)*2+2));
%                 yax=SO_allmouse_mut;
%         se_yax=nanstd(yax,1)./sqrt(size(yax,1));
%         yax=nanmean(yax);
%         hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
%         if sum(isnan(se_yax))~=length(se_yax)
%             h10.patch.FaceColor = color{4}; h10.mainLine.Color = color{4}; h10.edge(1).Color = color{4};
%             h10.edge(2).Color=color{4};
%         end
%         ylim(setylimmanual2);
%         ylims = ylim;
%         pls = plot([0 0],ylims,'--k','Linewidth',1);
%         ylim(ylims)
%         pls.Color(4) = 0.5
%         if p==1
%             ylabel('Early Days')
%         else
%             ylabel('Late Days')
%         end
%                yax=SO_allmouse_mut_roe;
%         plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
%         
%         %%%%%%%%%%%%%%% combine all the rest except SO
       
        
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
        
       
        
%         subplot(2,2,((p-1)*2+2));
%           yax=withoutSO_allmouse_mut;
%         se_yax=nanstd(yax,1)./sqrt(size(yax,1));
%         yax=nanmean(yax)
%         hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
%         if sum(isnan(se_yax))~=length(se_yax)
%             h10.patch.FaceColor = color{1}; h10.mainLine.Color = color{1}; h10.edge(1).Color = color{1};
%             h10.edge(2).Color=color{1};
%         end
%         ylim(setylimmanual2);
%         
%         
%         ylims = ylim;
%         pls = plot([0 0],ylims,'--k','Linewidth',1);
%         ylim(ylims)
%         pls.Color(4) = 0.5;
%         if p==1
%             ylabel('Early Days')
%             
%         else
%             ylabel('Late Days')
%             xlabel('Time onset')
%         end
%         title('GrabDA-mutant')
%         
%         
%         
%         yax=withoutSO_allmouse_mut_roe;
%         plot(roexax,nanmean(yax)/maxspeedlim*diff(roerescale)+roerescale(1),'k')
%         
        
        
        
        
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
        
        
        
        
%         yax=SO_allmouse_mut;
%         se_yax=nanstd(yax,1)./sqrt(size(yax,1));
%         yax=nanmean(yax);
%         hold on, ;h11 = shadedErrorBar(xax,yax,se_yax,[],1);
%         if sum(isnan(se_yax))~=length(se_yax)
%             h11.patch.FaceColor = color{end}/2; h11.mainLine.Color = color{end}/2; h11.edge(1).Color = color{end}/2;
%             h11.edge(2).Color=color{end}/2; h11.mainLine.LineStyle='-'; h11.edge(2).LineStyle='-';
%         end
        ylim(setylimmanual2);
%         legend([h11.edge(2) h10.edge(2) ], 'SO-GRABDA-mut','SO-GRABDA','onset','Location','northwest')
        
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
        
        
%         yax=withoutSO_allmouse_mut;
%         se_yax=nanstd(yax,1)./sqrt(size(yax,1));
%         yax=nanmean(yax);
%         h11 = shadedErrorBar(xax,yax',se_yax,[],1);
%         if sum(isnan(se_yax))~=length(se_yax)
%             h11.patch.FaceColor = color{1}/2; h11.mainLine.Color = color{1}/2; h11.edge(1).Color = color{1}/2;
%             h11.edge(2).Color=color{1}/2; h11.mainLine.LineStyle='-';h11.edge(2).LineStyle='-';
%         end
%         legend([h11.edge(2) h10.edge(2)], 'AllexceptSO-GRABDA-mut','AllexceptSO-GRABDA','onset','Location','northwest')
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
        
         grabda_rows=[1 :4];
    grabdamut_rows=[];
        
    pst=find(xax>timeforpre(1)&xax<=timeforpre(2)); pst=find(xax>timeforpost(1)&xax<=timeforpost(2));
    
%     pstE157 = find(xax>timeforpost(1)+0.5&xax<=timeforpost(2)+0.5);
    
%     mtm_vals_mut=[];
%     temp = [0 find(diff(notSO_allmouse_idmut)<=0) length(notSO_allmouse_idmut)]*4;%4=numdays
%     for n = 1:length(temp)-1
%         mtm_vals_mut(n) = mean(mean(withoutSO_allmouse_mut(temp(n)+1:temp(n+1),pst),2));
%     end
    
    mtm_vals_dop=[];
    temp = [0 find(diff(notSO_allmouse_iddop)<=0)*3 length(notSO_allmouse_iddop)*3-3]; %4=numdays
    for n = 1:length(temp)-1
        if n == 4
            mtm_vals_dop(n) = mean(mean(withoutSO_allmouse_dop(temp(n)+1:temp(n+1),pst),2));
        else
        mtm_vals_dop(n) = mean(mean(withoutSO_allmouse_dop(temp(n)+1:temp(n+1),pst),2));
        end
    end
    
    mtm_vals_SOdop = [];
    for n = 1:size(SO_allmouse_dop,1)/4
        if n == 4
            mtm_vals_SOdop(n) =mean(mean(SO_allmouse_dop(n*3-2:n*3-1,pst),2));
        else
    mtm_vals_SOdop(n) = mean(mean(SO_allmouse_dop(n*3-2:n*3,pst),2));
        end
    end
    
%     mtm_vals_SOmut = [];
%     for n = 1:size(SO_allmouse_mut,1)/4
%     mtm_vals_SOmut(n) = mean(mean(SO_allmouse_mut(n*4-3:n*4,pst),2));
%     end

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
    
    
    
%     subplot(4,2,(p-1)*4+4);
%     
%     hold on
%     scatter(ones(size(mtm_vals_mut))*2,mtm_vals_mut,25,[0.5 0.5 0.5],'filled','Jitter','on','Jitteramount',0.04)
%     
%  scatter(ones(size(mtm_vals_SOmut)),mtm_vals_SOmut,25,[1 0 0],'filled','Jitter','on','Jitteramount',0.04)
%     
%         [~,pl] = ttest(mtm_vals_SOmut-1);
%     text(1,nanmean(mtm_vals_SOmut),['p:' num2str(pl)])
%     
%     [~,pl] = ttest(mtm_vals_mut-1);
%     text(2,nanmean(mtm_vals_mut),['p:' num2str(pl)])
%     xlim([0.5 2.5])
%      plot(xlims,[1 1],'--','Color','k')
%      
%       
%     plot([1 2],[nanmean(mtm_vals_SOmut) nanmean(mtm_vals_mut)],'ks')
% %     errorbar([1 2],[nanmean(mtm_vals_SOmut) nanmean(mtm_vals_mut)],[nanstd(mtm_vals_SOmut) nanstd(mtm_vals_mut)]/sqrt(length(mtm_vals_mut)),[nanstd(mtm_vals_SOdop) nanstd(mtm_vals_dop)]/sqrt(length(mtm_vals_dop)),'k','Capsize',0,'LineStyle','none','LineWidth',0.5)
%     plot([1 1],nanmean(mtm_vals_SOmut)+ [nanstd(mtm_vals_SOmut) -nanstd(mtm_vals_SOmut)]/sqrt(length(mtm_vals_mut)),'k-','LineWidth',1.5)
%     plot([2 2],nanmean(mtm_vals_mut)+ [nanstd(mtm_vals_mut) -nanstd(mtm_vals_mut)]/sqrt(length(mtm_vals_mut)),'k-','LineWidth',1.5)
%    ylim([0.99 1.01])
%     yticks(0.99:0.005:1.01)
%     
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



%%%
%%
find_figure('ovrlay'); clf
find_figure('diff_plot');clf
clear  all_sig
for jj=1:size(df_f2,1)
     
    for kk=1:size(df_f2,2)
        find_figure('ovrlay'); 
        
        subplot(size(df_f2,2),size(df_f2,1),(kk-1)*length(workspaces)+jj)
        if kk==1
        title(workspaces{jj}(1:3))
        end
        for ff=1:2
           
            hold on
            
            
            if ff==1
               
                find_figure('ovrlay');
                yax=squeeze(df_f2(jj,kk,ff,:));
                h10 = shadedErrorBar(xax,squeeze(df_f2(jj,kk,ff,:)),squeeze(df_f2_se(jj,kk,ff,:)),[],0);
                color = {[0 0 1]/2,[0 1 0]/2,[204 164 61]/256/2,[231 84 128]/256/2};
                if sum(isnan(se_yax))~=length(se_yax)
                    h10.patch.FaceColor = color{kk}; h10.mainLine.Color = color{kk}; h10.edge(1).Color = color{kk};
                    h10.edge(2).Color=color{kk};
                    text(xt(kk),yt(kk),ROI_labels{currmouse}{kk},'Color',color{kk})
                end
xaxn=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;                
                plot(xaxn,squeeze(roe_trace2(jj,1,ff,:)),'k')
            else
                find_figure('ovrlay');
                color = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
                yax=squeeze(df_f2(jj,kk,ff,:));
                h10 = shadedErrorBar(xax,squeeze(df_f2(jj,kk,ff,:)),squeeze(df_f2_se(jj,kk,ff,:)),[],1);
                if sum(isnan(se_yax))~=length(se_yax)
                    h10.patch.FaceColor = color{kk}; h10.mainLine.Color = color{kk}; h10.edge(1).Color = color{kk};
                    h10.edge(2).Color=color{kk};
                     h10.patch.FaceAlpha = 0.07;
                h10.mainLine.LineWidth = 1.5;
                h10.edge(1).Color(4) = 0.07;
                h10.edge(2).Color(4) = 0.07;
                    text(xt(kk),yt(kk),ROI_labels{currmouse}{kk},'Color',color{kk})
                end
                   plot(xaxn,squeeze(roe_trace2(jj,1,ff,:)),'--k')
                %%%%%%%%compute moving avg significance
                mov_win=0.5; %1sec
                mov_win_frames=round(mov_win/frame_time);
                win_sig=[40:79];
                for ll=1:length(win_sig)-mov_win_frames
                    win= ll:mov_win_frames+ll;
                    win_find=win_sig(win);
                    var1=nanmean(squeeze(df_f2all{jj,1}(:,kk,win_find)),2);
                    var2=nanmean(squeeze(df_f2all{jj,2}(:,kk,win_find)),2);
                    
                
                    vals_1=cellfun(@(x) nanmean(x(win_find,:),1) ,df_f2_CS{jj,1} ,'UniformOutput' ,false);
                    var1=cat(2,vals_1{:,kk})';
                     vals_2=cellfun(@(x) nanmean(x(win_find,:),1) ,df_f2_CS{jj,2} ,'UniformOutput' ,false);
                    var2=cat(2,vals_2{:,kk})';
                    [h,p]=ttest2(var1,var2);
                    sig_all(1,ll)=p;
                    sig_all(2,ll)=h;
                    if p==1
                        x=[xax(win_find(1)) xax(win_find(end)) xax(win_find(end)) xax(win_find(1))];
                        y=[0.985 0.985 1.02 1.02];
                        patch(x,y,'red','FaceAlpha',.3,'EdgeColor','none')
                        
                    end
                    
                    
                end
              
%                 all_sig(jj,kk,:)=sig_all;
                
                
                
                
                
               
                yax_diff=squeeze(df_f2(jj,kk,2,:))-squeeze(df_f2(jj,kk,1,:));
                yax_diff_roe=squeeze(roe_trace2(jj,1,2,:))-squeeze(roe_trace2(jj,1,1,:));
                find_figure('diff_plot');
                subplot(size(df_f2,2),size(df_f2,1),(kk-1)*length(workspaces)+jj), plot(xax,yax_diff','Color',color{kk},'Linewidth',2);
                hold on
                plot([-5 5],[0 0],'--k')
                plot(xaxn,yax_diff_roe','--k','Linewidth',2);
                if kk==1
                    title(workspaces{jj}(1:3))
                end
                ylims = ylim;
                pls = plot([0 0],ylims,'--k','Linewidth',1);
                ylim(ylims)
                pls.Color(4) = 0.5;
                text(xt(kk),yt(kk),ROI_labels{currmouse}{kk},'Color',color{kk})
                xlabel('time from US')
                if kk==4
                    legend({' df/f',' Speed'}, 'Location','southwest')
                end
            end
            
            
        end
        find_figure('ovrlay');
  
        if kk<4
            setylimmanual=[0.985 1.01]
            ylim(setylimmanual);
            ylims = ylim;
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            ylim(ylims)
            pls.Color(4) = 0.5;
%             legend('Rewarded US WITHOUT CS','Rewarded US WITH CS')
        else
            setylimmanual=[0.985 1.07];
            ylim(setylimmanual);
            ylims = ylim;
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            ylim(ylims)
            pls.Color(4) = 0.5;
            xlabel('time from US')
            legend('US with CS','US without CS')
        end
    end
end

%%%
clear vals vals_1 var_f
for jj=1:size(df_f2,1) %% currmouse
    
    for kk=1:size(df_f2,2) %%% roi
        for ll=1:2 %%% 1- unrewarded 2-rewarded
            win_sc=[40 42; 44 52; 56 72];
           
            for mm=1:3 %%% 1-40-48 frames (1s) 2-56 -72 frames(2s)
               vals(jj,kk,ll,mm)=nanmean(nanmean(squeeze(df_f2all{jj,ll}(:,kk,win_sc(mm,1):win_sc(mm,2))),2));
               win_find=win_sc(mm,1):win_sc(mm,2);
               if ll==1
                 vals_1=cellfun(@(x) nanmean(x(win_find,:),1) ,df_f2_CS{jj,1} ,'UniformOutput' ,false);
                    var_f{jj,kk}{ll,mm}=cat(2,vals_1{:,kk})';
               else
                   vals_1=cellfun(@(x) nanmean(x(win_find,:),1) ,df_f2_CS{jj,2} ,'UniformOutput' ,false);
                    var_f{jj,kk}{ll,mm}=cat(2,vals_1{:,kk})';
               end
                   
            end
        end
    end
end
%%
 find_figure('population summary');clf
 m_anz=4;
idz1=[];
    for kk=1:3
        mean_val=[]; p_sig=[]; se_val=[];
        for ll=1:4
            idz=(ll-1)*3+1;
            find_figure('population summary')
            subplot(3,1,kk),plot(idz*ones(1,m_anz),squeeze(vals(1:m_anz,ll,1,kk)),'o','Color',color{ll})
            hold on
            plot(idz+1*ones(1,m_anz),squeeze(vals(1:m_anz,ll,2,kk)),'o','Color',color{ll})
            idz1=[idz1 idz idz+1 NaN];
            mean_val=[mean_val nanmean(squeeze(vals(1:m_anz,ll,1,kk))) nanmean(squeeze(vals(1:m_anz,ll,2,kk))) NaN];
            se_val=[se_val std(squeeze(vals(1:m_anz,ll,1,kk)))/length(squeeze(vals(1:m_anz,ll,1,kk))) std(squeeze(vals(1:m_anz,ll,2,kk)))/length(squeeze(vals(1:m_anz,ll,2,kk))) NaN];
            [h,p]=ttest(squeeze(vals(1:m_anz,ll,1,kk)),squeeze(vals(1:m_anz,ll,2,kk)))
            p_sig=[p_sig NaN p NaN];
        end
        set(gca,'xlim',[0 12])
        if kk==1
            title('0.2s after US')
        elseif kk==2
            title('between 0.5s-1.5s after US')
        else
            title('between 2s-4s after US')
        end
        hold on; errorbar(mean_val,se_val);
        text((1:length(mean_val))-0.5, 1.04*ones(1,length(mean_val)),string(p_sig))
    end
    
  %%%
  
  %%
 find_figure('population summary');clf
idz1=[];

    for kk=1:3
        mean_val=[]; p_sig=[]; se_val=[];
        for ll=1:4
            idz=(ll-1)*3+1;
            find_figure('population summary')  
            chk1=cat(1,var_f{1:m_anz,ll});
            vals_r=cat(1,chk1{2:2:size(chk1,1),kk});
            vals_ur=cat(1,chk1{1:2:size(chk1,1),kk});
            
            subplot(3,1,kk),plot(idz*ones(1,length(vals_r)),vals_r,'o','Color',color{ll})
            hold on
            plot(idz+1*ones(1,length(vals_ur)),vals_ur,'o','Color',color{ll})
            idz1=[idz1 idz idz+1 NaN];
            mean_val=[mean_val nanmean(vals_r) nanmean(vals_ur) NaN];
            se_val=[se_val std(vals_r)/length(vals_r) std(vals_ur)/length(vals_ur) NaN];
            [h,p]=ttest2(vals_r,vals_ur)
            p_sig=[p_sig NaN p NaN];
            text(idz,0.96,string(length(vals_r)));
            text(idz+1,0.96,string(length(vals_ur)));
            names={'USwithCS', 'USwithoutCS'};
            hold on
           set(gca,'xtick',[idz idz+1],'xticklabel',names)
        end
        set(gca,'xlim',[0 12])
        if kk==1
            title('0.2s after US')
        elseif kk==2
            title('between 0.5s-1.5s after US')
        else
            title('between 2s-4s after US')
        end
        hold on; errorbar(mean_val,se_val,'--k','Linewidth',2);
        
        
        plot([0 12],[1 1],'k','Linewidth',0.5)
        text((1:length(mean_val))-0.5, 1.04*ones(1,length(mean_val)),string(p_sig))
    end

     
    
      %% just make the comparison the way ed wants it
    
    %SO
    
    y1 = [];
    y2 = [];
    
    for mus = 1:size(df_f2_CS,1)
        y1 = [y1 nanmean(nanmean(cellfun(@(x) nanmean(nanmean(x(41:43,:),1)),df_f2_CS{mus,1}(:,4),'UniformOutput',1)))];
        y2 = [y2 nanmean(nanmean(cellfun(@(x) nanmean(nanmean(x(41:43,:),1)),df_f2_CS{mus,2}(:,4),'UniformOutput',1)))];

    end
    figure;
    comparestatsplotspair(y2,y1,'r',[0.8 0.4 0],'k',[1 2])
    
    %except SO
     y1 = [];
    y2 = [];
    
    for mus = 1:size(df_f2_CS,1)
        y1 = [y1 nanmean(nanmean(cellfun(@(x) nanmean(nanmean(x(41:43,:),1)),df_f2_CS{mus,1}(:,1:3),'UniformOutput',1)))];
        y2 = [y2 nanmean(nanmean(cellfun(@(x) nanmean(nanmean(x(41:43,:),1)),df_f2_CS{mus,2}(:,1:3),'UniformOutput',1)))];

    end
    
    comparestatsplotspair(y2,y1,'b',[0.1 0.5 0.5],'k',[4 5])
    
    xlim([0.5 5.5])
    xticks([1.5 4.5])
    xticklabels({'SO','All Except SO'})
    chils = get(gca,'Children');
    legend(chils([22 21 11 10]),{'Queued US','UnQueued US','Queued US','UnQueued US'})
    
    
    
    