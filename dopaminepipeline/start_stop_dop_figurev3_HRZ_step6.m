%% ROI version
workspace = 'T:\E158\158_HRZ_workspace_00.mat';
mouseId = '158';
load(workspace)
close all
savepath = 'T:\E158\HRZfigures';
saving = 1;

%for 156 HRZ
% files = [7 8 13 14 17:20 22];
% for 157 HRZ

%for 158 HRZ
% files = [4 5 43 44];

% find_figure('dop_days_loc_planes');clf
if ~exist('files')
    files = 1:length(day_labels)
% files = 5:length(day_labels);
end

if length(day_labels)~= length(files)
    day_labels = day_labels(files);
end

% dosedays = 1:2:length(day_labels);
dosedays = [];

addition = 0;
earlydays = 1 + addition;
latedays = length(files) + addition;

%149 dark Rewards labels
% ROI_labels = {'Plane 1 SR','Plane 1 SP','Plane 2 SP','Plane 2 SP/SO','Plane 3 SO','Plane 4 SO'};
%149 etic roi 1
% ROI_labels = {'Plane 1 SR','Plane 2 SR','Plane 2 SR/SP','Plane 3 SP','Plane 4 SO/SP'};
%149 etic roi 2
% ROI_labels = {'Plane 1 SR','Plane 2 SR','Plane 2 SP','Plane 3 SP','Plane 3 SR','Plane 4 SO_SP'}; %ROI 2

%158 old rois
% ROI_labels = {'Plane 1 SP','Plane 2 SO', 'Plane 3 SO','Plane 4 SO'};
%158 new rois
ROI_labels = {'Plane 1 SP','Plane 1 SR','Plane 2 SO','Plane 2 SP', 'Plane 3 SO','Plane 4 SO'};

% %E149 stopping
% yax1 = [0.989 1.005];
% yax2 = [0.99 1.02];
% cax1 = [0.994 1.008];
% cax2 = [0.992 1.01];
% speedax = [-5 45];
% speedcax = [-15 35];

% % E149 perireward
% yax1 = [0.996 1.005];
% yax2 = [0.995 1.005];
% cax1 = [0.996 1.008];
% cax2 = [0.995 1.01];
% speedax = [-5 40];
% speedcax = [-15 40];

% % dont care right now
% yax1 = 'auto';
% yax2 = 'auto';
% cax1 = 'auto';
% cax2 = 'auto';
% speedax = 'auto';
% speedcax = 'auto';

% % E149 Starting
% yax1 = [0.995 1.003];
% yax2 = [0.995 1.007];
% cax1 = [0.996 1.005];
% cax2 = [0.993 1.007];
% speedax = [-1 65];
% speedcax = [-15 55];

% % E158 stopping
% yax1 = [0.996 1.006];
% yax2 = [0.993 1.01];
% cax1 = [0.997 1.005];
% cax2 = [0.995 1.006];
% speedax = [-1 11];
% speedcax = [-3 10];

% % E158 perireward
% yax1 = [0.996 1.007];
% yax2 = [0.993 1.011];
% cax1 = [0.996 1.007];
% cax2 = [0.993 1.017];
% speedax = [-5 20];
% speedcax = [-5 12];

% % E158 starting
% yax1 = [0.995 1.006];
% yax2 = [0.993 1.01];
% cax1 = [0.995 1.006];
% cax2 = [0.995 1.01];
% speedax = [-1 25];
% speedcax = [-9 15];

%E158 HRZ peri reward
% yax1 = [0.9978 1.0025];
% yax2 = [0.993 1.006];
% cax1 = [0.998 1.004];
% cax2 = [0.993 1.006];
% speedax = [-5 50];
% speedcax = [-20 50];


%E158 HRZ stop
% yax1 = [0.9978 1.002];
% yax2 = [0.995 1.005];
% cax1 = [0.998 1.002];
% cax2 = [0.995 1.005];
% speedax = [-5 40];
% speedcax = [-10 40];

%E158 HRZ start
% yax1 = [0.997 1.006];
% yax2 = [0.996 1.013];
% cax1 = [0.997 1.004];
% cax2 = [0.996 1.01];
% speedax = [-5 50];
% speedcax = [-20 60];

% % E157 Starting
% yax1 = [0.99 1.0041];
% yax2 = [0.992 1.025];
% cax1 = [0.996 1.006];
% cax2 = [0.995 1.014];
% speedax = [-1 30];
% speedcax = [-9 25];


% % E157 stopping
% yax1 = [0.991 1.0065];
% yax2 = [0.975 1.04];
% cax1 = [0.996 1.007];
% cax2 = [0.992 1.02];
% speedax = [-1 25];
% speedcax = [-9 25];

% % E157 perireward
% yax1 = [0.985 1.01];
% yax2 = [0.975 1.03];
% cax1 = [0.996 1.015];
% cax2 = [0.98 1.03];
% speedax = [-1 25];
% speedcax = [-9 25];

%E157 HRZ starting
% yax1 = [0.986 1.01];
% yax2 = [0.992 1.01];
% cax1 = [0.996 1.007];
% cax2 = [0.992 1.02];
% speedax = [-5 75];
% speedcax = [-20 75];

%E157 HRZ stopping
% yax1 = [0.99 1.01];
% yax2 = [0.99 1.007];
% cax1 = [0.996 1.01];
% cax2 = [0.99 1.01];
% speedax = [-5 60];
% speedcax = [-20 65];

%E157 HRZ perireward
% yax1 = [0.99 1.01];
% yax2 = [0.99 1.007];
% cax1 = [0.996 1.009];
% cax2 = [0.99 1.02];
% speedax = [-5 60];
% speedcax = [-20 75];

% % E156 Starting
% yax1 = [0.994 1.0025];
% yax2 = [0.995 1.02];
% cax1 = [0.995 1.005];
% cax2 = [0.995 1.014];
% speedax = [-1 25];
% speedcax = [-9 25];

% % E156 Perireward
% yax1 = [0.994 1.003];
% yax2 = [0.995 1.011];
% cax1 = [0.997 1.003];
% cax2 = [0.995 1.015];
% speedax = [-1 20];
% speedcax = [-15 30];

%E156 HRZ Perireward
% yax1 = [0.994 1.01];
% yax2 = [0.995 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.995 1.005];
% speedax = [-1 40];
% speedcax = [-15 40];

%E156 HRZ stop
% yax1 = [0.994 1.01];
% yax2 = [0.995 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.995 1.005];
% speedax = [-1 40];
% speedcax = [-15 40];

%E156 HRZ start
% yax1 = [0.995 1.005];
% yax2 = [0.997 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.997 1.005];
% speedax = [-1 55];
% speedcax = [-15 60];

% % E148 Starting
% yax1 = [0.996 1.006];
% yax2 = [0.991 1.01];
% cax1 = [0.998 1.007];
% cax2 = [0.993 1.014];
% speedax = [-1 15];
% speedcax = [-9 20];

% % E148 Stopping
% yax1 = [0.996 1.004];
% yax2 = [0.991 1.006];
% cax1 = [0.9985 1.003];
% cax2 = [0.995 1.006];
% speedax = [-1 10];
% speedcax = [-5 12];

% % E148 perireward
% yax1 = [0.992 1.005];
% yax2 = [0.995 1.03];
% cax1 = [0.996 1.01];
% cax2 = [0.996 1.03];
% speedax = [-5 9];
% speedcax = [-5 9];

% cax1 = [0.995 1.005];
% cax2 = cax1;
speedcax = [-5 40];
% yax1 = [0.99 1.01];
% yax2 = yax1;

% mouse = 'E148';

planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
roiplaneidx = cellfun(@(x) str2num(x(7)),ROI_labels,'UniformOutput',1);
color = planecolors(roiplaneidx);

for checkforemptyfiles = files
    if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
        roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        
        roi_dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,1} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,2} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,3} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,4} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));

    end
end

%     roevariable = roe_allsuc_mov;
% roeindividualvariable = roe_alldays_planes_success_mov;
% dopvariable = dop_allsuc_mov;
% dopindividualvariable = dop_alldays_planes_success_mov;
roevariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_0,'UniformOutput',0);
roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roe_alldays_planes_perireward_0,'UniformOutput',0));


dopvariable = roi_dop_allsuc_perireward;


dopindividualvariable =  cellfun(@(x) transpose(x),roi_dop_alldays_planes_perireward,'UniformOutput',0);
roeindividualvariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_0,'UniformOutput',0);

roevariables{1} = roevariable; %perisingle
dopvariables{1} = dopvariable; %perisingle
roeindividualvariables{1} = roeindividualvariable; %perisingle
dopindividualvariables{1} = dopindividualvariable;% perisingle
variablelable{1} = 'Peri Single Reward'; %perisingle


roevariable = cellfun(@(x) transpose(x),roe_dop_alldays_planes_first3,'UniformOutput',0);
roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roe_dop_alldays_planes_first3,'UniformOutput',0));
roeindividualvariable = cellfun(@(x) transpose(x),roe_dop_alldays_planes_first3,'UniformOutput',0);

dopvariable = roi_dop_allsuc_first3;

dopindividualvariable =  cellfun(@(x) transpose(x),roi_dop_alldays_planes_first3,'UniformOutput',0);


roevariables{2} = roevariable; %peridouble
dopvariables{2} = dopvariable; %peridouble
roeindividualvariables{2} = roeindividualvariable; %peridouble
dopindividualvariables{2} = dopindividualvariable;% peridouble
variablelable{2} = 'Peri First 3 Rewards'; %peridouble

roevariables{3} = roe_allsuc_mov;
roeindividualvariables{3} = roe_alldays_planes_success_mov;
dopvariables{3} = roi_dop_allsuc_mov;
dopindividualvariables{3} = roi_dop_alldays_planes_success_mov;
variablelable{3} = 'Peri Start Triggered';

roevariables{4} = roe_allsuc_stop;
roeindividualvariables{4} = roe_alldays_planes_success_stop;
dopvariables{4} = roi_dop_allsuc_stop;
dopindividualvariables{4} = roi_dop_alldays_planes_success_stop;
variablelable{4} = 'Peri Stop Triggered';

roevariables{5} = roe_allsuc_stop_no_reward;
roeindividualvariables{5} = roe_alldays_planes_success_stop_no_reward;
dopvariables{5} = roi_dop_allsuc_stop_no_reward;
dopindividualvariables{5} = roi_dop_alldays_planes_success_stop_no_reward;
variablelable{5} = 'Peri Unrewarded Stop Triggered';

roevariables{6} = roe_allsuc_stop_reward;
roeindividualvariables{6} = roe_alldays_planes_success_stop_reward;
dopvariables{6} = roi_dop_allsuc_stop_reward;
dopindividualvariables{6} = roi_dop_alldays_planes_success_stop_reward;
variablelable{6} = 'Peri Rewarded Stop Triggered';

roevariables{7} = roi_roe_allsuc_perirewardCS;
roeindividualvariables{7} = cellfun(@(x) transpose(x),roi_roe_alldays_planes_periCS,'UniformOutput',0);
dopvariables{7} = roi_dop_allsuc_perirewardCS;
dopindividualvariables{7} = cellfun(@(x) transpose(x),roi_dop_alldays_planes_periCS,'UniformOutput',0);
variablelable{7} = 'Peri Conditioned Stimulus';

roevariable = cellfun(@(x) transpose(x),roe_dop_alldays_planes_last3,'UniformOutput',0);
roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roe_dop_alldays_planes_last3,'UniformOutput',0));
roeindividualvariable = cellfun(@(x) transpose(x),roe_dop_alldays_planes_last3,'UniformOutput',0);

dopvariable = roi_dop_allsuc_last3;

dopindividualvariable =  cellfun(@(x) transpose(x),roi_dop_alldays_planes_last3,'UniformOutput',0);


roevariables{8} = roevariable; %peridouble
dopvariables{8} = dopvariable; %peridouble
roeindividualvariables{8} = roeindividualvariable; %peridouble
dopindividualvariables{8} = dopindividualvariable;% peridouble
variablelable{8} = 'Peri Last 3 Rewards of Epoch 1'; %peridouble

numROIs = length(ROI_labels);

for varsi = 1:length(roevariables)
    roevariable = roevariables{varsi};
    roeindividualvariable = roeindividualvariables{varsi};
    dopvariable = dopvariables{varsi};
    dopindividualvariable = dopindividualvariables{varsi};
    bottom = min(min(min(dopvariable(dopvariable>0))));
    top = max(max(max(dopvariable(dopvariable>0))));
    maxdif = max([abs(1-bottom) abs(1-top)]);
    cax1 = [1-maxdif  1+maxdif];
    cax2 = cax1;
    yax1 = cax1;
    yax2 = cax1;
figure;
for jj=1:numROIs
%     jj = ll;
    ll = numROIs-jj+1;
    %speed
%     find_figure('dop_days_loc_planes')
    ax(jj)=subplot(numROIs+1,3,1),imagesc(squeeze(roevariable(files,1,:))); hold on
    plot([40 40],[0 length(files)+0.5],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
    if exist('speedcax')
    caxis(speedcax)
    end
    if exist('dosedays')
        yticks(dosedays)
        yticklabels(day_labels(dosedays))
    end
    
%     text(10, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
%     text(62, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
    ylabel('Days')

    ax(jj)=subplot(numROIs+1,3,2);
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
    if length(earlydays) == 1
        if size(roeindividualvariable{earlydays,1},1) > 1
        yax=mean(squeeze(roeindividualvariable{earlydays,1}),1);
        se_yax=std(squeeze(roeindividualvariable{earlydays,1}),0,1)./sqrt(size(squeeze(roeindividualvariable{earlydays,1}),1));
        else
            yax = mean(squeeze(roeindividualvariable{earlydays,1}),1);
            se_yax = zeros(size(yax));
        end
    else
        yax=mean(squeeze(roevariable(earlydays,1,:)),1);
        se_yax=std(squeeze(roevariable(earlydays,1,:)),0,1)./sqrt(size(squeeze(roevariable(earlydays,1,:)),1));
    end
    hold on, shadedErrorBar(xax,yax,se_yax,'k');
    if exist('speedax')
    ylim(speedax)
    end
    text('Units','normalized','Position',[0.99 0.99],'String',['n = ' num2str(size(squeeze(roeindividualvariable{earlydays,1}),1))])
    xlim([-5 5])
    ylims = ylim;
    plot([0 0],ylims,'k','Linewidth',2)
    clear xlabel
    if length(earlydays) == 1
    title(day_labels{earlydays-addition},'Interpreter','none')
    else
        title(['Days ' num2str(earlydays(1)) ':' num2str(earlydays(end))])
    end
    
    ax(jj)=subplot(numROIs+1,3,3);
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        
        if length(latedays) == 1

            if size(roeindividualvariable{latedays,1},1) > 1
                yax=mean(squeeze(roeindividualvariable{latedays,1}),1);
                se_yax=std(squeeze(roeindividualvariable{latedays,1}),0,1)./sqrt(size(squeeze(roeindividualvariable{latedays,1}),1));
            else
                yax = mean(squeeze(roeindividualvariable{latedays,1}),1);
                se_yax = zeros(size(yax));
            end
        else
            yax=mean(squeeze(roevariable(latedays,1,:)),1);
            se_yax=std(squeeze(roevariable(latedays,1,:)),0,1)./sqrt(size(squeeze(roevariable(latedays,1,:)),1));
        end
       hold on, 
       shadedErrorBar(xax,yax,se_yax,'k');
    if exist('speedax')
        ylim(speedax)
    end
        ylims = ylim;
    plot([0 0],ylims,'k','Linewidth',2)
    text('Units','normalized','Position',[0.99 0.99],'String',['n = ' num2str(size(squeeze(roeindividualvariable{latedays,1}),1))])
    clear xlabel
       if length(earlydays) == 1
    title(day_labels{latedays-addition},'Interpreter','none')
    else
        title(['Days ' num2str(latedays(1)) ':' num2str(latedays(end))])
    end
    
    %
    
    %image of all days dFF
    ax(jj)=subplot(numROIs+1,3,ll*3+1),imagesc(squeeze(dopvariable(files,jj,:))); hold on
    if jj>3
        if exist('cax2')
      caxis(cax2)
        end
    else
        if exist('cax1')
        caxis(cax1)
        end
    end
    if exist('dosedays')
        yticks(dosedays)
        yticklabels(day_labels(dosedays))
    end
    plot([40 40],[0 length(files)+0.5],'Linewidth',2); colormap(ax(jj),fake_parula);
%     title(strcat('success_Plane',num2str(jj),'__Start Triggered'));set(gca,'xtick',[])
    title(ROI_labels{jj},'Interpreter','none'); set(gca,'xtick',[])
    ylabel('Days')
    % early
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        if length(earlydays) == 1
            if size(dopindividualvariable{earlydays,jj},1)>1
            yax=mean(squeeze(dopindividualvariable{earlydays,jj}),1);
            se_yax=std(squeeze(dopindividualvariable{earlydays,jj}),0,1)./sqrt(size(squeeze(dopindividualvariable{earlydays,jj}),1));
            else
                yax=mean(squeeze(dopindividualvariable{earlydays,jj}),1);
                se_yax = zeros(size(yax));
            end
            
        else
            yax=mean(squeeze(dopvariable(earlydays,jj,:)),1);
            se_yax=std(squeeze(dopvariable(earlydays,jj,:)),0,1)./sqrt(size(squeeze(dopvariable(earlydays,jj,:)),1));
        end
    axe(jj) = subplot(numROIs+1,3,ll*3+2); hold on,h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj};
    if jj>numROIs-1
        if exist('yax2')
        ylim(yax2)
          plot([0 0],yax2,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
          clear xlabel
    else
        if exist('yax1')
        ylim(yax1)
          plot([0 0],yax1,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
              clear xlabel; xlabel('Time(s)');
    end
    %late
    
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        if length(latedays) == 1
            if size(squeeze(dopindividualvariable{latedays,jj}),1) >1
            yax=mean(squeeze(dopindividualvariable{latedays,jj}),1);
            se_yax=std(squeeze(dopindividualvariable{latedays,jj}),0,1)./sqrt(size(squeeze(dopindividualvariable{latedays,jj}),1));
            else
                yax=mean(squeeze(dopindividualvariable{latedays,jj}),1);
                se_yax = zeros(size(yax));
            end
        else
            yax=mean(squeeze(dopvariable(latedays,jj,:)),1);
            se_yax=std(squeeze(dopvariable(latedays,jj,:)),0,1)./sqrt(size(squeeze(dopvariable(latedays,jj,:)),1));
        end
    axl(jj) = subplot(numROIs+1,3,ll*3+3),hold on, h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj};
    if jj>numROIs-1
        if exist('yax2')
        ylim(yax2)
        plot([0 0],yax2,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
        clear xlabel
    else
        if exist('yax1')
        ylim(yax1)
        plot([0 0],yax1,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
        clear xlabel; xlabel('Time(s)');
    end
    
end
mtit([mouseId ' ' variablelable{varsi}])
linkaxes([axe axl],'y')
if saving
%     stripROIlabels = cellfun(@(x) )
     set(gcf,'units','normalized','outerposition',[0 0 1 1])
    saveas(gcf,[savepath '/' mouseId '_' variablelable{varsi} '_full_figure.svg'],'svg')
    hAx = findobj('type', 'axes');
    subtitles = {'_Late_Day','_Early_Day','AllDay_colorplot'};
    for iAx = 1:length(hAx)-1
        panel = mod(iAx,3);
        if panel == 0
            panel = 3;
        end
        if (numROIs - ceil(iAx/3)+1)>0
        planeidx = numROIs - ceil(iAx/3)+1;
        
        axtitle = [mouseId '_' ROI_labels{planeidx} subtitles{panel}];
        else
            axtitle = [mouseId '_Speed' subtitles{panel} ];
        end

        fNew = figure('units','normalized','outerposition',[0 0 1 1]);
        hNew = copyobj(hAx(iAx), fNew);
        set(hNew, 'pos', [0.23162 0.2233 0.72058 0.63107])
        set(gca,'fontsize', 18)
        set(gca,'FontName','Arial')
%         InSet = get(hNew, 'TightInset');
        %         set(hNew, 'Position', [InSet(1:2)+0.05, 1-InSet(1)-InSet(3)-0.1, 1-InSet(2)-InSet(4)-0.05])
        if ~exist([savepath '/' variablelable{varsi}])
            mkdir([savepath '/' variablelable{varsi}])
        end
        set(gca,'units','normalized','outerposition',[0.01 0.1 0.95 0.88])
        %         set(gcf, 'Renderer', 'Painters');

        saveas(fNew,[savepath '/' variablelable{varsi} '/' axtitle '.svg'],'svg')
    end
    close all
end
end
% 
%  if saving
%      set(gcf,'units','normalized','outerposition',[0 0 1 1])
%     saveas(gcf,[savepath 'full_figure.svg'],'svg')
% 
%     planesubplotnames = fliplr({'AlldayImagesc',['Day_' num2str(earlydays)],['Day_' num2str(latedays)]});
%     
%     hAx = findobj('type', 'axes');
%     for iAx = 1:length(hAx)
%         planenumber = 4-ceil(iAx/3)+1;
%         if rem(iAx,3) == 0
%             plotind = 3;
%         else
%             plotind = rem(iAx,3);
%         end
%         if iAx < 13
%         axtitle = ['Plane' num2str(planenumber) '_' planesubplotnames(plotind)];
%         else
%             axtitle = ['Speed_' planesubplotnames(plotind)];
%         end
%         if iscell(axtitle)
%         axtitle = cell2mat(axtitle);
%         end
%         fNew = figure('units','normalized','outerposition',[0 0 1 1]);
%         hNew = copyobj(hAx(iAx), fNew);
% %         title(axtitle)
%         % Change the axes position so it fills whole figure
% %         set(hNew, 'pos', [0.23162 0.2233 0.72058 0.63107])
%         set(gca,'fontsize', 18)
%         set(gca,'FontName','Arial')
%         InSet = get(hNew, 'TightInset');
% %         set(hNew, 'Position', [InSet(1:2)+0.05, 1-InSet(1)-InSet(3)-0.1, 1-InSet(2)-InSet(4)-0.05])
%         set(gca,'units','normalized','outerposition',[0.01 0.1 0.95 0.88])
% %         set(gcf, 'Renderer', 'Painters');
%         
%         saveas(fNew,[savepath axtitle '.svg'],'svg')
% %         export_fig(['D:\munneworkspaces\darkrewardfigures\StartTriggered\E148\E148_' axtitle '.svg'])
%     end
%     
%     figure; colorbar; colormap(fake_parula); caxis(cax1)
%     saveas(gcf,[savepath 'Planes1-3colorbar.svg'],'svg')
%     figure; colorbar; colormap(fake_parula); caxis(cax2)
%     saveas(gcf,[savepath 'Plane4colorbar.svg'],'svg')
%     figure; colorbar; colormap(gray); caxis(speedcax)
%     saveas(gcf,[savepath 'Speedcolorbar.svg'],'svg')

%  end

 %% ROI version Single Panels for Summary
workspace = 'K:\149_Eticlo_workspace.mat';
mouseId = '149';
load(workspace)
close all


savepath = 'K:\zSingleRewardExampleDays\';
saving = 1;
color={[0 0 1],[0 1 0],[0 1 0],[204 164 61]/256,[204 164 61]/256,[231 84 128]/256};
%for 156 HRZ
% files = [7 8 13 14 17:20 22];
% for 157 HRZ

%for 158 HRZ
% files = [4 5 43 44];

if ~exist('files')
%     files = 1:length(day_labels)
files = 5:length(day_labels);
end


dosedays = 1:2:length(day_labels);

earlydays = 6;
latedays = 7;

%149 dark Rewards labels
% ROI_labels = {'Plane 1 SR','Plane 1 SP','Plane 2 SP','Plane 2 SP/SO','Plane 3 SO','Plane 4 SO'};
%149 etic roi 1
% ROI_labels = {'Plane 1 SR','Plane 2 SR','Plane 2 SR/SP','Plane 3 SP','Plane 4 SO/SP'};
%149 etic roi 2
ROI_labels = {'Plane 1 SR','Plane 2 SR','Plane 2 SP','Plane 3 SP','Plane 3 SR','Plane 4 SO/SP'}; %ROI 2

% %E149 stopping
% yax1 = [0.989 1.005];
% yax2 = [0.99 1.02];
% cax1 = [0.994 1.008];
% cax2 = [0.992 1.01];
% speedax = [-5 45];
% speedcax = [-15 35];

% % E149 perireward
% yax1 = [0.996 1.005];
% yax2 = [0.995 1.005];
% cax1 = [0.996 1.008];
% cax2 = [0.995 1.01];
% speedax = [-5 40];
% speedcax = [-15 40];

% % dont care right now
% yax1 = 'auto';
% yax2 = 'auto';
% cax1 = 'auto';
% cax2 = 'auto';
% speedax = 'auto';
% speedcax = 'auto';

% % E149 Starting
% yax1 = [0.995 1.003];
% yax2 = [0.995 1.007];
% cax1 = [0.996 1.005];
% cax2 = [0.993 1.007];
% speedax = [-1 65];
% speedcax = [-15 55];

% % E158 stopping
% yax1 = [0.996 1.006];
% yax2 = [0.993 1.01];
% cax1 = [0.997 1.005];
% cax2 = [0.995 1.006];
% speedax = [-1 11];
% speedcax = [-3 10];

% % E158 perireward
% yax1 = [0.996 1.007];
% yax2 = [0.993 1.011];
% cax1 = [0.996 1.007];
% cax2 = [0.993 1.017];
% speedax = [-5 20];
% speedcax = [-5 12];

% % E158 starting
% yax1 = [0.995 1.006];
% yax2 = [0.993 1.01];
% cax1 = [0.995 1.006];
% cax2 = [0.995 1.01];
% speedax = [-1 25];
% speedcax = [-9 15];

%E158 HRZ peri reward
% yax1 = [0.9978 1.0025];
% yax2 = [0.993 1.006];
% cax1 = [0.998 1.004];
% cax2 = [0.993 1.006];
% speedax = [-5 50];
% speedcax = [-20 50];


%E158 HRZ stop
% yax1 = [0.9978 1.002];
% yax2 = [0.995 1.005];
% cax1 = [0.998 1.002];
% cax2 = [0.995 1.005];
% speedax = [-5 40];
% speedcax = [-10 40];

%E158 HRZ start
% yax1 = [0.997 1.006];
% yax2 = [0.996 1.013];
% cax1 = [0.997 1.004];
% cax2 = [0.996 1.01];
% speedax = [-5 50];
% speedcax = [-20 60];

% % E157 Starting
% yax1 = [0.99 1.0041];
% yax2 = [0.992 1.025];
% cax1 = [0.996 1.006];
% cax2 = [0.995 1.014];
% speedax = [-1 30];
% speedcax = [-9 25];


% % E157 stopping
% yax1 = [0.991 1.0065];
% yax2 = [0.975 1.04];
% cax1 = [0.996 1.007];
% cax2 = [0.992 1.02];
% speedax = [-1 25];
% speedcax = [-9 25];

% % E157 perireward
% yax1 = [0.985 1.01];
% yax2 = [0.975 1.03];
% cax1 = [0.996 1.015];
% cax2 = [0.98 1.03];
% speedax = [-1 25];
% speedcax = [-9 25];

%E157 HRZ starting
% yax1 = [0.986 1.01];
% yax2 = [0.992 1.01];
% cax1 = [0.996 1.007];
% cax2 = [0.992 1.02];
% speedax = [-5 75];
% speedcax = [-20 75];

%E157 HRZ stopping
% yax1 = [0.99 1.01];
% yax2 = [0.99 1.007];
% cax1 = [0.996 1.01];
% cax2 = [0.99 1.01];
% speedax = [-5 60];
% speedcax = [-20 65];

%E157 HRZ perireward
% yax1 = [0.99 1.01];
% yax2 = [0.99 1.007];
% cax1 = [0.996 1.009];
% cax2 = [0.99 1.02];
% speedax = [-5 60];
% speedcax = [-20 75];

% % E156 Starting
% yax1 = [0.994 1.0025];
% yax2 = [0.995 1.02];
% cax1 = [0.995 1.005];
% cax2 = [0.995 1.014];
% speedax = [-1 25];
% speedcax = [-9 25];

% % E156 Perireward
% yax1 = [0.994 1.003];
% yax2 = [0.995 1.011];
% cax1 = [0.997 1.003];
% cax2 = [0.995 1.015];
% speedax = [-1 20];
% speedcax = [-15 30];

%E156 HRZ Perireward
% yax1 = [0.994 1.01];
% yax2 = [0.995 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.995 1.005];
% speedax = [-1 40];
% speedcax = [-15 40];

%E156 HRZ stop
% yax1 = [0.994 1.01];
% yax2 = [0.995 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.995 1.005];
% speedax = [-1 40];
% speedcax = [-15 40];

%E156 HRZ start
% yax1 = [0.995 1.005];
% yax2 = [0.997 1.007];
% cax1 = [0.994 1.005];
% cax2 = [0.997 1.005];
% speedax = [-1 55];
% speedcax = [-15 60];

% % E148 Starting
% yax1 = [0.996 1.006];
% yax2 = [0.991 1.01];
% cax1 = [0.998 1.007];
% cax2 = [0.993 1.014];
% speedax = [-1 15];
% speedcax = [-9 20];

% % E148 Stopping
% yax1 = [0.996 1.004];
% yax2 = [0.991 1.006];
% cax1 = [0.9985 1.003];
% cax2 = [0.995 1.006];
% speedax = [-1 10];
% speedcax = [-5 12];

% % E148 perireward
% yax1 = [0.992 1.005];
% yax2 = [0.995 1.03];
% cax1 = [0.996 1.01];
% cax2 = [0.996 1.03];
% speedax = [-5 9];
% speedcax = [-5 9];

yax1 = [0.992 1.01];
yax2 = yax1;

% mouse = 'E148';


for checkforemptyfiles = files
    if isempty(roe_alldays_planes_perireward{checkforemptyfiles,1})
        roe_alldays_planes_perireward_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        roe_alldays_planes_perireward_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_0{files(1),1}));
        
        roi_dop_alldays_planes_perireward{checkforemptyfiles,1} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,2} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,3} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        roi_dop_alldays_planes_perireward{checkforemptyfiles,4} = NaN(size(roi_dop_alldays_planes_perireward{files(1),1}));
        
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,1} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,2} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,3} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        roe_alldays_planes_perireward_double_0{checkforemptyfiles,4} = NaN(size(roe_alldays_planes_perireward_double_0{files(1),1}));
        
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,1} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,2} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,3} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));
        roi_dop_alldays_planes_perireward_double{checkforemptyfiles,4} = NaN(size(roi_dop_alldays_planes_perireward_double{files(1),1}));

    end
end

%     roevariable = roe_allsuc_mov;
% roeindividualvariable = roe_alldays_planes_success_mov;
% dopvariable = dop_allsuc_mov;
% dopindividualvariable = dop_alldays_planes_success_mov;
roevariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_0,'UniformOutput',0);
roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roe_alldays_planes_perireward_0,'UniformOutput',0));


dopvariable = roi_dop_allsuc_perireward;


dopindividualvariable =  cellfun(@(x) transpose(x),roi_dop_alldays_planes_perireward,'UniformOutput',0);
roeindividualvariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_0,'UniformOutput',0);

roevariables{1} = roevariable; %perisingle
dopvariables{1} = dopvariable; %perisingle
roeindividualvariables{1} = roeindividualvariable; %perisingle
dopindividualvariables{1} = dopindividualvariable;% perisingle
variablelable{1} = 'Peri Single Reward'; %perisingle


roevariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_double_0,'UniformOutput',0);
roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roe_alldays_planes_perireward_double_0,'UniformOutput',0));
roeindividualvariable = cellfun(@(x) transpose(x),roe_alldays_planes_perireward_double_0,'UniformOutput',0);

dopvariable = roi_dop_allsuc_perireward_double;

dopindividualvariable =  cellfun(@(x) transpose(x),roi_dop_alldays_planes_perireward_double,'UniformOutput',0);


roevariables{2} = roevariable; %peridouble
dopvariables{2} = dopvariable; %peridouble
roeindividualvariables{2} = roeindividualvariable; %peridouble
dopindividualvariables{2} = dopindividualvariable;% peridouble
variablelable{2} = 'Peri Double Reward'; %peridouble

roevariables{3} = roe_allsuc_mov;
roeindividualvariables{3} = roe_alldays_planes_success_mov;
dopvariables{3} = roi_dop_allsuc_mov;
dopindividualvariables{3} = roi_dop_alldays_planes_success_mov;
variablelable{3} = 'Peri Start Triggered';

roevariables{4} = roe_allsuc_stop;
roeindividualvariables{4} = roe_alldays_planes_success_stop;
dopvariables{4} = roi_dop_allsuc_stop;
dopindividualvariables{4} = roi_dop_alldays_planes_success_stop;
variablelable{4} = 'Peri Stop Triggered';

roevariables{5} = roe_allsuc_stop_no_reward;
roeindividualvariables{5} = roe_alldays_planes_success_stop_no_reward;
dopvariables{5} = roi_dop_allsuc_stop_no_reward;
dopindividualvariables{5} = roi_dop_alldays_planes_success_stop_no_reward;
variablelable{5} = 'Peri Unrewarded Stop Triggered';

roevariables{6} = roe_allsuc_stop_reward;
roeindividualvariables{6} = roe_alldays_planes_success_stop_reward;
dopvariables{6} = roi_dop_allsuc_stop_reward;
dopindividualvariables{6} = roi_dop_alldays_planes_success_stop_reward;
variablelable{6} = 'Peri Rewarded Stop Triggered';

numROIs = length(ROI_labels);

for varsi = 1%1:length(roevariables)
    roevariable = roevariables{varsi};
    roeindividualvariable = roeindividualvariables{varsi};
    dopvariable = dopvariables{varsi};
    dopindividualvariable = dopindividualvariables{varsi};

for jj=1:numROIs
%     jj = ll;
    ll = numROIs-jj+1;
    fakeroilabel = ROI_labels{jj};
    fakeroilabel(regexp(fakeroilabel,'/')) = [];
 figure("name",[variablelable{varsi} ' early Day'])
    % early
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        if length(earlydays) == 1
            yax=mean(squeeze(dopindividualvariable{earlydays,jj}),1);
            se_yax=std(squeeze(dopindividualvariable{earlydays,jj}),1)./sqrt(size(squeeze(dopindividualvariable{latedays,jj}),1));
        else
            yax=mean(squeeze(dopvariable(earlydays,jj,:)),1);
            se_yax=std(squeeze(dopvariable(earlydays,jj,:)),1)./sqrt(size(squeeze(dopvariable(earlydays,jj,:)),1));
        end
    hold on,h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj};
    if jj>numROIs-1
        if exist('yax2')
        ylim(yax2)
          plot([0 0],yax2,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
          clear xlabel
    else
        if exist('yax1')
        ylim(yax1)
          plot([0 0],yax1,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
              clear xlabel; xlabel('Time(s)');
    end
    title(ROI_labels{jj})
    if saving
        saveas(gcf,[savepath fakeroilabel variablelable{varsi} '_early_day.svg'],'svg')
        xticks([])
        xlabel([])
        ylabel([])
        yticks([])
    end

    %late
    figure("name",[variablelable{varsi} ' Late Day'])
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
        if length(latedays) == 1
            yax=mean(squeeze(dopindividualvariable{latedays,jj}),1);
            se_yax=std(squeeze(dopindividualvariable{latedays,jj}),1)./sqrt(size(squeeze(dopindividualvariable{latedays,jj}),1));
        else
            yax=mean(squeeze(dopvariable(latedays,jj,:)),1);
            se_yax=std(squeeze(dopvariable(latedays,jj,:)),1)./sqrt(size(squeeze(dopvariable(latedays,jj,:)),1));
        end
    hold on, h10 = shadedErrorBar(xax,yax,se_yax,[],1);     h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj}; h10.edge(2).Color=color{jj};
    if jj>numROIs-1
        if exist('yax2')
        ylim(yax2)
        plot([0 0],yax2,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
        clear xlabel
    else
        if exist('yax1')
        ylim(yax1)
        plot([0 0],yax1,'k','Linewidth',2)
        else
            ylims = ylim;
            plot([0 0],ylims,'k','Linewidth',2)
        end
        clear xlabel; xlabel('Time(s)');
    end
    title(ROI_labels{jj})
     if saving
        saveas(gcf,[savepath fakeroilabel variablelable{varsi} '_late_day.svg'],'svg')
        xticks([])
        xlabel([])
        ylabel([])
        yticks([])
    end
end

end

%  if saving
%     hAx = findobj('type', 'figure');
%     for iAx = 1:length(hAx)
%         currentF = figure(hAx(iAx))
% 
% %         set(gcf, 'Renderer', 'Painters');
%         
%         saveas(currentF,[savepath currentF.Name '.svg'],'svg')
% %         export_fig(['D:\munneworkspaces\darkrewardfigures\StartTriggered\E148\E148_' axtitle '.svg'])
%     end
%  end






%% stop triggered
% path = 'N:\Munni';
close all

% path='G:\'
% path='G:\analysed\'
path = 'D:\munneworkspaces\';
% workspaces = {'E149_workspace','E156_workspace.mat','E157_workspace.mat','E158_workspace.mat'};
% workspaces = {'E148_RR_D1-8.mat','E149_RR_D1-7.mat','E156_RR_d5-7_9_14_F.mat',...
%     'E157_RR_d5_7-12_14_F.mat','E158_RR_d5-12_14_F.mat'};

% workspaces = {'E148_workspace_D1-12.mat','E149_workspace_D1-13.mat','E156_RR_F.mat',...
%     'E157_RR_F.mat','E158_RR_F.mat'};
workspaces = {'E156_HRZ.mat','E157_HRZ.mat','E158_HRZ.mat'};
fileslist = {[7 8 13 14 17:20 22],[5 7 15 18:25],[4 5 43 44]};
mice = cellfun(@(x) x(1:4),workspaces,'UniformOutput',0);


dopvariablename = 'dop_allsuc_mov';
roevariablename = 'roe_allsuc_mov';

savepath = 'D:\munneworkspaces\HRZfigures\StartTriggered\summaryfigure\';
saving = 1;


setylimmanual = [0.985 1.015];
roerescale = [0.99 1];
setxlimmanualsec = [-5 5];
Pcolor = {[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
timeforpost = [.5 2];
% mousestyle =
pstmouse = {};
cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
cats5={ 'dop_allsuc_perireward'};
cats6={'roe_allsuc_perireward'};
earlydays = [1 2];
latedays = [10 11];


for currmouse = 1:length(workspaces)
    load([path '\' workspaces{currmouse}])
    if currmouse == 1 
        close
    end
    files = fileslist{currmouse};
    find_figure('allmouse')
    
    cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    cats5={ 'dop_allsuc_perireward'};
cats6={'roe_allsuc_perireward'};
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
    
    

    
    roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
    
else 
       dopvariable = eval(dopvariablename);
    roevariable = eval(roevariablename);
end
    color = Pcolor;
    for jj = 1:4
        eval(sprintf('data1=%s',cats5{1}))%%%20 days dop
        subplot(2,length(workspaces),currmouse+length(workspaces))
        ylim(setylimmanual);
        ylims = ylim;
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
        yax=mean(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1);
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
        se_yax=std(squeeze(dopvariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)./sqrt(size(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1))
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
        h10.edge(2).Color=color{jj};
        ylim(setylimmanual);
        %     yticks([])  ylim(setylimmanual);
        
        %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
        %     hp=plot(yax,'Color',color{jj})
        %     legend(hp,strcat('plane',num2str(jj)));hold on
        %     xlabel('Time(s)');
        xlim(setxlimmanualsec)
        %     set(gca,'ylim',[0.99 1.01])
        if jj == 1c
            plot(xax,nanmean(squeeze(roevariable(files(fliplr(length(files)-earlydays+1)),jj,:)),1)/100*diff(roerescale)+roerescale(1),'k')
        end
        ylims = ylim;
        if jj == 4
            ylims = ylim;
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            ylim(ylims)
            pls.Color(4) = 0.5;
        end
        if currmouse == 1
            ylabel('Late Days')
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
        
    end
end


for currmouse = 1:length(workspaces)
    load([path '\' workspaces{currmouse}])
    if currmouse == 1
        close
    end
    files = fileslist{currmouse};
    find_figure('allmouse')
    
    cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
    cats4={ 'dop_suc_movt_pre'   'dop_suc_stopt_pre' };
    cats2={ 'dop_allsuc_mov'   'dop_allsuc_stop' };
    cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
    cats5={ 'dop_allsuc_perireward'};
    cats6={'roe_allsuc_perireward'};
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




roevariable = cell2mat(cellfun(@(x) reshape(nanmean(x,2),1,1,size(x,1)),roevariable,'UniformOutput',0));
    else
        dopvariable = eval(dopvariablename);
        roevariable = eval(roevariablename);
    end
    
    
    color = Pcolor;
    for jj = 1:4
        eval(sprintf('data1=%s',cats5{1}))%%%20 days dop
        subplot(2,length(workspaces),currmouse)
        ylim(setylimmanual);
        ylims = ylim;
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
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
        se_yax=std(squeeze(dopvariable(files(earlydays),jj,:)),1)./sqrt(size(squeeze(roevariable(files(earlydays),jj,:)),1))
        hold on, ;h10 = shadedErrorBar(xax,yax,se_yax,[],1);
        h10.patch.FaceColor = color{jj}; h10.mainLine.Color = color{jj}; h10.edge(1).Color = color{jj};
        h10.edge(2).Color=color{jj};
        ylim(setylimmanual);
        %     yticks([])  ylim(setylimmanual);
        
        %     hp=shadedErrorBar(xax,yax,se_yax,'Color',color{jj},1)
        %     hp=plot(yax,'Color',color{jj})
        %     legend(hp,strcat('plane',num2str(jj)));hold on
        %     xlabel('Time(s)');
        xlim(setxlimmanualsec)
        %     set(gca,'ylim',[0.99 1.01])
        if jj == 1c
            plot(xax,mean(squeeze(roevariable(files(earlydays),jj,:)),1)/100*diff(roerescale)+roerescale(1),'k')
        end
        ylims = ylim;
        if jj == 4
            ylims = ylim;
            pls = plot([0 0],ylims,'--k','Linewidth',1);
            ylim(ylims)
            pls.Color(4) = 0.5;
        end
        
        title(workspaces{currmouse}(1:4))
        if currmouse == 1
            ylabel('Early Days')
        end
        
        pst=nanmean(squeeze(data1(files,jj,40+ceil(timeforpost(1)/5*40):40+ceil(timeforpost(2)/5*40))),2);%%% 0.64-2.56
        mean_pre_pst(1,2)=nanmean(pst);
        meanpstmouse{currmouse,jj} = mean_pre_pst(1,2);
        se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
        pstmouse{currmouse,jj} = pst;
        corrcoef(pstmouse{currmouse,jj},pst)
        %             subplot(2,3,3+currmouse)
        %             imagesc(squeeze(data1(:,jj,:)))
        %             colormap(fake_parula)
        
    end
end






















% subplot(2,length(workspaces),length(workspaces)+1:2*length(workspaces)), cla()
% scaling = 2;
% spacescale = 0.3;
% xtickm = [];
% xtickl = {};
% ptest = [];
% clearvars xticks
% for jj = 1:4
%     for currmouse = 1:length(workspaces)
%         xtickm = [xtickm jj*scaling+spacescale*(currmouse-1)];
%         scatter(ones(size(pstmouse{currmouse,jj}))*jj*scaling+spacescale*(currmouse-1),...
%             pstmouse{currmouse,jj},20,color{jj},'filled','Jitter','on', 'jitterAmount', 0.1)
%         hold on
%         scatter(jj*scaling+spacescale*(currmouse-1),meanpstmouse{currmouse,jj},100,'k','s','LineWidth',2)
%         ylims = ylim;
%         xtickl = [xtickl, workspaces{currmouse}(1:4)];
%         if currmouse<length(workspaces)
%             [h(currmouse,jj),ptest(currmouse,jj)] = ttest2(pstmouse{currmouse,jj},pstmouse{length(workspaces),jj});
%             
%         end
%     end
%     
% end
% % realhs = bonferroni_holm(reshape(ptest,1,size(ptest,1)*size(ptest,2)));
% realhs = h;
% realhs = reshape(realhs,size(ptest,1),size(ptest,2));
% 
% for jj = 1:size(realhs,2)
%     for kk = 1:size(realhs,1)
%         if realhs(kk,jj) == 1
%             plot([jj*scaling+spacescale*(kk-1) jj*scaling+spacescale*(length(workspaces)-1)],[1.006+0.0019-kk*0.0005 1.006+0.0019-kk*0.0005],'k-')
%         end
%     end
% end
% xlim([1.5 10])
% xticks(xtickm)
% xticklabels(xtickl)
% title('Post Stop Average')
 flipmice = mice;
if saving
     set(gcf,'units','normalized','outerposition',[0 0 1 1])
    saveas(gcf,[savepath 'full_figure.svg'],'svg')


    hAx = findobj('type', 'axes');
    for iAx = 1:length(hAx)
        if iAx <= length(mice)
            if rem(iAx,length(mice)) == 0
                axtitle = [mice{length(mice)} '_Early_Days'];
            else
                axtitle = [mice{rem(iAx,length(mice))} '_Early_Days'];
            end
        else
            if rem(iAx,length(mice)) == 0
                axtitle = [mice{length(mice)} '_Late_Days'];
            else
                axtitle = [mice{rem(iAx,length(mice))} '_Late_Days'];
            end
        end
        
        fNew = figure('units','normalized','outerposition',[0 0 1 1]);
        hNew = copyobj(hAx(iAx), fNew);
%         title(axtitle)
        % Change the axes position so it fills whole figure
%         set(hNew, 'pos', [0.23162 0.2233 0.72058 0.63107])
        set(gca,'fontsize', 18)
        set(gca,'FontName','Arial')
        InSet = get(hNew, 'TightInset');
%         set(hNew, 'Position', [InSet(1:2)+0.05, 1-InSet(1)-InSet(3)-0.1, 1-InSet(2)-InSet(4)-0.05])
        set(gca,'units','normalized','outerposition',[0.01 0.1 0.95 0.88])
%         set(gcf, 'Renderer', 'Painters');
        
        saveas(fNew,[savepath axtitle '.svg'],'svg')
%         export_fig(['D:\munneworkspaces\darkrewardfigures\StartTriggered\E148\E148_' axtitle '.svg'])
    end
    

 end

