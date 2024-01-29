path = 'D:\workspaces_darkreward\';
cd(path)

workspace = {'167_dark_reward_AllDays_CutPlanes_workspace.mat','168_dark_reward_workspace.mat','169_dark_reward_AllDays_CutPlanes_workspace.mat',...
        '171_dark_reward_AllDays_CutPlanes_workspace','179_dark_reward1_workspace_00.mat','181_dark_reward1_workspace.mat'};



  mCSlicklatency = cell(length(workspace),1);
  mCSstoplatency = cell(length(workspace),1);
  mCSUSdopavg = cell(length(workspace),4);
  mCSwindowavg = cell(length(workspace),4);
  mCSwindowmax = cell(length(workspace),4);
  mCSwindowcom = cell(length(workspace),4);
  mCSwindowcomMean = cell(length(workspace),4);
  mCSwindownegcom = cell(length(workspace),4);

for ws = 1:length(workspace)
    load(workspace{ws})
mCSlicklatency{ws} = cellfun(@nanmean,CS_alldays_lick_latency)';

mCSstoplatency{ws} = cellfun(@nanmean,CS_alldays_stop_gap)';

        if exist('mouseid')
        if mouseid == 181 && isempty(roi_dop_alldays_planes_periCS{19,4})
            temp = [repmat({NaN(79,3)},5,1) roi_dop_alldays_planes_periCS(19:23,:)];
            temp(:,5) = [];
        roi_dop_alldays_planes_periCS(19:23,:) = temp;
%         roi_dop_alldays_planes_periCS(:,5) = [];
        temp = [repmat({NaN(79,3)},5,1) roi_dop_alldays_planes_perireward(19:23,:)];
            temp(:,5) = [];
        roi_dop_alldays_planes_perireward(19:23,:) = temp;
        end
      
        end

        means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
        for pl = 1:4
            mCSUSdopavg{ws,pl} = means(:,pl)';
        end
        for d = 1:size(roi_dop_alldays_planes_perireward,1)
            xax = linspace(-5,5,size(roi_dop_alldays_planes_perireward{d,1},1));
            windowind = find(xax>=-0.5&xax<=1);
            for pl = 1:4
            mCSwindowavg{ws,pl}(d) = nanmean(nanmean(roi_dop_alldays_planes_perireward{d,pl}(windowind,:),1));
            mCSwindowmax{ws,pl}(d) = nanmean(max(roi_dop_alldays_planes_perireward{d,pl}(windowind,:),[],1));
            mCSwindowcom{ws,pl}(d) = nanmean(calc_COM_EH(rescale_row(roi_dop_alldays_planes_perireward{d,pl}(windowind,:))',1)*(xax(2)-xax(1))+xax(windowind(1)));
            mCSwindownegcom{ws,pl}(d) = nanmean(calc_COM_EH((rescale_row(roi_dop_alldays_planes_perireward{d,pl}(windowind,:))-1)'*-1,1)*(xax(2)-xax(1))+xax(windowind(1)));
            mCSwindowcomMean{ws,pl}(d) = calc_COM_EH(rescale_row(nanmean(roi_dop_alldays_planes_perireward{d,pl}(windowind,:),2)'),1)*(xax(2)-xax(1))+xax(windowind(1));
            end
        end

end

planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
clearvars xticks

 %% re aligning Dop variables - lick
       
dopvariables = {'mCSwindowavg','mCSwindowmax','mCSwindowcom','mCSwindownegcom','mCSwindowcomMean'};
doptitles = {'Average from -0.5 to 1','Max from -0.5 to 1','COM of Dop -0.5 to 1','COM of -1*Dop -0.5 to 1','COM of Avg Dop -0.5 to 1'};
dopylabels = {'dFF','dFF','Time (s)','Time (s)','Time (s)'};
  for dov = 1:length(dopvariables)
      currvar = eval(dopvariables{dov});
      temp = currvar(:,2);
      temp(3) = currvar(3,1);
% find_figure([doptitles{dov} ' individual mice align 1']);
lickcombined = padcatcell2mat(cellfun(@transpose,mCSlicklatency,'uniformOutput',0)',1,'left');
dopcombined = padcatcell2mat(cellfun(@transpose,temp,'uniformOutput',0)',1,'left');
%        errorbar(nanmean(lickcombined,2),nanstd(lickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
aligninds = [];
startingind = [];
for m = 1:size(lickcombined,2)
    aligninds(m) = find(lickcombined(:,m)<0.25,1);
    startingind(m) = find(~isnan(lickcombined(:,m)),1);
end
alignlickcombined = NaN(max(aligninds-startingind)+size(lickcombined,1)-min(aligninds)+1,size(lickcombined,2));
for m = 1:size(lickcombined,2)
    alignlickcombined(max(aligninds-startingind)-(aligninds(m)-startingind(m))+1:max(aligninds-startingind)+(size(lickcombined,1)-aligninds(m))+1,m) = dopcombined(startingind(m):end,m);
end
xax = (1:size(alignlickcombined))-max(aligninds-startingind)-1;
% plot(xax,alignlickcombined)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%        title(doptitles{dov})
%        xticks(xax)
%        legend({'167','168','169','171'})
       
       
%       find_figure([doptitles{dov} ' combined align 1']);
%        subplot(2,1,1)
%        errorbar(xax,nanmean(alignlickcombined,2),nanstd(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
       
%        subplot(2,1,2)
%        errorbar(xax,mean(alignlickcombined,2),std(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 0.55])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%       title(doptitles{dov})
%        xticks(xax)
%        xlim([-3.5 6.5])
       
       find_figure([doptitles{dov} ' quant align 1'])
       datind = find(xax<0&sum(~isnan(alignlickcombined),2)' == size(alignlickcombined,2));
       DAdata = reshape(alignlickcombined(datind,1:4),1,[]);
       Mutdata = reshape(alignlickcombined(datind,5:end),1,[]);
       if dov<3
           DAdata = DAdata-1;
           Mutdata = Mutdata-1;
       end
       plot(ones(1,length(DAdata)),DAdata,'r.')
       [~,p] = ttest(DAdata);
       text(1,max(DAdata),['P = ' num2str(p)])
       hold on
        plot(ones(1,length(Mutdata))*2,Mutdata,'k.')
        [~,p] = ttest(Mutdata);
       text(2,max(Mutdata),['P = ' num2str(p)])
        legend('DA','Mut')
        xlim([0 3])
       
       

% find_figure([doptitles{dov} ' individual mice align 2']);
lickcombined = padcatcell2mat(cellfun(@transpose,mCSlicklatency,'uniformOutput',0)',1,'left');
%        errorbar(nanmean(lickcombined,2),nanstd(lickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
aligninds = [];
startingind = [];
for m = 1:size(lickcombined,2)
    aligninds(m) = find(lickcombined(:,m)<0.25,1);
    startingind(m) = find(~isnan(lickcombined(:,m)),1);
end
temp = consecutive_stretch(find(lickcombined(:,3)<0.25));
aligninds(3) = temp{2}(1);
alignlickcombined = NaN(max(aligninds-startingind)+size(lickcombined,1)-min(aligninds)+1,size(lickcombined,2));
for m = 1:size(lickcombined,2)
    alignlickcombined(max(aligninds-startingind)-(aligninds(m)-startingind(m))+1:max(aligninds-startingind)+(size(lickcombined,1)-aligninds(m))+1,m) = dopcombined(startingind(m):end,m);
end
xax = (1:size(alignlickcombined))-max(aligninds-startingind)-1;
% plot(xax,alignlickcombined)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        legend({'167','168','169','171'})
       
       
%        find_figure([doptitles{dov} ' combined align 2']);
%        subplot(2,1,1)
%        errorbar(xax,nanmean(alignlickcombined,2),nanstd(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        
%        subplot(2,1,2)
%        errorbar(xax,mean(alignlickcombined,2),std(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 0.55])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        xlim([-3.5 6.5])
       
      find_figure([doptitles{dov} ' quant align 2'])
       datind = find(xax<0&sum(~isnan(alignlickcombined),2)' == size(alignlickcombined,2));
       DAdata = reshape(alignlickcombined(datind,1:4),1,[]);
       Mutdata = reshape(alignlickcombined(datind,5:end),1,[]);
       if dov<3
           DAdata = DAdata-1;
           Mutdata = Mutdata-1;
       end
       plot(ones(1,length(DAdata)),DAdata,'r.')
       [~,p] = ttest(DAdata);
       text(1,max(DAdata),['P = ' num2str(p)])
       hold on
        plot(ones(1,length(Mutdata))*2,Mutdata,'k.')
        [~,p] = ttest(Mutdata);
       text(2,max(Mutdata),['P = ' num2str(p)]) 
        legend('DA','Mut')
        xlim([0 3])
        
       
%        find_figure([doptitles{dov} ' individual mice align 3']);
lickcombined = padcatcell2mat(cellfun(@transpose,mCSlicklatency,'uniformOutput',0)',1,'left');
%        errorbar(nanmean(lickcombined,2),nanstd(lickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
aligninds = [];
startingind = [];
for m = 1:size(lickcombined,2)
    aligninds(m) = find(lickcombined(:,m)<0.25,1);
    startingind(m) = find(~isnan(lickcombined(:,m)),1);
end
temp = consecutive_stretch(find(lickcombined(:,3)<0.25));
% aligninds(3) = temp{2}(1);
aligninds(1) = 13;
alignlickcombined = NaN(max(aligninds-startingind)+size(lickcombined,1)-min(aligninds)+1,size(lickcombined,2));
for m = 1:size(lickcombined,2)
    alignlickcombined(max(aligninds-startingind)-(aligninds(m)-startingind(m))+1:max(aligninds-startingind)+(size(lickcombined,1)-aligninds(m))+1,m) = dopcombined(startingind(m):end,m);
end
xax = (1:size(alignlickcombined))-max(aligninds-startingind)-1;
% plot(xax,alignlickcombined)
%        ylabel(dopylabels{dov})
%        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        legend({'167','168','169','171'})
       
       
%        find_figure([doptitles{dov} ' combined align 3']);
%        subplot(2,1,1)
%        errorbar(xax,nanmean(alignlickcombined,2),nanstd(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
       
%        subplot(2,1,2)
%        errorbar(xax,mean(alignlickcombined,2),std(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 0.55])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        xlim([-3.5 6.5])

find_figure([doptitles{dov} ' quant align 3'])
       datind = find(xax<0&sum(~isnan(alignlickcombined),2)' == size(alignlickcombined,2));
       DAdata = reshape(alignlickcombined(datind,1:4),1,[]);
       Mutdata = reshape(alignlickcombined(datind,5:end),1,[]);
       if dov<3
           DAdata = DAdata-1;
           Mutdata = Mutdata-1;
       end
       plot(ones(1,length(DAdata)),DAdata,'r.')
       [~,p] = ttest(DAdata);
       text(1,max(DAdata),['P = ' num2str(p)])
       hold on
        plot(ones(1,length(Mutdata))*2,Mutdata,'k.')
        [~,p] = ttest(Mutdata);
       text(2,max(Mutdata),['P = ' num2str(p)])
       legend('DA','Mut')
       xlim([0 3])
       
  end
  
   %% re aligning Dop variables - stop
       
dopvariables = {'mCSwindowavg','mCSwindowmax','mCSwindowcom','mCSwindownegcom','mCSwindowcomMean'};
doptitles = {'Average from -0.5 to 1','Max from -0.5 to 1','COM of Dop -0.5 to 1','COM of -1*Dop -0.5 to 1','COM of Avg Dop -0.5 to 1'};
dopylabels = {'dFF','dFF','Time (s)','Time (s)','Time (s)'};
  for dov = 1:length(dopvariables)
      currvar = eval(dopvariables{dov});
      temp = currvar(:,2);
      temp(3) = currvar(3,1);
% find_figure([doptitles{dov} ' individual mice align 1']);
lickcombined = padcatcell2mat(cellfun(@transpose,mCSstoplatency,'uniformOutput',0)',1,'left');
dopcombined = padcatcell2mat(cellfun(@transpose,temp,'uniformOutput',0)',1,'left');
%        errorbar(nanmean(lickcombined,2),nanstd(lickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
aligninds = [];
startingind = [];
for m = 1:size(lickcombined,2)
    aligninds(m) = find(lickcombined(:,m)<0.5,1);
    startingind(m) = find(~isnan(lickcombined(:,m)),1);
end
alignlickcombined = NaN(max(aligninds-startingind)+size(lickcombined,1)-min(aligninds)+1,size(lickcombined,2));
for m = 1:size(lickcombined,2)
    alignlickcombined(max(aligninds-startingind)-(aligninds(m)-startingind(m))+1:max(aligninds-startingind)+(size(lickcombined,1)-aligninds(m))+1,m) = dopcombined(startingind(m):end,m);
end
xax = (1:size(alignlickcombined))-max(aligninds-startingind)-1;
% plot(xax,alignlickcombined)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%        title(doptitles{dov})
%        xticks(xax)
%        legend({'167','168','169','171'})
       
       
%       find_figure([doptitles{dov} ' combined align 1']);
%        subplot(2,1,1)
%        errorbar(xax,nanmean(alignlickcombined,2),nanstd(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
       
%        subplot(2,1,2)
%        errorbar(xax,mean(alignlickcombined,2),std(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 0.55])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%       title(doptitles{dov})
%        xticks(xax)
%        xlim([-3.5 6.5])
       
       find_figure([doptitles{dov} ' quant align 1'])
       datind = find(xax<0&sum(~isnan(alignlickcombined),2)' == size(alignlickcombined,2));
       DAdata = reshape(alignlickcombined(datind,1:4),1,[]);
       Mutdata = reshape(alignlickcombined(datind,5:end),1,[]);
       if dov<3
           DAdata = DAdata-1;
           Mutdata = Mutdata-1;
       end
       plot(ones(1,length(DAdata)),DAdata,'r.')
       [~,p] = ttest(DAdata);
       text(1,max(DAdata),['P = ' num2str(p)])
       hold on
        plot(ones(1,length(Mutdata))*2,Mutdata,'k.')
        [~,p] = ttest(Mutdata);
       text(2,max(Mutdata),['P = ' num2str(p)])
        legend('DA','Mut')
        xlim([0 3])
       
       

% find_figure([doptitles{dov} ' individual mice align 2']);
lickcombined = padcatcell2mat(cellfun(@transpose,mCSstoplatency,'uniformOutput',0)',1,'left');
%        errorbar(nanmean(lickcombined,2),nanstd(lickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
aligninds = [];
startingind = [];
for m = 1:size(lickcombined,2)
    aligninds(m) = find(lickcombined(:,m)<0.5,1);
    startingind(m) = find(~isnan(lickcombined(:,m)),1);
end
temp = consecutive_stretch(find(lickcombined(:,3)<0.5));
aligninds(3) = temp{3}(1);
alignlickcombined = NaN(max(aligninds-startingind)+size(lickcombined,1)-min(aligninds)+1,size(lickcombined,2));
for m = 1:size(lickcombined,2)
    alignlickcombined(max(aligninds-startingind)-(aligninds(m)-startingind(m))+1:max(aligninds-startingind)+(size(lickcombined,1)-aligninds(m))+1,m) = dopcombined(startingind(m):end,m);
end
xax = (1:size(alignlickcombined))-max(aligninds-startingind)-1;
% plot(xax,alignlickcombined)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        legend({'167','168','169','171'})
       
       
%        find_figure([doptitles{dov} ' combined align 2']);
%        subplot(2,1,1)
%        errorbar(xax,nanmean(alignlickcombined,2),nanstd(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 1.2])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        
%        subplot(2,1,2)
%        errorbar(xax,mean(alignlickcombined,2),std(alignlickcombined,[],2)/sqrt(4),'-','Capsize',0,'LineWidth',1.5)
%        ylabel(dopylabels{dov})
% %        ylim([0 0.55])
%        ylims = ylim;
%        text(xax,ones(1,size(alignlickcombined,1))*ylims(2)-0.001,num2str(sum(~isnan(alignlickcombined(:,1:4)),2)),'HorizontalAlignment','center')
%         title(doptitles{dov})
%        xticks(xax)
%        xlim([-3.5 6.5])
       
      find_figure([doptitles{dov} ' quant align 2'])
       datind = find(xax<0&sum(~isnan(alignlickcombined),2)' == size(alignlickcombined,2));
       DAdata = reshape(alignlickcombined(datind,1:4),1,[]);
       Mutdata = reshape(alignlickcombined(datind,5:end),1,[]);
       if dov<3
           DAdata = DAdata-1;
           Mutdata = Mutdata-1;
       end
       plot(ones(1,length(DAdata)),DAdata,'r.')
       [~,p] = ttest(DAdata);
       text(1,max(DAdata),['P = ' num2str(p)])
       hold on
        plot(ones(1,length(Mutdata))*2,Mutdata,'k.')
        [~,p] = ttest(Mutdata);
       text(2,max(Mutdata),['P = ' num2str(p)]) 
        legend('DA','Mut')
        xlim([0 3])
        
 
  end
  