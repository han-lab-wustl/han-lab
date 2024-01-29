clear all
close all
wrk_dir = uipickfiles('Prompt','Pick the Workspace you would like to add');

load(wrk_dir{1})
%%
CS_trans_lags_alldays = cell(size(roi_dop_alldays_planes_periCS));
Fs = 31.25/numplanes;
test1 = roi_dop_alldays_planes_periCS;
test1 = rightAlignCell(test1);
test1 = test1(:,end-3:end);
test1pre = cellfun(@(x) x(1:ceil(size(x,1)/2)-1,:),test1,'UniformOutput',0);
test1post = cellfun(@(x) x(ceil(size(x,1)/2):end,:),test1,'UniformOutput',0);
test2pre = cell2mat(cellfun(@(x) reshape(x,[],1),test1pre,'UniformOutput',0));
test2post = cell2mat(cellfun(@(x) reshape(x,[],1),test1post,'UniformOutput',0));
for pl = 1:size(test1,2)
    bins = 0.9:0.0005:1.09;
    c1 = histc(test2pre(:,pl),bins);
    c2 = histc(test2post(:,pl),bins);
    if skewness(test2post(:,pl))>= 0
    [~,ind1] = max(c2(202:end)-c1(202:end)); %201 == "bins" index where value is 1
    ind1 = ind1+201;
    else
        [~,ind1] = max(c2(1:201)-c1(1:201));
    end
    if bins(ind1)>=1
        temp = find((c2(1:ind1)-c1(1:ind1))<=0,1,'last');
        ind2 = temp;
    else
        temp = find((c2(ind1:end)-c1(ind1:end))<=0,1);
        ind2 = temp+ind1;
    end
    peaktranscutoff = bins(ind1);
    startranscutoff = bins(ind2);
for d = 1:size(roi_dop_alldays_planes_periCS,1)
figure;
CStranslag = [];
for p = 1:size(roi_dop_alldays_planes_periCS{d,pl},2)
    subtightplot(8,8,p)
    currtrace = test1{d,pl}(:,p);
    plot(currtrace)
    hold on
    if peaktranscutoff>1
        temp = find(currtrace(ceil(length(currtrace)/2):end)>=peaktranscutoff,1);
        
        if ~isempty(temp)
            temp2 = find(currtrace(ceil(length(currtrace)/2):temp-1+ceil(length(currtrace)/2))<=startranscutoff,1,'last');
            if ~isempty(temp2)
                transstart = temp2;
            else
                transstart = 1;
            end
            plot(transstart+ceil(length(currtrace)/2)-1,currtrace(transstart+ceil(length(currtrace)/2)-1),'r*','MarkerSize',20)
            CStranslag(p) = (transstart-1)/Fs;
        else
            CStranslag(p) = NaN;
        end
    else
        temp = find(currtrace(ceil(length(currtrace)/2):end)<=peaktranscutoff,1);
        if ~isempty(temp)
            temp2 = find(currtrace(ceil(length(currtrace)/2):temp-1+ceil(length(currtrace)/2))>=startranscutoff,1,'last');
            if ~isempty(temp2)
                transstart = temp2;
            else
                transstart = 1;
            end
            plot(transstart+ceil(length(currtrace)/2)-1,currtrace(transstart+ceil(length(currtrace)/2)-1),'r*','MarkerSize',20)
            CStranslag(p) = (transstart-1)/Fs;
        else
            CStranslag(p) = NaN;
        end
        
    end
%     hline(1+2*std(roi_dop_alldays_planes_periCS{16,4}(35:39,p)))
end
CS_trans_lags_alldays{d,pl} = CStranslag;
mtit(['day ' num2str(d) ' Plane ' num2str(pl)])
end
end


figure; 
legendss = {};
for pl = 1:size(CS_trans_lags_alldays,2)
errorbar(cellfun(@nanmean,CS_trans_lags_alldays(:,pl)),cellfun(@(x) nanstd(x)/sqrt(sum(~isnan(x))),CS_trans_lags_alldays(:,pl),'UniformOutput',1))
xlabel('Days')
ylabel('Latency (s)')
title('Trans from CS')
hold on
legendss = [legendss {['ROI ' num2str(pl)]}];
end
legend(legendss)



% test1 = roi_dop_alldays_planes_periCS;
% test1pre = cellfun(@(x) x(1:ceil(size(x,1)/2)-1,:),test1,'UniformOutput',0);
% test1post = cellfun(@(x) x(ceil(size(x,1)/2):end,:),test1,'UniformOutput',0);
% test2pre = cell2mat(cellfun(@(x) reshape(x,[],1),test1pre,'UniformOutput',0));
% test2post = cell2mat(cellfun(@(x) reshape(x,[],1),test1post,'UniformOutput',0));
% figure;
% for pl = 1:size(test2post,2)
%     subplot(ceil(size(test2post,2)/2),2,pl)
%     histogram(test2pre(:,pl),'BinWidth',0.0005)
%     hold on
%     histogram(test2post(:,pl),'BinWidth',0.0005)
%     if pl == 2
%         legend({'Pre-CS','Post-CS'})
%     end
%     title(['ROI ' num2str(pl)])
% end
% figure;
% for pl = 1:size(test2post,2)
%     subplot(ceil(size(test2post,2)/2),2,pl)
%     bins = 0.9:0.0005:1.09;
%     c1 = histc(test2pre(:,pl),bins);
%     c2 = histc(test2post(:,pl),bins);
%     plot(c2-c1);
%     title(['ROI ' num2str(pl)])
% end

% %% Speed
% CS_stop_lags_alldays = cell(size(roi_roe_alldays_planes_periCS,1),1);
% Fs = 31.25;
% thresh = 5; %cm/s
% for pl = 1
%   
% for d = 1:size(roi_roe_alldays_planes_periCS,1)
% figure;
%   CSstoplag = [];
% for p = 1:size(roi_roe_alldays_planes_periCS{d,pl},2)
%     subtightplot(8,8,p)
%     currtrace = roi_roe_alldays_planes_periCS{d,pl}(:,p);
%     plot(currtrace)
%     hold on
%     transstart = find(currtrace(ceil(length(currtrace)/2):end)<=thresh,1);
%         if ~isempty(transstart)
%             plot(transstart+ceil(length(currtrace)/2)-1,currtrace(transstart+ceil(length(currtrace)/2)-1),'r*','MarkerSize',20)
%             CSstoplag(p) = (transstart-1)/Fs;
%         else
%             CSstoplag(p) = NaN;
%         end
% 
% end
% CS_stop_lags_alldays{d,pl} = CSstoplag;
% mtit(['day ' num2str(d) ' Speed'])
% end
% end
% figure; 
% errorbar(cellfun(@nanmean,CS_stop_lags_alldays),cellfun(@(x) nanstd(x)/sqrt(sum(~isnan(x))),CS_stop_lags_alldays,'UniformOutput',1),'k')
% xlabel('Days')
% ylabel('Latency (s)')
% title('Stop from CS')


%  view transient histograms
% test1 = roi_dop_alldays_planes_periCS;
% if strcmp(pr_dir0{1},'M:\dark_reward\E156\Day_3')
% [E156fixidxx,E156fixidxy] = find(cellfun(@(x) sum(size(x)),test1,'UniformOutput',1)==0);
% for p = 0:4
% test1{E156fixidxx,6-p} = test1{E156fixidxx,6-p-1};
% end
% test1{E156fixidxx,1} = NaN(size(test1{E156fixidxx,2}));
% end
% test1pre = cellfun(@(x) x(1:ceil(size(x,1)/2)-1,:),test1,'UniformOutput',0);
% test1post = cellfun(@(x) x(ceil(size(x,1)/2):end,:),test1,'UniformOutput',0);
% test2pre = cell2mat(cellfun(@(x) reshape(x,[],1),test1pre,'UniformOutput',0));
% test2post = cell2mat(cellfun(@(x) reshape(x,[],1),test1post,'UniformOutput',0));
% figure;
% for pl = 1:size(test2post,2)
%     subplot(ceil(size(test2post,2)/2),2,pl)
%     histogram(test2pre(:,pl),'BinWidth',0.0005)
%     hold on
%     histogram(test2post(:,pl),'BinWidth',0.0005)
%     if pl == 2
%         legend({'Pre-CS','Post-CS'})
%     end
%     title(['ROI ' num2str(pl)])
%     xlim([0.96 1.09])
% end
% figure;
% for pl = 1:size(test2post,2)
%     subplot(ceil(size(test2post,2)/2),2,pl)
%     bins = 0.9:0.0005:1.09;
%     c1 = histc(test2pre(:,pl),bins);
%     c2 = histc(test2post(:,pl),bins);
%     plot(bins,c2-c1);
%     title(['ROI ' num2str(pl)])
% end