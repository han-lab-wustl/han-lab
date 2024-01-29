%load(INPUT wORKSPACE PATH HERE)

ROI_labels = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
 dopvarnames = {'roi_dop_alldays_planes_perireward','roi_dop_alldays_planes_peridoubleCS'};
 roevarnames = {'roe_alldays_planes_perireward','roi_roe_alldays_planes_peridoubleCS'};
%  for dop = 1:length(dopvarnames)


% if dop <5
%     dopvar = cellfun(@transpose,dopvar,'UniformOutput',0);
%     roevar = cellfun(@transpose,roevar,'UniformOutput',0);
% end
    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
test = eval(dopvarnames{1});
ndays = size(test,1);
for d = 1:ndays
    find_figure([dopvarnames{1} ' part ' num2str(ceil(d/10))]);
for r = 1:4
    for dop = 1
        dopvar = eval(dopvarnames{dop});
roevar = eval(roevarnames{dop});
    dopx = linspace(-5,5,size(dopvar{d,r},1));
    roex = linspace(-5,5,size(roevar{d,r},1));
    if ~all(all(isnan(dopvar{d,r})))
    subplot(4,10,d-floor(d/10.1)*10+((4-r+1)*10-10))
%     xax = 
yyaxis left
 yax1 = nanmean(dopvar{d,r},2);
    seyax1 = nanstd(dopvar{d,r},[],2)/sqrt(size(dopvar{d,r},2));
    xax = linspace(-5,5,length(yax1));
    
%     clearvars h
hold on
yyaxis left
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r}/dop;
    h.patch.FaceColor = planecolors{r}/dop;
    h.edge(1).Color = planecolors{r}/dop;
    h.edge(2).Color = planecolors{r}/dop;
%     clearvars h
%     plot(dopx,dopvar{d,r})
    hold on
    l = vline(0);
    l.Color = [0 0 0];
%     plot(dopx,nanmean(dopvar{d,r},2),'k-','LineWidth',1.5)
    ylim([min(min(dopvar{d,r}))-(max(max(dopvar{d,r}))-min(min(dopvar{d,r}))) max(max(dopvar{d,r}))]) 
    ylabel(ROI_labels{r})
    title(['Day ' num2str(d) ' n = ' num2str(size(dopvar{d,r},2))])
    
           
    
%     plot
    end
    end
    for dop = 1
        roevar = eval(roevarnames{dop});
        if ~all(all(isnan(roevar{d,r})))
    dopx = linspace(-5,5,size(dopvar{d,r},1));
    roex = linspace(-5,5,size(roevar{d,r},1));

          yyaxis right
             if dop == 1
             plot(roex,nanmean(roevar{d,r},2),'-','Color',[0.5 0.5 0.5],'LineWidth',1.5)
             else
                 plot(roex,nanmean(roevar{d,r},2),'k--','LineWidth',1.5)
             end
             ylim([min(nanmean(roevar{d,r},2)) max(nanmean(roevar{d,r},2))+range(nanmean(roevar{d,r},2))])
               yyaxis left
        end
    end
end
end
%  end