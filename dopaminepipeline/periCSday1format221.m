SOshadecolor = [230 84 128]/255;
SOshadealpha = 38/255;
notSOlinecolor = [153 153 153]/255;
notSOshadecolor = [77 77 77]/255;   
notSOshadealpha = 38/255;
dop4speedconv = [0.985 0.995];
speed2dopconv = [0 25];

figure;
yax = roi_dop_alldays_planes_periCS{1,4};
xax = linspace(-5,5,size(yax,1));
h = shadedErrorBar(xax,nanmean(yax,2),nanstd(yax,[],2)/sqrt(size(yax,2)),{'Color',SOshadecolor},1);
h.patch.FaceAlpha = SOshadealpha;
h.edge(1).LineStyle = 'none';
h.edge(2).LineStyle = 'none';
hold on

yax = cell2mat(roi_dop_alldays_planes_periCS(1,1:3));
xax = linspace(-5,5,size(yax,1));
h = shadedErrorBar(xax,nanmean(yax,2),nanstd(yax,[],2)/sqrt(size(yax,2)),{'Color',notSOshadecolor},1);
h.mainLine.Color = notSOlinecolor;
h.patch.FaceAlpha = notSOshadealpha;
h.edge(1).LineStyle = 'none';
h.edge(2).LineStyle = 'none';

speedyax = roi_roe_alldays_planes_periCS{1,1};
yax = nanmean(speedyax,2);
xax = linspace(-5,5,size(speedyax,1));
plot(xax,yax*(range(dop4speedconv)/range(speed2dopconv))+dop4speedconv(1),'k-')

plot([-3 -3],[1.01 1.02],'k-','LineWidth',2)
text(-3,1.02,'1dFF and 25cm/s')

xlim([-5 5])
xticks([-5 0 5])
ylims = ylim;
plot([0 0],ylims,'--','Color',[0 0 0 128]/255)
plot([0.5 0.5],ylims,'--','Color',[0 0 0 128]/255)