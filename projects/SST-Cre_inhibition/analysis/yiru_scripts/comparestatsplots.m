function p = comparestatsplots(allfirsttrialaverages,alllasttrialaverages,pre_color,post_color,x)
d_aspect=[4 0.25 1];

% subplot(1,2,1)

line_color=[.75 .75 .75];%RGB triplet to gray

scatter_alpha=0.6;

tick_length=[0.01 0.025];%default is [0.01 0.025]

scatter_size = 30;

capsize_size = 0;
% pre_color='k';%symbol color
% 
% post_color='r';

mean_symbol='s';%square

mean_line_size=2;

mean_size=100;%symbol

mean_symbol_line=0.7;%symbol line thickness


xticks(x)

% x_lim=[0.6 2.4];
% 
% xlim(x_lim)

% daspect(d_aspect)

x0=40;

y0=40;

width=200;

height=400;


hold on

% title('CPP')

y = [nanmean(allfirsttrialaverages), nanmean(alllasttrialaverages)];

sem(1) = std(allfirsttrialaverages)/sqrt(length(allfirsttrialaverages));

sem(2) = std(alllasttrialaverages)/sqrt(length(alllasttrialaverages)); % Original: allfirsttrialaverages


errorbar(x,y,sem,'Color',[0.5, 0.5, 0.5],'LineWidth',mean_line_size,'CapSize',capsize_size)%error bars, connecting line, color

errorbar(x(2),y(2),sem(2),'Color',[0.5, 0.5, 0.5],'LineWidth',mean_line_size,'CapSize',capsize_size)

scatter(ones(length(allfirsttrialaverages),1)*x(1),allfirsttrialaverages,scatter_size,pre_color,'MarkerEdgeAlpha',scatter_alpha, 'LineWidth', 1.5)%pre symbols

scatter(ones(length(alllasttrialaverages),1)*x(2),alllasttrialaverages,scatter_size,post_color,'MarkerEdgeAlpha',scatter_alpha, 'LineWidth', 1.5)%post symbols

scatter(x(1),y(1),mean_size,'k',mean_symbol,'LineWidth',mean_symbol_line)%pre mean symbol

scatter(x(2),y(2),mean_size,'k',mean_symbol,'LineWidth',mean_symbol_line)%post mean symbol

[h,p] = ttest2(allfirsttrialaverages,alllasttrialaverages);
% [p] = ranksum(allfirsttrialaverages,alllasttrialaverages);

text(x(2)+0.15,(y(1)+y(2))/2,['p=' num2str(p,'%.4g')]);
if isempty(allfirsttrialaverages)
    allfirsttrialaverages = [0];
end
if isempty(alllasttrialaverages)
    alllasttrialaverages = [0];
end
text(x(1)-0.6,max(allfirsttrialaverages),['n = ' num2str(length(allfirsttrialaverages))])
text(x(2)+0.2,max(alllasttrialaverages),['n = ' num2str(length(alllasttrialaverages))])

% temp_array={nanmean(allfirsttrialaverages), nanmean(alllasttrialaverages),sem(1),sem(2),h,p};
% 
% stats_array=vertcat(stats_array,temp_array);
ylabel('Speed (cm/s)')
% set(gca,'xticklabel',[],'FontSize',20)

ax1 = gca;                   % gca = get current axis

% ax1.TickLength=tick_length;


% set(gcf,'position',[x0,y0,width,height])

% tightfig;