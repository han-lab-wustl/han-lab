 function significance_plot_alldays(subplotidz,p,mtm_vals_mut,mtm_vals_dop,ylims,color,coloridz,workspaces)
mean_mtm_vals=[]; std_mtm_vals=[];
mean_mtm_vals(1,:)=mean(mtm_vals_mut);  std_mtm_vals(1,:)=std(mtm_vals_mut)/sqrt(size(mtm_vals_mut,1))
mean_mtm_vals(2,:)=mean(mtm_vals_dop);  std_mtm_vals(2,:)=std(mtm_vals_dop)/sqrt(size(mtm_vals_dop,1))

subplot(4,2,(p-1)*4+subplotidz);
y=NaN(max([size(mtm_vals_mut) size(mtm_vals_dop)]),2);
y(1:size(mtm_vals_mut,1),1)=mtm_vals_mut;y(1:size(mtm_vals_dop,1),2)=mtm_vals_dop;
[r, c] = size(y);
xdata = repmat(1:c, r, 1);
ydata=y;
% for explanation see
% http://undocumentedmatlab.com/blog/undocumented-scatter-plot-jitter
scatter(xdata(:), y(:), 5, color{coloridz},'filled', 'jitter','on', 'jitterAmount', 0.01);

xticks([1 2])
xticklabels({'GRABDA-mut','GRABDA'})

set(gca,'xlim',[0.5 2.5]); hold on
errorbar(mean_mtm_vals', std_mtm_vals','k','Linewidth',2)
ylim(ylims)
[h,pv,df,tstats]=ttest2(mtm_vals_dop,mtm_vals_mut)
text(2.1,1.01,strcat('pval=',num2str(pv)))
text(1.1,0.99,strcat('n=',num2str(size(mtm_vals_mut,1))));
text(2.1,0.99,strcat('n=',num2str(size(mtm_vals_dop,1))));

