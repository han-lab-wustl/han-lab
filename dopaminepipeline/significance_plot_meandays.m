 function significance_plot_meandays(subplotidz,p,mtm_vals_mut,mtm_vals_dop,ylims,color,coloridz,mouse_color,colormut,colordop,workspaces)
mean_mtm_vals=[]; std_mtm_vals=[];
mtm_vals_mut
% 3,p,mtm_vals_mut',mtm_vals_dop',ylims,color,4,coloridx,mouse_idz_SO_mut,mouse_idz_SO_dop,workspaces


mean_mtm_vals(1,:)=mean(mtm_vals_mut);  std_mtm_vals(1,:)=std(mtm_vals_mut)/sqrt(size(mtm_vals_mut,1))
mean_mtm_vals(2,:)=mean(mtm_vals_dop);  std_mtm_vals(2,:)=std(mtm_vals_dop)/sqrt(size(mtm_vals_dop,1))
% ylims=[0.97 1.05]
subplot(4,2,(p-1)*4+subplotidz);
y=NaN(max([size(mtm_vals_mut) size(mtm_vals_dop)]),2);
y(1:size(mtm_vals_mut,1),1)=mtm_vals_mut;y(1:size(mtm_vals_dop,1),2)=mtm_vals_dop;
[r, c] = size(y);
xdata = repmat(1:c, r, 1);
ydata=y;
% for explanation see
% http://undocumentedmatlab.com/blog/undocumented-scatter-plot-jitter
% scatter(xdata(:), y(:), 20, color{coloridz},'filled', 'jitter','on', 'jitterAmount', 0.01);

xticks([1 2])
xticklabels({'GRABDA-mut','GRABDA'})

set(gca,'xlim',[0.5 2.5]); hold on
errorbar(mean_mtm_vals', std_mtm_vals','k','Linewidth',2)
ylim(ylims)
[h,pv,df,tstats]=ttest2(mtm_vals_dop,mtm_vals_mut)
text(1.4,1.01,strcat('pval=',num2str(pv)))
text(1.1,0.99,strcat('n=',num2str(size(mtm_vals_mut,1))));
text(2.1,0.99,strcat('n=',num2str(size(mtm_vals_dop,1))));
%%%mutnat mouse idx color
mut_id=ones(1,length(colormut));
dop_id=2*ones(1,length(colordop));

cmut_uni=unique(colormut);
for jj=1:length(cmut_uni)
    
    smut_id=mean(mut_id(find(colormut==cmut_uni(jj))));
    smut_vals=mean(mtm_vals_mut(find(colormut==cmut_uni(jj))))
    ax(jj)=scatter(smut_id,smut_vals,15,mouse_color{cmut_uni(jj)},'filled','jitter','on', 'jitterAmount', 0.01)
    
end
% legend(ax,'Location','southwest')
% ah1=axes('position',get(gca,'position'),'visible','off');
%%% dopamine mouse idx color

cdop_uni=unique(colordop);
for jj=1:length(cdop_uni)
    sdop_id=mean(dop_id(find(colordop==cdop_uni(jj))));
     sdop_vals=mean(mtm_vals_dop(find(colordop==cdop_uni(jj))));
    ax2(jj)=scatter(sdop_id,sdop_vals,15,mouse_color{cdop_uni(jj)},'filled','jitter','on', 'jitterAmount', 0.01)

end
hold on

exp=cellfun(@(x) strcat('E',x(1:3)),workspaces,'UniformOutput',false);

legend(ax2,exp(1:length(cdop_uni)),'Location','southeast')



