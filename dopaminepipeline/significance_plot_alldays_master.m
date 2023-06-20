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




yax=SO_allmouse_mut;
se_yax=nanstd(yax,1)./sqrt(size(yax,1));
yax=nanmean(yax);
hold on, ;h11 = shadedErrorBar(xax,yax,se_yax,[],1);
if sum(isnan(se_yax))~=length(se_yax)
    h11.patch.FaceColor = color{end}/2; h11.mainLine.Color = color{end}/2; h11.edge(1).Color = color{end}/2;
    h11.edge(2).Color=color{end}/2; h11.mainLine.LineStyle='-'; h11.edge(2).LineStyle='-';
end
ylim(setylimmanual2);
legend([h11.edge(2) h10.edge(2) ], 'SO-GRABDA-mut','SO-GRABDA','onset','Location','northwest')

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
%%
%%%significance
%%mutant without SO

% cc=allmouse_dop_alldays{1,p}(grabdamut_rows(2):grabdamut_rows(end),1:3);out1= cat(1,cc{:});
% cc2=allmouse_dop_alldays{1,p}(grabdamut_rows(1),1:5); out3=(cat(1,cc2{:}));
% withoutSO_allmouse_mut_alldays=[out3; out1];


% cc=allmouse_dop_alldays{1,p}(2:grabda_rows(1),1:3);out1= cat(1,cc{:});
% cc2=allmouse_dop_alldays{1,p}(1,1:5); out2=(cat(1,cc2{:}));
% withoutSO_allmouse_dop_alldays=[out2 ;out1];%% mouse1-6



pst=find(xax>timeforpre(1)&xax<=timeforpre(2)); pst=find(xax>timeforpost(1)&xax<=timeforpost(2));
mtm_vals_mut=mean(withoutSO_allmouse_mut_alldays(:,pst),2);
mtm_vals_dop=mean(withoutSO_allmouse_dop_alldays(:,pst),2);

%%%%plotting significance

significance_plot_alldays(4,p,mtm_vals_mut,mtm_vals_dop,ylims,color,1,workspaces)

%%%%




%%%dop with SO

% SO_allmouse_dop_alldays=[cell2mat(allmouse_dop_alldays{1,p}(1,6)) ;cell2mat(allmouse_dop_alldays{1,p}(2:grabda_rows(end),4))];
% SO_allmouse_mut_alldays=[cell2mat(allmouse_dop_alldays{1,p}(grabdamut_rows(1),6)); cell2mat(allmouse_dop_alldays{1,p}(grabdamut_rows(2):grabdamut_rows(end),4))];

%%%
mtm_vals_mut=mean(SO_allmouse_mut_alldays(:,pst),2);
mtm_vals_dop=mean(SO_allmouse_dop_alldays(:,pst),2);

%%%%plotting significance

significance_plot_alldays(3,p,mtm_vals_mut,mtm_vals_dop,ylims,color,4,workspaces)
%%%%
%%
subplot(4,2,((p-1)*4+2)); hold on
yax=withoutSO_allmouse_dop;

se_yax=nanstd(yax,1)./sqrt(size(yax,1));
yax=nanmean(yax);
h10 = shadedErrorBar(xax,yax',se_yax,[],1);
if sum(isnan(se_yax))~=length(se_yax)
    h10.patch.FaceColor = color{1}; h10.mainLine.Color = color{1}; h10.edge(1).Color = color{1};
    h10.edge(2).Color=color{1};
end


yax=withoutSO_allmouse_mut;
se_yax=nanstd(yax,1)./sqrt(size(yax,1));
yax=nanmean(yax);
h11 = shadedErrorBar(xax,yax',se_yax,[],1);
if sum(isnan(se_yax))~=length(se_yax)
    h11.patch.FaceColor = color{1}/2; h11.mainLine.Color = color{1}/2; h11.edge(1).Color = color{1}/2;
    h11.edge(2).Color=color{1}/2; h11.mainLine.LineStyle='-';h11.edge(2).LineStyle='-';
end
legend([h11.edge(2) h10.edge(2)], 'AllexceptSO-GRABDA-mut','AllexceptSO-GRABDA','onset','Location','northwest')
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





