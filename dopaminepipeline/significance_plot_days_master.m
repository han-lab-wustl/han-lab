find_figure(strcat('casevscontrol days',cats5{allcat}))


subplot(4,2,(p-1)*4+1);
xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
patchy.EdgeAlpha = 0;
patchy.FaceAlpha - 0.5;

yax=SO_allmouse_dop;
se_yax=nanstd(yax,1)./sqrt(size(yax,1));
yax=nanmean(yax);
hold on, ;h10= shadedErrorBar(xax,yax,se_yax,[],1);
if sum(isnan(se_yax))~=length(se_yax)
    h10.patch.FaceColor = color{end}; h10.mainLine.Color = color{end}; h10.edge(1).Color = color{end};
    h10.edge(2).Color=color{end};
end

[r,c] = size(SO_allmouse_dop);
nlay  = length(grabda_rows);
SO_dop_mouse_split   = permute(reshape(SO_allmouse_dop',[c,r/nlay,nlay]),[3,2,1]);
% SO_dop_mouse_split=(reshape(SO_allmouse_dop,[6,size(SO_allmouse_dop,1)/6,size(SO_allmouse_dop,2)]));

allwor=1:length(workspaces);
ch=allwor'.*ones(size(allmouse_dop{1,1}));
id_mat=cellfun(@(x) ~isempty(x),allmouse_dop{1,1});
id_mat=id_mat.*ch;
id_mat=id_mat(:,1:end-1);
id_mat(find(id_mat==0))=NaN;
[r c]=find(isnan(id_mat)); id_mat(r,c-1)=NaN;
mid_dop=id_mat(1:length(grabda_rows),:); mid_dop=mid_dop';mid_dop=mid_dop(:)';
mid_dopidz=mid_dop(find(~isnan(mid_dop)));
mouse_idz_withoutSO_dop=repelem(mid_dopidz,4);
mouse_idz_SO_dop=repelem(1:length(grabda_rows),4);


[r,c] = size(withoutSO_allmouse_dop);
nlay  = size(mid_dopidz,2);
withoutSO_dop_mouse_split   = permute(reshape(withoutSO_allmouse_dop',[c,r/nlay,nlay]),[3,2,1]);

% withoutSO_dop_mouse_split=(reshape(withoutSO_allmouse_dop,[size(mid_dopidz,2),4,size(withoutSO_allmouse_dop,2)]));


yax=SO_allmouse_mut;
se_yax=nanstd(yax,1)./sqrt(size(yax,1));
yax=nanmean(yax);
hold on, ;h11 = shadedErrorBar(xax,yax,se_yax,[],1);
if sum(isnan(se_yax))~=length(se_yax)
    h11.patch.FaceColor = color{end}; h11.mainLine.Color = color{end}; h11.edge(1).Color = color{end};
    h11.edge(2).Color=color{end}; h11.mainLine.LineStyle='--'; h11.edge(2).LineStyle='--';
end
SO_mut_mouse_split=(reshape(SO_allmouse_mut,[3,size(SO_allmouse_mut,1)/3,size(SO_allmouse_mut,2)]));
mid_mut=id_mat(grabdamut_rows(1):grabdamut_rows(end),:); mid_mut=mid_mut';mid_mut=mid_mut(:)';
mid_mutidz=mid_mut(find(~isnan(mid_mut)));
mouse_idz_withoutSO_mut=repelem(mid_mutidz,4);
mouse_idz_SO_mut=repelem(grabdamut_rows(1):grabdamut_rows(end),4);



withoutSO_mut_mouse_split=(reshape(withoutSO_allmouse_mut,[size(mid_mutidz,2),4,size(withoutSO_allmouse_mut,2)]));


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

%%%significance
%%mutant without SO

xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes;
pst=find(xax>timeforpre(1)&xax<=timeforpre(2)); pst=find(xax>timeforpost(1)&xax<=timeforpost(2));

mtm_vals_dop=mean(withoutSO_dop_mouse_split(:,:,pst),3);
mtm_vals_dop=mtm_vals_dop';mtm_vals_dop=mtm_vals_dop(:)';
mtm_vals_mut=mean(withoutSO_mut_mouse_split(:,:,pst),3);
mtm_vals_mut=mtm_vals_mut';mtm_vals_mut=mtm_vals_mut(:)';

%%%%plotting significance
coloridx={[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880],[0.3010 0.7450 0.9330],...
    [0.6350 0.0780 0.1840],[1 0 0],[0 1 0]};


significance_plot_days(4,p,mtm_vals_mut',mtm_vals_dop',ylims,color,1,coloridx,mouse_idz_withoutSO_mut,mouse_idz_withoutSO_dop,workspaces)

%%%%




%%%dop with SO
%%%
mtm_vals_dop=mean(SO_dop_mouse_split(:,:,pst),3);
mtm_vals_dop=mtm_vals_dop';mtm_vals_dop=mtm_vals_dop(:)';
mtm_vals_mut=mean(SO_mut_mouse_split(:,:,pst),3);
mtm_vals_mut=mtm_vals_mut';mtm_vals_mut=mtm_vals_mut(:)';

%%%%plotting significance

significance_plot_days(3,p,mtm_vals_mut',mtm_vals_dop',ylims,color,4,coloridx,mouse_idz_SO_mut,mouse_idz_SO_dop,workspaces)
%%%%

subplot(4,2,((p-1)*4+2)); hold on
patchy = patch([timeforpost fliplr(timeforpost)], [ylims(1) ylims(1) ylims(2) ylims(2)],[0.85 0.85 0.85]);
patchy.EdgeAlpha = 0;
patchy.FaceAlpha - 0.5;
xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes
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
    h11.patch.FaceColor = color{1}; h11.mainLine.Color = color{1}; h11.edge(1).Color = color{1};
    h11.edge(2).Color=color{1}; h11.mainLine.LineStyle='--';h11.edge(2).LineStyle='--';
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





