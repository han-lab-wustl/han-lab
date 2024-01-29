differentlicks = {'success_pre_rew_far_licksind',... % naming all the different lick categories
    'success_pre_rew_near_licksind',...
    'success_in_rew_licksind_pre_us'...
    'success_in_rew_licksind_post_us'...
    'success_post_rew'...
    'probe_pre_rew_far_licksind'...
    'probe_pre_rew_near_licksind'...
    'probe_in_rew_licksind'...
    'probe_post_rew'...
    'failed_pre_rew_far_licksind'...
    'failed_pre_rew_near_licksind'...
    'failed_post_rew'};

dopvars = {};
speedvars = {};
speedtimevars = {};
titles = {};

for d = 1:length(differentlicks) % for each category setup a storing variable to compile across days
    temp1 = differentlicks{d};
    temp = temp1;
    [outstart,outstop] = regexp(temp,'_licksind');
    if ~isempty(outstart) % remove mention of licks ind for shorter precise variable names
        temp(outstart:outstop) = [];
    end
    dopvars = [dopvars {['ALL_dopnorm_allday_' temp]}]; %initialize pre window normalized dopamine variables
    speedvars = [speedvars {['ALL_Spd_allday_' temp] }]; % initialize speed variable
    speedtimevars = [speedtimevars {['ALL_Spdtime_allday_' temp] }]; % initialize speed variable
    titles = [titles {temp}];
end
%%

ylims = ([0.98 1.02])
 speedtransf = [0 25];
 doptransf = [0.981 0.99]; 
%  speedvars ={'roe_success_perimov_no_reward','roi_roe_allsuc_mov_no_reward_no_lick','roi_roe_allsuc_mov_no_reward_lick'};
% 
%  dopvars = {'roi_dop_allsuc_mov_no_reward','roi_dop_allsuc_mov_no_reward_no_lick','roi_dop_allsuc_mov_no_reward_lick'};

    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
figure;
for vars = 1:length(dopvars)
 speedvariable = eval(speedvars{vars});
 dopvariable = eval(dopvars{vars});
 dopvariable = dopvariable(:,1);
 speedvariable = speedvariable(:,1);
 if sum(cellfun(@isempty,speedvariable))>0
 speedvariable(cellfun(@isempty,speedvariable)) = {NaN(length(-5:0.01:5),2,1)};
 end
%  speedvariable = cellfun(@(x) squeeze(nanmean(x(:,:,1),2)),speedvariable,'UniformOutput',0);
 speedtimevariable = eval(speedtimevars{vars}); 
 speedtimevariable = speedtimevariable(:,1);
  if sum(cellfun(@isempty,speedtimevariable))>0
 speedtimevariable(cellfun(@isempty,speedtimevariable)) = {repmat((-5:0.01:5)',1,2,1)};
  end
%  speedtimevariable = cellfun(@(x) squeeze(nanmean(x(:,:,1),2)),speedtimevariable,'UniformOutput',0);
dopvariable(cellfun(@isempty,dopvariable)) = {NaN(78,2,4)};
 dopvariable = permute(cell2mat(reshape(cellfun(@(x) squeeze(nanmean(x,2)),dopvariable,'UniformOutput',0),1,1,[])),[3 2 1]);
%  speedvariable = permute(cell2mat(reshape(cellfun(@(x) squeeze(nanmean(x(,2)),speedvariable,'UniformOutput',0),1,1,[])),[3 2 1]);

for d = 1:length(speedvariable)
    newvar = [];
        timevar = discretize(reshape(speedtimevariable{d}(:,:,1),1,[]),-5:0.03:5);
        speedvar = reshape(speedvariable{d}(:,:,1),1,[]);
        speedvar(isnan(timevar)) = [];
        timevar(isnan(timevar)) = [];
        newvar = accumarray(timevar',speedvar',[],@nanmean);
        speedvariable{d} = newvar;
end
speedvariable = permute(cell2mat(reshape(speedvariable,1,1,[])),[3,2,1]);

    subplot(2,length(dopvars),vars+length(dopvars))
for r = 1:size(dopvariable,2)
    %late days
    yax1 = nanmean(dopvariable(max([1 size(dopvariable,1)-3]):end,r,:));
    seyax1 = nanstd((dopvariable(max([1 size(dopvariable,1)-3]):end,r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
    h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
        if vars == 1
            speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,:)));
        else
    speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,r,:)));
        end
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
%     doptransf = [0.976 0.989];
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
%     yyaxis right

    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars},'Interpreter','none')
ylabel('Late Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)

 subplot(2,length(dopvars),vars)
for r = 1:size(dopvariable,2)
    %early days
    yax1 = nanmean(dopvariable(1:min([4 size(dopvariable,1)]),r,:));
    seyax1 = nanstd((dopvariable(1:min([4 size(dopvariable,1)]),r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(1:min([4 size(dopvariable,1)]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
    
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars},'Interpreter','none')
ylabel('Early Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)


end
