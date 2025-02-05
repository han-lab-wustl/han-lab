% zahra's opto analysis for vip
clear all; close all
% mouse_name = "e218";
% days = [34,35,36,37,38,39,40,41,44,47,50];
% cells_to_plot = {[141, 17,20,7]+1, [453,63,26,38]+1, [111,41,65,2]+1, [72,41,27,14]+1,...
%     [301 17 13 320]+1, [98 33 17 3]+1, [92 20 17 26]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
%      [49 47 6 37]+1, [434,19,77,5]+1}; % indices of red cells from suite2p per day
mice = {"e216", "e217", "e218"};
% mice = {"e186"};%{"e201"};
% dys_s = {[2:5,31,32,33,36, 38 40 41]};%{[52:65]};
dys_s = {[37 41 57 60],[14, 26, 27],...
    [35,38,41,44,47,50]};
opto_ep_s = {[2 3 2 3],[2,3,3],...
    [3 2 3 2 3 2]};
cells_to_plot_s = {{135+1,1655+1,780,2356+1},{16+1,6+1,9+1} ...
    {[453,63,26,38]+1,...
    [301 17 13 320]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
    [49 47 6 37]+1, [434,19,77,5]+1}}; % indices of red cells from suite2p per day
src = "X:\vipcre";
dffs_cp_dys = {};
% get dff per red cell and correlate with success rate
% opto_ep = [2 3 2 2 2 2 2 3 2 3 2];
y = []; x = [];
mind=1; % day and mouse index
%%
for m=1:length(mice)
    mouse_name = mice{m};
    days = dys_s{m};
    cells_to_plot = cells_to_plot_s{m};
    opto_ep = opto_ep_s{m};
    dyind = 1;
for dy=days
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'dFF', 'changeRewLoc', 'VR', 'ybinned', 'forwardvel', ...
        'timedFF', 'rewards', 'licks', 'trialnum');
    disp(fullfile(daypth.folder,daypth.name))
    % plot with behavior with red cells
    %%
    grayColor = [.7 .7 .7];
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    gainf = 1/VR.scalingFACTOR;
    rewloc = changeRewLoc(changeRewLoc>0)*gainf;
    rewsize = VR.settings.rewardZone*gainf;
    ypos = ybinned*(gainf);
    velocity = forwardvel;
    figure; %fig = figure('Renderer', 'painters');
%     subplot(1,2,1)
    iind = 1;
    scatter(1:length(ypos), ypos, 2, 'filled', 'MarkerFaceColor', grayColor); hold on;
    plot(find(licks),ypos(find(licks)), ...
        'r.', 'MarkerSize',8)
    for mm = 1:length(eps)-1 %the rectangle indicating the reward location, overlaps the probe trials referring to the previous reward location
        rectangle('position',[eps(mm) rewloc(mm)-rewsize/2 ...
            eps(mm+1)-eps(mm) rewsize],'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
    end
    plot(find(rewards==0.5),ypos(find(rewards==0.5)),'b*', ...
        'MarkerSize',10)
    ylabel("Track Position (cm)")
    xticks([0:10000:length(timedFF)])
    tic = ceil([timedFF(1:10000:end) max(timedFF)]/60);
    xticklabels(tic)
    xlabel("Time (minutes)")
    ylim([0 270])
    xlim([0 length(rewards)])
    yticks([0:90:270])
    yticklabels([0:90:270])
    dffs_cp = {};
    indtemp = 1;
%     subplot(1,2,2)
    figure;
    for cp=cells_to_plot{dyind}              
        plot((dFF(:,cp)+iind)); hold on
%         ylim([0 20])
        iind = iind + 5;
        % compare to prev epoch
        rngopto = eps(opto_ep(dyind)):eps(opto_ep(dyind)+1);
        rngpreopto = eps(opto_ep(dyind)-1):eps(opto_ep(dyind));
        yposopto = ypos(rngopto);
        ypospreopto = ypos(rngpreopto);
        yposoptomask = yposopto<rewloc(opto_ep(dyind))-rewsize/2; % get rng pre reward
        ypospreoptomask = ypospreopto<rewloc(opto_ep(dyind)-1)-rewsize/2; % get rng pre reward
        dffopto = dFF(rngopto,:); dffpreopto = dFF(rngpreopto,:);
        dffs_cp{indtemp} = {dffopto(yposoptomask,cp), dffpreopto(ypospreoptomask,cp)};
        indtemp = indtemp + 1;
    end
    xticks([0:5000:length(timedFF)])
    tic = ([timedFF(1:5000:end) max(timedFF)]/60);
    xticklabels(tic)
    xlabel("Time (minutes)")
    %%
    dffs_cp_dys{mind} = dffs_cp; % collect opto trace per day    
    % get dff per red cell and correlate with success rate
    optorng = eps(opto_ep(dyind)):eps(opto_ep(dyind)+1);
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum(optorng),rewards(optorng));
    successrate = success/total_trials;
    y(mind) = successrate;
    dffs = dffs_cp_dys{mind};
    % for 4 cells, get average dff (ratio compared to prev epoch), only
    meandff = cell2mat(cellfun(@(x) mean(x{1}, 'omitnan'), dffs, ...
        'UniformOutput', false))/cell2mat(cellfun(@(x) mean(x{2}, 'omitnan'), dffs, 'UniformOutput', false));
    % meandff = median([mean(dffs{1}{1})/mean(dffs{1}{2}); ...
    %     mean(dffs{2}{1})/mean(dffs{2}{2}); mean(dffs{3}{1})/mean(dffs{3}{2}); mean(dffs{4}{1})/mean(dffs{4}{2})], 'omitnan');
    x(mind) = meandff;
    dyind = dyind+1;
    mind = mind+1;
end
end
%%
meandff_opto = cell2mat(cellfun(@(x) mean(x{1}{1}, 'omitnan'), dffs_cp_dys, ...
        'UniformOutput', false));
meandff_prev = cell2mat(cellfun(@(x) mean(x{1}{2}, 'omitnan'), dffs_cp_dys, ...
        'UniformOutput', false));

meandff_opto = cell2mat(cellfun(@(x) mean(x{1}{1}), dffs_cp_dys, ...
        'UniformOutput', false));
meandff_prev = cell2mat(cellfun(@(x) mean(x{1}{2}), dffs_cp_dys, ...
        'UniformOutput', false));

fig = figure('Renderer', 'painters'); 
bar([mean(meandff_prev), mean(meandff_opto)],'FaceColor', 'w','LineWidth',2); hold on

x_ = [ones(1,size(meandff_prev,2)), ones(1,size(meandff_opto,2))*2];
y_ = [meandff_prev, meandff_opto];
swarmchart(x_,y_, 'ko','LineWidth',2)
yerr = {meandff_prev, meandff_opto};
err = [];
for i=1:length(yerr)
    err(i) =(std(yerr{i},'omitnan')/sqrt(size(yerr{i},2))); 
end
er = errorbar([1 2],[mean(meandff_prev), mean(meandff_opto)],err,'LineWidth',2);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
ylabel('Mean dFF')

[h,p1,i,stats] = ttest(meandff_prev,meandff_opto);

%%
% correlate the vip activity with success rate
x = x(x<2); y = y(x<2);
fig = figure('Renderer', 'painters'); plot(x,y, 'ko','MarkerSize',10, 'LineWidth',2); hold on
% plot(x,y)
% Get coefficients of a line fit through the data.
coefficients = polyfit(x, y, 1);
% Create a new x axis with exactly 1000 points (or whatever you want).
xFit = linspace(min(x), max(x), 1000);
% Get the estimated yFit value for each of those 1000 new x locations.
yFit = polyval(coefficients , xFit);
% Plot everything.
plot(xFit, yFit, 'k-', 'LineWidth', 2); % Plot fitted line.

mdl = fitlm(x,y);
ylabel('Success Rate')
xlabel('dFF (LED on) / dFF (LED off)')
title(sprintf('r = %f, p = %f', sqrt(mdl.Rsquared.Ordinary), mdl.Coefficients.pValue(2)))
box off