% zahra's opto analysis for vip
clear all; close all
% mouse_name = "e218";
% days = [34,35,36,37,38,39,40,41,44,47,50];
% cells_to_plot = {[141, 17,20,7]+1, [453,63,26,38]+1, [111,41,65,2]+1, [72,41,27,14]+1,...
%     [301 17 13 320]+1, [98 33 17 3]+1, [92 20 17 26]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
%      [49 47 6 37]+1, [434,19,77,5]+1}; % indices of red cells from suite2p per day
mouse_name = "e218";
days = [35,38,41,44,47,50];
cells_to_plot = {[453,63,26,38]+1,...
    [301 17 13 320]+1, [17, 23, 36, 10]+1, [6, 114, 11, 24]+1,...
    [49 47 6 37]+1, [434,19,77,5]+1}; % indices of red cells from suite2p per day

src = "X:\vipcre";
dffs_cp_dys = {};
% get dff per red cell and correlate with success rate
% opto_ep = [2 3 2 2 2 2 2 3 2 3 2];
opto_ep = [3 2 3 2 3 2];
y = []; x = [];

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
    dffs_cp_dys{dyind} = dffs_cp; % collect opto trace per day    
    % get dff per red cell and correlate with success rate
    optorng = eps(opto_ep(dyind)):eps(opto_ep(dyind)+1);
    [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum(optorng),rewards(optorng));
    successrate = (success/total_trials)*100;
    y(dyind) = successrate;
    dffs = dffs_cp_dys{dyind};
    % for 4 cells, get average dff (ratio compared to prev epoch)
%     meandff = median(dffs{4}{1})/median(dffs{4}{2});
    meandff = median([mean(dffs{1}{1})/mean(dffs{1}{2}); ...
        mean(dffs{2}{1})/mean(dffs{2}{2}); mean(dffs{3}{1})/mean(dffs{3}{2}); mean(dffs{4}{1})/mean(dffs{4}{2})], 'omitnan');
    x(dyind) = meandff;
    dyind = dyind+1;
end
%%

fig = figure('Renderer', 'painters'); plot(x,y/100, 'ko')
mdl = fitlm(x,y/100);
ylabel('Fraction of Successful Trials')
xlabel('dFF (LED on) / dFF (LED off)')
title(sprintf('r = %f', sqrt(mdl.Rsquared.Ordinary)))
box off