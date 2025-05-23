% get mean dff per cell in opto vs. pre opto epochs
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before reward
% TODO lick rate outside rew zone
clear all; close all
mice = {"e216", "e218"};
% dys = [20,21,22,23,35,36,37,38,39,40,41,42,43,44,45,47 48 49 50 51 52];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:52, 55:59 60],...
    [20,21,22,23,35,36,37,38,39,40,41,42,43,44,45,47 48 49 50 51 52]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
% opto_ep = [-1 -1 -1 -1];
% opto_ep = [-1 -1 -1 -1,3 0 1 2 0 1 3, 0 1 2, 0 3 0 1 2 0 1]; 
opto_ep_s = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 0 1 2 0 1 3],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3, 0 1 2, 0 3 0 1 2 0 1]};
src = "X:\vipcre";
bin_size = 2; % cm
% ntrials = 8; % get licks for last n trials
dffs_m = {}; % 1 - success opto, 2 - success prev opto, 3 - success postopto, 4 - fails opto, 5 - fails prev opto, 6 - fails prev opto
% other days besides opto : 1 - success, 2 - fails
for m=1:length(mice)
    dys = dys_s{m};
    mouse_name = mice{m};
    opto_ep = opto_ep_s{m};
    epind = 1; % for indexing
    dffs = {};
for dy=dys
    fprintf("*******processing day %i*******\n", dy)
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
        'ybinned', 'timedFF', 'dFF', 'iscell', 'putative_pcs', 'stat', 'VR');
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]); % place cell bool
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    nbins = track_length/bin_size;
    ybinned = ybinned/VR.scalingFACTOR;
    rewlocs = changeRewLoc(changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    if opto_ep(epind)==3
        optoep = opto_ep(epind);
        [dff_opto_success, dff_opto_fails, dff_prevopto_success, ...
        dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, trialnum, ...
        rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs);
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
                dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};
        
    elseif opto_ep(epind)==2
        optoep = opto_ep(epind);
        [dff_opto_success, dff_opto_fails, dff_prevopto_success, ...
        dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, trialnum, ...
        rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs);
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
                dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};

    elseif opto_ep(epind)==-1 % just pre opto days
        optoep = 2; % specify fake opto ep to get vars
        [dff_opto_success, dff_opto_fails, dff_prevopto_success,dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, ...
        trialnum, rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs);
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
                dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};
        
    elseif opto_ep(epind)==0  % intermediate control days 1
        optoep = 2; % specify fake opto ep to get vars
        [dff_opto_success, dff_opto_fails, dff_prevopto_success, ...
        dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, trialnum, ...
        rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs);
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
                dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};
        
    elseif opto_ep(epind)==1  % intermediate control days 2
        optoep = 2; % specify fake opto ep to get vars
        [dff_opto_success, dff_opto_fails, dff_prevopto_success, ...
        dff_prevopto_fails,dff_postopto_success,dff_postopto_fails] = get_opto_dff(eps, optoep, epind, trialnum, ...
        rewards, licks, ybinned, rewlocs, iscell, stat, dFF, pcs);
        dffs{epind} = {dff_prevopto_success, dff_opto_success, dff_postopto_success, ...
                dff_prevopto_fails, dff_opto_fails, dff_postopto_fails};
    end    
    epind = epind+1;
end
dffs_m{epind} = dffs;
end

%%
% 1 - success opto, 2 - success prev opto, 3 - success postopto, 4 - fails opto, 5 - fails prev opto, 6 - fails POST opto
% other days besides opto : 1 - success, 2 - fails
% 
% mouse_name = "e218";
% dys = [20,21,22,23,35,36,37,38,39,40,41,42,43,44,45,47 48 49 50 51];
% % experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% % day2=1
% opto_ep = [-1 -1 -1 -1,3 0 1 2 0 1 3, 0 1 2, 0 3 0 1 2 0]; 
% plot certain days of opto vs. ctrl
% pairwise plots
% optodys = find(opto_ep==2);
% 
% for dy=1:length(optodys)
%     optoday = dffs{optodys(dy)};    
%     optodayprev_fails = mean(optoday{5},1,'omitnan'); % fails
%     optodayopto_fails = mean(optoday{4},1,'omitnan');
%     figure; subplot(1,2,1); plot(1,optodayprev_fails,'ko'); hold on; plot(2,optodayopto_fails,'ro'); xlim([0 3])
%     for ii=1:size(optodayopto_fails,2)
%         plot([1, 2],[optodayprev_fails(ii),optodayopto_fails(ii)], 'k'); hold on % pairwise plots
%     end
%     [h,p,i,stats] =ttest(optodayprev_fails,optodayopto_fails); % paired
%     ylabel('dFF')
%     xticks(0:3); xticklabels({NaN, 'previous epoch', 'opto epoch', NaN})
%     title(sprintf('failed trials, day %i, opto ep %i \n paired t-test p-val: %03d \n t-stat: %03d', ...
%         dys(optodys(dy)), opto_ep(optodys(dy)), p, stats.tstat))
% 
%     optodayprev_suc = mean(optoday{2},1,'omitnan'); % success
%     optodayopto_suc = mean(optoday{1},1,'omitnan');
%     subplot(1,2,2); plot(1,optodayprev_suc,'ko'); hold on; plot(2,optodayopto_suc,'ro'); xlim([0 3])
%     for ii=1:size(optodayopto_suc,2)
%         plot([1, 2],[optodayprev_suc(ii),optodayopto_suc(ii)], 'k'); hold on % pairwise plots
%     end
%     [h,p,i,stats] =ttest(optodayprev_suc,optodayopto_suc); % paired
%     ylabel('dFF')
%     xticks(0:3); xticklabels({NaN, 'previous epoch', 'opto epoch', NaN})
%     title(sprintf('successful trials, day %i, opto ep %i \n paired t-test p-val: %03d \n t-stat: %03d', ...
%         dys(optodys(dy)), opto_ep(optodys(dy)), p, stats.tstat))
% end
%%
% plot mean dff of pertubation sessions per session
optodys = find(opto_ep>1);
dffprev = []; dffopto = [];
for dy=1:length(optodys)
    optoday = dffs{optodys(dy)};
    dffprev(dy) = mean(mean(optoday{5},1,'omitnan'),'omitnan');
    dffopto(dy) = mean(mean(optoday{4},1,'omitnan'),'omitnan');
end
figure; 
bar([mean(dffprev) mean(dffopto)], 'FaceColor', 'w'); hold on
x = [repelem(1, length(dffprev)), repelem(2, length(dffopto))];
y = [dffprev, dffopto];
swarmchart(x,y, 'ko')
xticklabels(["previous epoch", "opto epoch"])
ylabel('mean dFF')
[~,p,i,stats] = ttest(dffprev, dffopto);
title(sprintf('failed trials, p=%f', p))

dffprev = []; dffopto = [];
for dy=1:length(optodys)
    optoday = dffs{optodys(dy)};
    dffprev(dy) = mean(mean(optoday{2},1,'omitnan'),'omitnan');
    dffopto(dy) = mean(mean(optoday{1},1,'omitnan'),'omitnan');
end
figure; 
bar([mean(dffprev) mean(dffopto)], 'FaceColor', 'w'); hold on
x = [repelem(1, length(dffprev)), repelem(2, length(dffopto))];
y = [dffprev, dffopto];
swarmchart(x,y, 'ko')
xlim([0 3])
xticklabels(["previous epoch", "opto epoch"])
ylabel('mean dFF')
[~,p,i,stats] = ttest(dffprev, dffopto);
title(sprintf('successful trials, p=%f', p))
