clear all; close all
% mouse_name = "e218";
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
transient_stats_m = {}; % 1 - success opto, 2 - success prev opto, 3 - success postopto, 4 - fails opto, 5 - fails prev opto, 6 - fails prev opto
% other days besides opto : 1 - success, 2 - fails
for m=1:length(mice)
    dys = dys_s{m};
    mouse_name = mice{m};
    opto_ep = opto_ep_s{m};
    epind = 1; % for indexing
    transien_stats = {};
for dy=dys
    fprintf("*******processing day %i*******\n", dy)
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
        'ybinned', 'timedFF', 'Fc3', 'iscell', 'putative_pcs', 'stat', 'VR');
    optoep = opto_ep(epind);
    [mean_length_of_transients_per_cell_opto,....
    mean_length_of_transients_per_cell_ctrl, auc_transients_per_cell_opto,...
    auc_transients_per_cell_ctrl] = get_transient_stats(changeRewLoc, VR,ybinned, putative_pcs, Fc3, iscell,...
    stat, optoep);
    transient_stats{epind} = [mean(mean_length_of_transients_per_cell_ctrl, 'omitnan'), ....
        mean(mean_length_of_transients_per_cell_opto, 'omitnan'), mean(auc_transients_per_cell_ctrl,'omitnan'),...
        mean(auc_transients_per_cell_opto, 'omitnan')];
    epind=epind+1;
end
transient_stats_m{m} = transient_stats;
end
%%
optodys1 = find(opto_ep_s{1}>1); optodys2= find(opto_ep_s{2}>1); t1 = transient_stats_m{1}; 
t2 = transient_stats_m{2};
ctrl = [cell2mat(cellfun(@(x) x(1), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(1), t2(optodys2), 'UniformOutput', false))];
opto = [cell2mat(cellfun(@(x) x(2), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(2), t2(optodys2), 'UniformOutput', false))];
y = [ctrl, opto];
x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
figure;
bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
swarmchart(x,y, 'ko')
ylabel('mean length of transients (s)')
xticklabels(["previous epoch", "opto epoch"])
[~,p,t,stats]=ttest(ctrl, opto)
title(sprintf('p = %f', p))

%%
ctrl = [cell2mat(cellfun(@(x) x(3), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(3), t2(optodys2), 'UniformOutput', false))];
opto = [cell2mat(cellfun(@(x) x(4), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(4), t2(optodys2), 'UniformOutput', false))];
y = [ctrl, opto];
x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
figure;
bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
swarmchart(x,y, 'ko')
ylabel('mean auc transient')
xticklabels(["previous epoch", "opto epoch"])
[~,p,t,stats]=ttest(ctrl, opto)
title(sprintf('p = %f', p))