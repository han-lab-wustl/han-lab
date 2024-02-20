clear all; close all
mice = {"e216", "e218"};
% mice = {"e186"};%{"e201"};
% dys_s = {[2:5,31,32,33,36, 38 40 41]};%{[52:65]};
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:52, 55:59 60],...
    [20,21,22,23,35,36,37,38,39,40,41,42,43,44,45,47 48 49 50 51 52]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_ep_s = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 0 1 2 0 1 3],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3, 0 1 2, 0 3 0 1 2 0 1]};
% opto_ep_s = {[-1 -1 -1 -1 2 3 2 2 2 2 2]};%{[-1 -1 -1 2 3 0 2 3 0 2 3 0 2 3]};
src = "X:\vipcre";
% src = "Y:\analysis\fmats";
% ntrials = 8; % get licks for last n trials
transient_stats_m = {}; % 1 - success opto, 2 - success prev opto, 3 - success postopto, 4 - fails opto, 5 - fails prev opto, 6 - fails prev opto
% other days besides opto : 1 - success, 2 - fails
for m=1:length(mice)
    dys = dys_s{m};
    mouse_name = mice{m};
    opto_ep = opto_ep_s{m};
    epind = 1; % for indexing
    transient_stats = {};
for dy=dys
    fprintf("*******processing day %i*******\n", dy)
    % daypth = dir(fullfile(src(m), mouse_name, 'days', sprintf('%s_day%03d*.mat', mouse_name, dy)));    
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
        'ybinned', 'timedFF', 'Fc3', 'iscell', 'putative_pcs', 'stat', 'VR');
    try
        gainf = 1/VR.scalingFACTOR;
    catch
        gainf = 3/2; % 3/2 VS. 1; in this pipeline the gain is multiplied everywhere
    end
    optoep = opto_ep(epind);
    % pre reward ypos
    pcs = reshape(cell2mat(putative_pcs), [length(putative_pcs{1}), length(putative_pcs)]);
    pc = logical(iscell(:,1));
    [~,bordercells] = remove_border_cells_all_cells(stat, Fc3);        
    bordercells_pc = bordercells(pc); % mask border cells
    fc3_pc = Fc3(eprng,pc); % only iscell
    fc3_pc = fc3_pc(:,~bordercells_pc); % remove border cells
    fc3_pc = fc3_pc(:, any(pcs,2)); % apply place cell filter, if a cell is considered a place cell in any ep!!    
    [mean_length_of_transients_per_cell_opto,....
    mean_length_of_transients_per_cell_ctrl, auc_transients_per_cell_opto,...
    auc_transients_per_cell_ctrl, peak_transients_per_cell_opto, ...
    peak_transients_per_cell_ctrl] = get_transient_stats(changeRewLoc,gainf,ybinned, putative_pcs, fc3_pc, iscell,...
    stat, optoep);
    transient_stats{epind} = [mean(mean_length_of_transients_per_cell_ctrl, 'omitnan'), ....
        mean(mean_length_of_transients_per_cell_opto, 'omitnan'), mean(auc_transients_per_cell_ctrl,'omitnan'),...
        mean(auc_transients_per_cell_opto, 'omitnan'), mean(peak_transients_per_cell_ctrl,'omitnan'),...
        mean(peak_transients_per_cell_opto, 'omitnan')];
    epind=epind+1;
end
transient_stats_m{m} = transient_stats;
end
%%
optodys1 = find(opto_ep_s{1}==-1); optodys2= find(opto_ep_s{2}==-1); t1 = transient_stats_m{1}; 
t2 = transient_stats_m{2};
% ctrl = [cell2mat(cellfun(@(x) x(1), t1(optodys1), 'UniformOutput', false)),...
%     cell2mat(cellfun(@(x) x(1), t2(optodys2), 'UniformOutput', false))];
% opto = [cell2mat(cellfun(@(x) x(2), t1(optodys1), 'UniformOutput', false)),...
%     cell2mat(cellfun(@(x) x(2), t2(optodys2), 'UniformOutput', false))];
% y = [ctrl, opto];
% x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
% figure;
% bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
% swarmchart(x,y, 'ko')
% ylabel('mean length of transients (s)')
% xticklabels(["previous epoch", "opto epoch"])
% [~,p,t,stats]=ttest(ctrl, opto)
% title(sprintf('p = %f', p))
% % mean across animals
% ctrl = [mean(cell2mat(cellfun(@(x) x(1), t1(optodys1), 'UniformOutput', false))),...
%     mean(cell2mat(cellfun(@(x) x(1), t2(optodys2), 'UniformOutput', false)))];
% opto = [mean(cell2mat(cellfun(@(x) x(2), t1(optodys1), 'UniformOutput', false))),...
%     mean(cell2mat(cellfun(@(x) x(2), t2(optodys2), 'UniformOutput', false)))];
% y = [ctrl, opto];
% x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
% figure;
% bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
% swarmchart(x,y, 'ko')
% ylabel('mean length of transients (s)')
% xticklabels(["previous epoch", "opto epoch"])
% [~,p,t,stats]=ttest(ctrl, opto);
% title(sprintf('p = %f', p))

%%
% mean across animals
% ctrl = [mean(cell2mat(cellfun(@(x) x(3), t1(optodys1), 'UniformOutput', false))),...
%     mean(cell2mat(cellfun(@(x) x(3), t2(optodys2), 'UniformOutput', false)))];
% opto = [mean(cell2mat(cellfun(@(x) x(4), t1(optodys1), 'UniformOutput', false))),...
%     mean(cell2mat(cellfun(@(x) x(4), t2(optodys2), 'UniformOutput', false)))];
ctrl = [cell2mat(cellfun(@(x) x(3), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(3), t2(optodys2), 'UniformOutput', false))];
opto = [cell2mat(cellfun(@(x) x(4), t1(optodys1), 'UniformOutput', false)),...
    cell2mat(cellfun(@(x) x(4), t2(optodys2), 'UniformOutput', false))];
y = [ctrl, opto];
x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
figure;
bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
swarmchart(x,y, 'ko')
yerr = {ctrl, opto};
err = [];
for i=1:length(yerr)
    err(i) =(std(yerr{i},'omitnan')/sqrt(size(yerr{i},2))); 
end
er = errorbar([1 2],[mean(ctrl), mean(opto)],err);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

ylabel('Mean AUC transient')
xticklabels(["LED off", "LED on"])
[~,p,t,stats]=ttest(ctrl, opto)
title(sprintf('p = %f', p))
box off

% %% peak fc3
% ctrl = [cell2mat(cellfun(@(x) x(5), t1(optodys1), 'UniformOutput', false)),...
%     cell2mat(cellfun(@(x) x(5), t2(optodys2), 'UniformOutput', false))];
% opto = [cell2mat(cellfun(@(x) x(6), t1(optodys1), 'UniformOutput', false)),...
%     cell2mat(cellfun(@(x) x(6), t2(optodys2), 'UniformOutput', false))];
% y = [ctrl, opto];
% x = [repelem(1, length(ctrl)), repelem(2, length(opto))];
% figure;
% bar([mean(ctrl), mean(opto)], 'FaceColor', 'w'); hold on
% swarmchart(x,y, 'ko')
% ylabel('mean of transient')
% xticklabels(["previous epoch", "opto epoch"])
% [~,p,t,stats]=ttest(ctrl, opto)
% title(sprintf('p = %f', p))
