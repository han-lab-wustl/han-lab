% Zahra
% analysis of VR variables on opto days
% based on Zahra's folder structure
clear all; close all;
animals = {'e200', 'e201'};
drives = {'Y:\sstcre_imaging', 'Z:\sstcre_imaging'};
days = {[65:69], [55:68]}; % opto sequence, ep2 5 trials, ep3 5 trials, control
conditions = {'ep2', 'ep3', 'control'};
opts = detectImportOptions('Y:\data_organization.csv');
% assumes VR align has been run
addpath(fullfile(pwd, "hidden_reward_zone_task\behavior"));

for i=1:length(animals)
    successes = {}; fails = {};speeds = {}; totals = {};
    
    for d=1:length(days{i})
        fmatfl = dir(fullfile(drives{i}, animals{i}, string(days{i}(d)), "behavior", "vr\*.mat")); 
        condind = rem(d,3); 
        if condind == 0
            condind = length(conditions);
        end
        [speed_5trialsep, speed_ep, speed_5trialsep1, speed_ep1, speed_5trialsep2, speed_ep2, ...
            speed_5trialsep3, speed_ep3, s, f, t] = get_mean_speed_success_failure_trials(fullfile(fmatfl.folder,fmatfl.name),conditions{condind});
        successes{d} = s;
        fails{d} = f;     
        totals{d} = t;
        speeds{d} = [speed_5trialsep, speed_ep, speed_5trialsep1, speed_ep1, ...
            speed_5trialsep2, speed_ep2, speed_5trialsep3, speed_ep3];
    end

    % plot trial performance as bar graph    
    for j=1:length(conditions) % iterate through conditions
        % plot speeds across conditions
        sp = speeds(j:3:end); sc = successes(j:3:end); fl = fails(j:3:end); tr = totals(j:3:end);
        % init
        optotrials = {}; restofep = {}; spep1 = {}; spep2 = {}; spep3 = {}; spep25trials = {};
        spep35trails = {}; spep15trails = {};
        for ii=1:length(sp) % iterature thorugh n
            optotrials{ii} = sp{ii}(1);
            restofep{ii} = sp{ii}(2);
            spep15trials{ii} = sp{ii}(3);
            spep1{ii} = sp{ii}(4);
            spep25trials{ii} = sp{ii}(5);
            spep2{ii} = sp{ii}(6);
            spep35trials{ii} = sp{ii}(7);
            spep3{ii} = sp{ii}(8);
            
        end
        figure;
        % automatically segments diff conditions
        bar([mean(cell2mat(cellfun(@(x) x{1}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{1}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{2}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{2}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{3}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{3}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{4}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{4}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{5}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{5}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{6}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{6}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{7}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{7}(1), tr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{8}(1), sc, 'UniformOutput', false))/cell2mat(cellfun(@(x) x{8}(1), tr, 'UniformOutput', false)), 'omitnan')]','grouped','FaceColor','flat');
        hold on
        plot(1,cell2mat(cellfun(@(x) x{1}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{1}(1), tr, 'UniformOutput', false)),'ok')
        plot(2,cell2mat(cellfun(@(x) x{2}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{2}(1), tr, 'UniformOutput', false)),'ok')
        plot(3,cell2mat(cellfun(@(x) x{3}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{3}(1), tr, 'UniformOutput', false)),'ok')
        plot(4,cell2mat(cellfun(@(x) x{4}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{4}(1), tr, 'UniformOutput', false)),'ok')
        plot(5,cell2mat(cellfun(@(x) x{5}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{5}(1), tr, 'UniformOutput', false)),'ok')
        plot(6, cell2mat(cellfun(@(x) x{6}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{6}(1), tr, 'UniformOutput', false)), 'ok')
        plot(7, cell2mat(cellfun(@(x) x{7}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{7}(1), tr, 'UniformOutput', false)), 'ok')
        plot(8, cell2mat(cellfun(@(x) x{8}(1), sc, 'UniformOutput', false))./cell2mat(cellfun(@(x) x{8}(1), tr, 'UniformOutput', false)), 'ok')
        
        xticklabels(["optotrials", "rest of opto epoch", "1st 5 trials epoch 1", ...
            "rest of epoch 1", "1st 5 trials epoch 2", "rest of epoch 2", "1st 5 trials epoch 3", ...
            "rest of epoch 3"])
        ylabel("% successful trials")
        title(sprintf("animal %s, %s", animals{i}, conditions{j}))

        % plot mean speed
        figure;
        bar([mean([optotrials{:}], 'omitnan');mean([restofep{:}], 'omitnan'); ...
            mean([spep15trials{:}], 'omitnan');mean([spep1{:}], 'omitnan'); ...
            mean([spep25trials{:}], 'omitnan'); mean([spep2{:}], 'omitnan'); ...
            mean([spep35trials{:}], 'omitnan'); mean([spep3{:}], 'omitnan')]','grouped','FaceColor','flat');
        hold on
        plot(1,[optotrials{:}],'ok')
        plot(2,[restofep{:}],'ok')
        plot(3,[spep15trials{:}],'ok')
        plot(4,[spep1{:}],'ok')
        plot(5,[spep25trials{:}],'ok')
        plot(6, [spep2{:}], 'ok')
        plot(7, [spep35trials{:}], 'ok')
        plot(8, [spep3{:}], 'ok')
        
        xticklabels(["optotrials", "rest of opto epoch", "1st 5 trials epoch 1", ...
            "rest of epoch 1", "1st 5 trials epoch 2", "rest of epoch 2", "1st 5 trials epoch 3", ...
            "rest of epoch 3"])        
        ylabel("mean speed")
        title(sprintf("animal %s, %s", animals{i}, conditions{j}))

    end
end