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
    success_prop = {}; fail_prop = {};speeds = {};
    
    for d=1:length(days{i})
        fmatfl = dir(fullfile(drives{i}, animals{i}, string(days{i}(d)), "behavior", "vr\*.mat")); 
%         fmat = load(fullfile(fmatfl.folder,fmatfl.name));
        [s,f,t] = get_success_failure_trials(fullfile(fmatfl.folder,fmatfl.name),'vrfile');
        condind = rem(d,3); 
        if condind == 0
            condind = length(conditions);
        end
        [speed_5trialsep, speed_ep, speed_ep1, speed_ep2, speed_ep3, speed_5trialsep2, speed_5trialsep3] = get_mean_speed(fullfile(fmatfl.folder,fmatfl.name),conditions{condind});
        success_prop{d} = s/t;
        fail_prop{d} = f/t;     
        speeds{d} = [speed_5trialsep, speed_ep, speed_ep1, speed_ep2,speed_ep3, speed_5trialsep2, speed_5trialsep3];
    end

    % plot trial performance as bar graph    
    x = [success_prop{:}];    
    y = [fail_prop{:}];
    ep2x = x(1:3:end); ep2y = y(1:3:end); % TODO: fix to make conditions modular
    ep3x = x(2:3:end); ep3y = y(2:3:end);
    ctrlx = x(3:3:end); ctrly = y(3:3:end);
    xs = {ep2x, ep3x, ctrlx};
    ys = {ep2y, ep3y, ctrly};
    for j=1:length(ys) % iterate through conditions
        figure;
        bar([mean(ys{j});mean(xs{j})]','grouped','FaceColor','flat');
        hold on
        plot(1,ys{j},'ok')
        plot(2,xs{j},'ok')
        xticklabels(["Fails" "Successes"])
        ylabel("Proportion of trials")
        title(sprintf("animal %s, %s", animals{i}, conditions{j}))
        % plot speeds across conditions
        sp = speeds(j:3:end);
        % init
        optotrials = {}; restofep = {}; spep1 = {}; spep2 = {}; spep3 = {}; spep25trials = {};
        spep35trails = {};
        for ii=1:length(sp) % iterature thorugh n
            optotrials{ii} = sp{ii}(1);
            restofep{ii} = sp{ii}(2);
            spep1{ii} = sp{ii}(3);
            spep2{ii} = sp{ii}(4);
            spep3{ii} = sp{ii}(5);
            spep25trials{ii} = sp{ii}(6);
            spep35trials{ii} = sp{ii}(7);
        end
        figure;
        bar([mean([optotrials{:}]);mean([restofep{:}]);mean([spep1{:}]);mean([spep2{:}]); ...
            mean([spep3{:}]); nanmean([spep25trials{:}]); nanmean([spep35trials{:}])]','grouped','FaceColor','flat');
        hold on
        plot(1,[optotrials{:}],'ok')
        plot(2,[restofep{:}],'ok')
        plot(3,[spep1{:}],'ok')
        plot(4,[spep2{:}],'ok')
        plot(5,[spep3{:}],'ok')
        plot(6, [spep25trials{:}], 'ok')
        plot(7, [spep35trials{:}], 'ok')
        
        xticklabels(["optotrials" "rest opto epoch", "epoch 1", ...
            "epoch 2", "epoch 3", "1st 5 trials epoch 2", "1st 5 trials epoch 3"])
        ylabel("mean ROE")
        title(sprintf("animal %s, %s", animals{i}, conditions{j}))

    end
end