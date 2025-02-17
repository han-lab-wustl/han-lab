% Zahra
% analysis of VR variables on opto days
% based on Zahra's folder structure
clear all; close all;
animals = {'e200', 'e201'};
drives = {'Y:\sstcre_imaging', 'Z:\sstcre_imaging'};
days = {[65:76,78:93], [55:73,75:80, 82:92]}; % opto sequence, ep2 5 trials, ep3 5 trials, control
conditions = {'ep2', 'ep3', 'control'};
cond_days = {{[65    68    71    74    78 81 84], ...
    [66    69    72    75    79 82 85], ....
    [67    70    73    80 83 86]}, ....
    {[55    58    61    64    67 70 73], ....
    [56    59    62    65    68 71 75], ....
    [57    60    63    66    69 72 76]}}; % animal x condition
colors = {'b' 'r' 'k' 'g' 'm' 'c' };

% only works on VR files
addpath(fullfile(pwd, "hidden_reward_zone_task\behavior"));
ur_lrs = {}; r_lrs = {}; scs = {};
for i=1:length(animals)
    successes = {}; fails = {};speeds = {}; totals = {}; ur_lick_rates = {}; 
    r_lick_rates = {}; rewzones = {};
    for d=1:length(days{i})
        fmatfl = dir(fullfile(drives{i}, animals{i}, string(days{i}(d)), "behavior", "vr\*.mat")); 
        % find condition of day
        cond_an = cond_days{i};
        for k=1:length(cond_an)
            if ismember(days{i}(d), cond_an{k})>0
                condind = k;
            end
        end
        [speed_5trialsep, speed_ep, speed_5trialsep1, speed_ep1, speed_5trialsep2, speed_ep2, ...
            speed_5trialsep3, speed_ep3, s, f, t, ur_lick_rate, r_lick_rate, ...
            rewzone] = get_opto_behavior(fullfile(fmatfl.folder,fmatfl.name), ...
            conditions{condind});
        successes{d} = s;
        srs{d} = cell2mat(s)./cell2mat(t);
        fails{d} = f;     
        totals{d} = t;
        ur_lick_rates{d} = ur_lick_rate;
        r_lick_rates{d} = r_lick_rate;
        speeds{d} = [speed_5trialsep, speed_ep, speed_5trialsep1, speed_ep1, ...
            speed_5trialsep2, speed_ep2, speed_5trialsep3, speed_ep3];
        rewzones{d} = rewzone;
    end
    ur_lrs{i} = ur_lick_rates;
    r_lrs{i} = r_lick_rates;
    scs{i} = srs; % success rates
    
    % import interlick intervals

    % plot trial performance as bar graph    
    for j=1:length(conditions) % iterate through conditions
        % split into condition days
        conddays = cond_days{i}; % per animal
        mask = ismember(days{i},conddays{j});
        sp = speeds(mask); sc = successes(mask); fl = fails(mask); tr = totals(mask);
        ur_lr = ur_lick_rates(mask); r_lr = r_lick_rates(mask);
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
        % plot interlick distance by reward zones
        % plot COM split by reward zones
        % get COM
        comfl = sprintf('Y:\\sstcre_analysis\\hrz\\lick_analysis\\%s_COMsplit_condition_%s.mat', ...
                animals{i}, conditions{j}); % saved from Gerardo's com code
        com = getCOMopto(comfl, j); % just for ep2 
        coms{j} = com;
        if i==1 && sum(ismember(days{i}(j:3:end), 75))==1 % exlclude e200 day 75 ep 3 bc not enough trials in ep 3
            com = cellfun(@(x) x(1:4), com, 'UniformOutput', false);
        end
        figure;
        bar([mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{1}, 'UniformOutput', false)), 'omitnan'); ... %this unravels the cellarray
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{2}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{3}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{4}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{5}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{6}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{7}, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{8}, 'UniformOutput', false)), 'omitnan')]','grouped','FaceColor','red');
        hold on
        plot(1,cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{1}, 'UniformOutput', false)),'ok')
        plot(2,cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{2}, 'UniformOutput', false)),'ok')
        plot(3,cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{3}, 'UniformOutput', false)),'ok')
        plot(4,cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{4}, 'UniformOutput', false)),'ok')
        plot(5,cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{5}, 'UniformOutput', false)),'ok')
        plot(6, cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{6}, 'UniformOutput', false)), 'ok')
        plot(7, cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{7}, 'UniformOutput', false)), 'ok')
        plot(8, cell2mat(cellfun(@(x) mean(x, 'omitnan'), com{8}, 'UniformOutput', false)), 'ok')

        xticklabels(["optotrials", "rest of opto epoch", "1st 5 trials epoch 1", ...
            "rest of epoch 1", "1st 5 trials epoch 2", "rest of epoch 2", "1st 5 trials epoch 3", ...
            "rest of epoch 3"])
        ylabel("COM licks of successful trials")
        title(sprintf("animal %s, %s", animals{i}, conditions{j}))
        % across all days of epoch 2,3,control, split by rew zone

        % plot rew zone representation across days
%         if j==1 % check what rew zone is epoch 2 
%             rew1 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(2))==1), rewzones(mask), 'UniformOutput', false)));
%             rew2 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(2))==2), rewzones(mask), 'UniformOutput', false)));
%             rew3 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(2))==3), rewzones(mask), 'UniformOutput', false)));
%             figure;
%             bar([rew1,rew2,rew3],'grouped','FaceColor','flat');
%             xticklabels(["rewzone 1", "rewzone 2", "rewzone 3"])
%             ylabel("# of days")
%             title(sprintf("animal %s, %s", animals{i}, conditions{j}))
%         elseif j==2
%             rew1 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(3))==1), rewzones(mask), 'UniformOutput', false)));
%             rew2 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(3))==2), rewzones(mask), 'UniformOutput', false)));
%             rew3 = sum(cell2mat(cellfun(@(x) sum(cell2mat(x(3))==3), rewzones(mask), 'UniformOutput', false)));    
%             figure;
%             bar([rew1,rew2,rew3],'grouped','FaceColor','flat');
%             xticklabels(["rewzone 1", "rewzone 2", "rewzone 3"])
%             ylabel("# of days")
%             title(sprintf("animal %s, %s", animals{i}, conditions{j}))
%         end
        
        % plot success rates       
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

        % plot lick rate
        figure;
        % automatically segments diff (opto) conditions
        bar([mean(cell2mat(cellfun(@(x) x{1}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ... %this unravels the cellarray
        mean(cell2mat(cellfun(@(x) x{2}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{3}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{4}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{5}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{6}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{7}(1), ur_lr, 'UniformOutput', false)), 'omitnan'); ...
        mean(cell2mat(cellfun(@(x) x{8}(1), ur_lr, 'UniformOutput', false)), 'omitnan')]','grouped','FaceColor','flat');
        hold on
        plot(1,cell2mat(cellfun(@(x) x{1}(1), ur_lr, 'UniformOutput', false)),'ok')
        plot(2,cell2mat(cellfun(@(x) x{2}(1), ur_lr, 'UniformOutput', false)),'ok')
        plot(3,cell2mat(cellfun(@(x) x{3}(1), ur_lr, 'UniformOutput', false)),'ok')
        plot(4,cell2mat(cellfun(@(x) x{4}(1), ur_lr, 'UniformOutput', false)),'ok')
        plot(5,cell2mat(cellfun(@(x) x{5}(1), ur_lr, 'UniformOutput', false)),'ok')
        plot(6, cell2mat(cellfun(@(x) x{6}(1), ur_lr, 'UniformOutput', false)), 'ok')
        plot(7, cell2mat(cellfun(@(x) x{7}(1), ur_lr, 'UniformOutput', false)), 'ok')
        plot(8, cell2mat(cellfun(@(x) x{8}(1), ur_lr, 'UniformOutput', false)), 'ok')
        
        xticklabels(["optotrials", "rest of opto epoch", "1st 5 trials epoch 1", ...
            "rest of epoch 1", "1st 5 trials epoch 2", "rest of epoch 2", "1st 5 trials epoch 3", ...
            "rest of epoch 3"])
        ylabel("unrewarded licks/num trials")
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
    com_an{i} = coms;
end

%%
% significance for lick rate;
% ep2
ep2restofopto = {};ep2restofcontrol = {}; % across animals
for an=1:length(animals)
lick_rates = ur_lrs{an}; j=1; % condition 1
lr = lick_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep2restofopto{an} = cell2mat(cellfun(@(x) x{2}(1), lr, 'UniformOutput', false));
j=3;
lr = lick_rates(j:3:end);
ep2restofcontrol{an} = cell2mat(cellfun(@(x) x{6}(1), lr, 'UniformOutput', false));
end
[h,p,~,stats]=ttest2(cell2mat(ep2restofopto),cell2mat(ep2restofcontrol))

% ep3
ep3restofopto = {};ep3restofcontrol = {};
for an=1:length(animals)
lick_rates = ur_lrs{an}; j=2; % condition 2
lr = lick_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep3restofopto{an} = cell2mat(cellfun(@(x) x{2}(1), lr, 'UniformOutput', false));
j=3;
lr = lick_rates(j:3:end);
ep3restofcontrol{an} = cell2mat(cellfun(@(x) x{8}(1), lr, 'UniformOutput', false));
end
[h,p,~,stats]=ttest2(cell2mat(ep3restofopto),cell2mat(ep3restofcontrol))

%%
% significance for success rate;
% ep2
ep2restofopto = {};ep2restofcontrol = {};
for an=1:length(animals)
success_rates = scs{an}; j=1; % condition 1
lr = success_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep2restofopto{an} = cell2mat(cellfun(@(x) x(2), lr, 'UniformOutput', false));
j=3;
lr = success_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep2restofcontrol{an} = cell2mat(cellfun(@(x) x(6), lr, 'UniformOutput', false));
end
[h,p,~,stats]=ttest2(cell2mat(ep2restofopto),cell2mat(ep2restofcontrol))

% ep3
ep3restofopto = {};ep3restofcontrol = {};
for an=1:length(animals)
success_rates = scs{an}; j=2; % condition 2
lr =  success_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep3restofopto{an} = cell2mat(cellfun(@(x) x(2), lr, 'UniformOutput', false));
j=3;
lr = success_rates(ismember(days{an},cell2mat(cond_days{an}(j))));
ep3restofcontrol{an} = cell2mat(cellfun(@(x) x(8), lr, 'UniformOutput', false));
end
[h,p,~,stats]=ttest2(cell2mat(ep3restofopto),cell2mat(ep3restofcontrol))

%%
% COM analysis
% for each condition, COM is split by rew zones
% always limited to 3 rew zones
ep2restofopto = {};ep2restofcontrol = {};
trials_to_test = 5;

% ep2
for comi=1:length(com_an)
    coms = com_an{comi};
    j=1; cond=2;
    ep2restofopto{comi} = cell2mat(cellfun(@(x) mean(x(end-trials_to_test:end)), coms{j}{cond}, 'UniformOutput', false)); % 2 here is the opto condtiion
    j=3; cond=6;
    ep2restofcontrol{comi} = cell2mat(cellfun(@(x) mean(x(end-trials_to_test:end)), coms{j}{cond}, 'UniformOutput', false)); % 6 here is the epoch condtiion
end
[h,p,~,stats]=ttest2(cell2mat(ep2restofopto),cell2mat(ep2restofcontrol))

ep3restofopto = {};ep3restofcontrol = {};
% ep3
for comi=1:length(com_an)
    coms = com_an{comi};
    j=2; cond=2;
    mask = cell2mat(cellfun(@(x) length(x), coms{j}{cond}, 'UniformOutput', false))>trials_to_test; % only get days with sufficient trials post opto
    cellarr = coms{j}{cond}(mask);
    ep3restofopto{comi} = cell2mat(cellfun(@(x) mean(x(end-trials_to_test:end)), cellarr, 'UniformOutput', false)); % 2 here is the opto condtiion
    j=3; cond=8;
    mask = cell2mat(cellfun(@(x) length(x), coms{j}{cond}, 'UniformOutput', false))>trials_to_test; % only get days with sufficient trials post opto
    cellarr = coms{j}{cond}(mask);
    ep3restofcontrol{comi} = cell2mat(cellfun(@(x) mean(x(end-trials_to_test:end)), cellarr, 'UniformOutput', false)); % 8 here is the epoch condtiion
end
[h,p,~,stats]=ttest2(cell2mat(ep3restofopto),cell2mat(ep3restofcontrol))