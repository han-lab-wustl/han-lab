% calc opto epoch success and fails
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before reward
% TODO lick rate outside rew zone
clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218"];
dys_s = {[7 8 9 37 38 39 40 41 42 43 44 45 46 48 50 51 52 53 54 55 56 57 58], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 2 -1 -1 -1 0 1 2 3 0 1 2 3 0 1 2 0],
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0]};
src = "X:\vipcre";

% mouse_name = "e218";
% dys = 
% % experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% % day2=1
% opto_ep = [-1 -1 -1 -1,3 0 1 2 0 1 3,...
%     0 1 2, 0 3 0 1 2 0 1 2 0]; 
% src = "X:\vipcre";
rates = {};
for m=1:length(mice)
dys = dys_s(m); dys = dys{1};
opto_ep = opto_eps(m); opto_ep = opto_ep{1};
epind = 1; % for indexing
bin_size = 2; % cm bins for lick
% ntrials = 8; % get licks for last n trials
prerew_opto = []; rates_ctrl = []; rates_preopto = []; rates_inctrl_1 = []; rates_inctrl_2 = []; rates_postopto = [];
licks_opto = []; licks_ctrl = []; licks_preopto = []; licks_inctrl_1 = []; licks_inctrl_2 = []; licks_postopto = [];
for dy=dys
%     daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
%     load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
%         'ybinned', 'timedFF', 'VR');
    daypth = dir(fullfile(src, mice(m), string(dy), 'behavior\vr\*.mat'));
    load(fullfile(daypth.folder, daypth.name))
    eps = find(VR.changeRewLoc>0);
    eps = [eps length(VR.changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    nbins = track_length/bin_size;
    ybinned = VR.ypos/VR.scalingFACTOR;
    vel = -0.013*VR.ROE./diff(VR.time(2:end));  
    rewlocs = VR.changeRewLoc(VR.changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    trialnum = VR.trialNum; % use VR variables
    rewards = VR.reward;
    licks = logical(VR.lick);
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        vel_ = vel(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng); 
        % around reward
        vel_near_reward_opto = vel_((ybinned_>=(rewloc-rewsize/2)-5) && (ybinned_<(rewloc+rewsize/2)+30));
        % vs. previous epoch
        eprng = eps(2):eps(3);        
        vel_ = vel(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng); 
        % around reward
        vel_near_reward_beforeopto = vel_((ybinned_>=(rewloc-rewsize/2)-30) && (ybinned_<(rewloc+rewsize/2)+30));
        % vs. next ep
        if length(eps)>4
            eprng = eps(4):eps(5);
            vel_ = vel(eprng);
            reward_ = rewards(eprng);
            licks_ = licks(eprng);
            ybinned_ = ybinned(eprng); 
            % around reward
            vel_near_reward_afteropto = vel_((ybinned_>=(rewloc-rewsize/2)-30) && (ybinned_<(rewloc+rewsize/2)+30));
        else
            successrate_postopto = 0; lick_selectivity_postopto=0; com_success_postopto=0; com_fails_postopto=0;
        end

        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = lick_selectivity_opto;
        rates_postopto(epind) = successrate_postopto;
        licks_postopto(epind) = lick_selectivity_postopto;
        com_opto_success(epind) = mean(com_success_opto,'omitnan');
        com_opto_fails(epind) = mean(com_fails_opto, 'omitnan');
        com_postopto_success(epind) = mean(com_success_postopto,'omitnan');
        com_postopto_fails(epind) = mean(com_fails_postopto, 'omitnan');
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = lick_selectivity_ctrl;
        com_ctrl_success(epind) = mean(com_success_ctrl,'omitnan');
        com_ctrl_fails(epind) = mean(com_fails_ctrl,'omitnan');
    elseif opto_ep(epind)==2
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_opto = success/total_trials;
        last8rng = (ismember(trialnum_,str));
        lick_selectivity_opto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, ...
            rewsize);
        success=1;[com_success_opto] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms successful trials
        success=0;[com_fails_opto] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms failed trials
        % vs. previous epoch
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_ctrl = success/total_trials;
        last8rng = (ismember(trialnum_,str));
        success=1; [com_success_ctrl] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize,success); % coms successful trials
        success=0; [com_fails_ctrl] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize,success); % coms failed trials
        % vs. next ep
        if length(eps)>3
            eprng = eps(3):eps(4);
            trialnum_ = trialnum(eprng);
            reward_ = rewards(eprng);
            licks_ = licks(eprng);
            ybinned_ = ybinned(eprng);
            rewloc = rewlocs(3);
            [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
            successrate_postopto = success/total_trials;
            last8rng = (ismember(trialnum_,str));
            lick_selectivity_postopto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, ...
                rewsize);
            success=1;[com_success_postopto] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
            rewsize,success); % coms successful trials
            success=0;[com_fails_postopto] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
            rewsize,success); % coms failed trials
        else
            successrate_postopto = 0; lick_selectivity_postopto=0; com_success_postopto=0; com_fails_postopto=0;
        end

        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = lick_selectivity_opto;
        rates_postopto(epind) = successrate_postopto;
        licks_postopto(epind) = lick_selectivity_postopto;
        com_opto_success(epind) = mean(com_success_opto,'omitnan');
        com_opto_fails(epind) = mean(com_fails_opto, 'omitnan');
        com_postopto_success(epind) = mean(com_success_postopto,'omitnan');
        com_postopto_fails(epind) = mean(com_fails_postopto, 'omitnan');
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = lick_selectivity_ctrl;
        com_ctrl_success(epind) = mean(com_success_ctrl,'omitnan');
        com_ctrl_fails(epind) = mean(com_fails_ctrl,'omitnan');

    elseif opto_ep(epind)==-1 % just pre opto days
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        last8rng = (ismember(trialnum_,str));
        lick_selectivity_ctrl = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, ...
            rewsize);
        successrate = success/total_trials;
        success = 1; [com_success_ctrl] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms successful trials
        success = 0; [com_fails_ctrl] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms failed trials
        rates_preopto(epind) = successrate;
        licks_preopto(epind) = lick_selectivity_ctrl;
        com_preopto_success(epind) = mean(com_success_ctrl,'omitnan');
        com_preopto_fails(epind) = mean(com_fails_ctrl,'omitnan');
    elseif opto_ep(epind)==0  % intermediate control days 1
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        last8rng = (ismember(trialnum_,str));
        lick_selectivity_ctrl = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, ...
            rewsize);
        successrate = success/total_trials;
        success=1; [com_success_ctrl] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms successful trials
        success=0; [com_fails_ctrl] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms failed trials
        rates_inctrl_1(epind) = successrate;
        licks_inctrl_1(epind) = lick_selectivity_ctrl;
        com_inctrl_1_success(epind) = mean(com_success_ctrl,'omitnan');
        com_inctrl_1_fails(epind) = mean(com_fails_ctrl,'omitnan');
    elseif opto_ep(epind)==1  % intermediate control days 2
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        last8rng = (ismember(trialnum_,str));
        lick_selectivity_ctrl = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, ...
            rewsize);
        successrate = success/total_trials;
        success=1; [com_success_ctrl] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms successful trials
        success=0; [com_fails_ctrl] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms failed trials
        rates_inctrl_2(epind) = successrate;
        licks_inctrl_2(epind) = lick_selectivity_ctrl;
        com_inctrl_2_success(epind) = mean(com_success_ctrl,'omitnan');
        com_inctrl_2_fails(epind) = mean(com_fails_ctrl,'omitnan');
    end
    epind = epind+1;   
end

%remove zeros from other days
rates_opto = nonzeros(rates_opto);
rates_preopto = nonzeros(rates_preopto);
rates_postopto = nonzeros(rates_postopto);
rates_ctrl = nonzeros(rates_ctrl);
rates_inctrl_1 = nonzeros(rates_inctrl_1);
rates_inctrl_2 = nonzeros(rates_inctrl_2);
licks_preopto = nonzeros(licks_preopto);
licks_postopto = nonzeros(licks_postopto);
licks_opto = nonzeros(licks_opto);
licks_ctrl = nonzeros(licks_ctrl);
licks_inctrl_1 = nonzeros(licks_inctrl_1);
licks_inctrl_2 = nonzeros(licks_inctrl_2);
com_preopto_success = nonzeros(com_preopto_success);
com_postopto_success = nonzeros(com_postopto_success);
com_opto_success = nonzeros(com_opto_success);
com_ctrl_success = nonzeros(com_ctrl_success);
com_inctrl_1_success = nonzeros(com_inctrl_1_success);
com_inctrl_2_success = nonzeros(com_inctrl_2_success);
com_preopto_fails = nonzeros(com_preopto_fails);
com_postopto_fails = nonzeros(com_postopto_fails);
com_opto_fails = nonzeros(com_opto_fails);
com_ctrl_fails = nonzeros(com_ctrl_fails);
com_inctrl_1_fails = nonzeros(com_inctrl_1_fails);
com_inctrl_2_fails = nonzeros(com_inctrl_2_fails);
rates{m} = {rates_preopto, rates_inctrl_1  rates_inctrl_2  rates_ctrl  rates_opto, rates_postopto};
com_success{m} = {com_preopto_success, com_inctrl_1_success  com_inctrl_2_success  com_ctrl_success  com_opto_success, com_postopto_success};
com_fail{m} = {com_preopto_fails, com_inctrl_1_fails  com_inctrl_2_fails  com_ctrl_fails  com_opto_fails, com_postopto_fails};
end
%%
% barplot
figure;
bar([mean([rates{1}{1}' rates{2}{1}'])...
    mean([rates{1}{2}' rates{2}{2}']) ...
    mean([rates{1}{3}' rates{2}{3}'])...
    mean([rates{1}{4}' rates{2}{4}'])...
    mean([rates{1}{5}' rates{2}{5}'])...
    mean([rates{1}{6}' rates{2}{6}'])], 'FaceColor', 'w'); hold on
plot(1, [rates{1}{1}' rates{2}{1}'], 'ko')
plot(2, [rates{1}{2}' rates{2}{2}'], 'ko')
plot(3, [rates{1}{3}' rates{2}{3}'], 'ko')
plot(4, [rates{1}{4}' rates{2}{4}'], 'ko')
plot(5, [rates{1}{5}' rates{2}{5}'], 'ko')
plot(6, [rates{1}{6}' rates{2}{6}'], 'ko')
ylabel('success rate')
xlabel('conditions')
xticklabels(["preopto days all ep", "control day 1 in b/wn opto", "control day 2 in b/wn opto", ...
    "previous ep", "opto ep", "postopto ep"])
[h,p,i,stats] = ttest2([rates{1}{4}' rates{2}{4}'], ....
   [rates{1}{5}' rates{2}{5}']); % sig
%%
% com
figure;
bar([mean([com_success{1}{1}' com_success{2}{1}'])...
    mean([com_success{1}{2}' com_success{2}{2}']) ...
    mean([com_success{1}{3}' com_success{2}{3}'])...
    mean([com_success{1}{4}' com_success{2}{4}'])...
    mean([com_success{1}{5}' com_success{2}{5}'])...
    mean([com_success{1}{6}' com_success{2}{6}'])], 'FaceColor', 'w'); hold on
plot(1, [com_success{1}{1}' com_success{2}{1}'], 'ko')
plot(2, [com_success{1}{2}' com_success{2}{2}'], 'ko')
plot(3, [com_success{1}{3}' com_success{2}{3}'], 'ko')
plot(4, [com_success{1}{4}' com_success{2}{4}'], 'ko')
plot(5, [com_success{1}{5}' com_success{2}{5}'], 'ko')
plot(6, [com_success{1}{6}' com_success{2}{6}'], 'ko')
ylabel(['ypos COM of licks - rewloc start' newline '(excluding consumption licks)'])
xlabel('conditions')
xticklabels(["preopto days all ep", "control day 1 in b/wn opto", "control day 2 in b/wn opto", ...
    "previous ep", "opto ep",  "postopto ep"])
title('successful trials')
[h,p,i,stats] = ttest2([com_success{1}{4}' com_success{2}{4}'], ....
   [com_success{1}{5}' com_success{2}{5}']); % sig

% com fails

figure;
bar([mean([com_fail{1}{1}' com_fail{2}{1}'])...
    mean([com_fail{1}{2}' com_fail{2}{2}']) ...
    mean([com_fail{1}{3}' com_fail{2}{3}'])...
    mean([com_fail{1}{4}' com_fail{2}{4}'])...
    mean([com_fail{1}{5}' com_fail{2}{5}'])...
    mean([com_fail{1}{6}' com_fail{2}{6}'])], 'FaceColor', 'w'); hold on
plot(1, [com_fail{1}{1}' com_fail{2}{1}'], 'ko')
plot(2, [com_fail{1}{2}' com_fail{2}{2}'], 'ko')
plot(3, [com_fail{1}{3}' com_fail{2}{3}'], 'ko')
plot(4, [com_fail{1}{4}' com_fail{2}{4}'], 'ko')
plot(5, [com_fail{1}{5}' com_fail{2}{5}'], 'ko')
plot(6, [com_fail{1}{6}' com_fail{2}{6}'], 'ko')
ylabel(['ypos COM of licks - rewloc start' newline '(excluding consumption licks)'])
xlabel('conditions')
xticklabels(["preopto days all ep", "control day 1 in b/wn opto", "control day 2 in b/wn opto", ...
    "previous ep", "opto ep",  "postopto ep"])
title('failed trials')
[h,p,i,stats] = ttest2([com_fail{1}{4}' com_fail{2}{4}'], ....
   [com_fail{1}{5}' com_fail{2}{5}']); % sig