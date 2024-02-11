% calc opto epoch success and fails
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before reward
% TODO lick rate outside rew zone
clear all; close all
% mouse_name = "e216";
mice = ["e216", "e218", "e189", "e190", "e201", "e186"];
cond = ["vip", "vip", "ctrl", "ctrl", "sst", "sst"];%, "pv"];
dys_s = {[7 8 9 37 38 39 40 41 42 44 45 46 48 50:59], ...
    [20,21,22,23,35,36,37,38,39,40,41,...
     42,43,44,45,47 48 49 50 51 52 55 56],...
     [35:42,44],...
     [33:35, 40:43, 45]...
     [52:59], [2:5,31,32,33]};
% experiment conditions: preopto=-1; optoep=3/2; control day1=0; control
% day2=1
opto_eps = {[-1 -1 -1 2 -1 0 1 3 -1 -1 0 1 2 3 0 1 2 3 0 1 2 0 1],...
    [-1 -1 -1 -1,3 0 1 2 0 1 3,0 1 2, 0 3 0 1 2 0 1 2 0],...
    [-1 -1 -1 -1 2 3 2 0 2],...
    [-1 -1 -1 3 0 1 2 3],...
    [-1 -1 -1 2 3 0 2 3],...
    [-1 -1 -1 -1 2 3 2]};
src = ["X:\vipcre", "X:\vipcre", '\\storage1.ris.wustl.edu\ebhan\Active\dzahra', ...
    '\\storage1.ris.wustl.edu\ebhan\Active\dzahra', 'Z:\sstcre_imaging', ...
    'Y:\analysis\fmats'];

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
rates_opto = []; rates_ctrl = []; rates_preopto = []; rates_inctrl_1 = []; rates_inctrl_2 = []; rates_postopto = [];
licks_opto = []; licks_ctrl = []; licks_preopto = []; licks_inctrl_1 = []; licks_inctrl_2 = []; licks_postopto = [];
com_opto_success = []; com_ctrl_success = []; com_preopto_success = []; com_inctrl_1_success = []; com_inctrl_2_success = []; com_postopto_success = [];
com_opto_fails = []; com_ctrl_fails = []; com_preopto_fails = []; com_inctrl_1_fails = []; com_inctrl_2_fails = []; com_postopto_fails = [];
for dy=dys
%     daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
%     load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
%         'ybinned', 'timedFF', 'VR');
    daypth = dir(fullfile(src(m), mice(m), string(dy), '**', '*time*.mat'));    
    if m==6 % e186 in a diff format
        daypth = dir(fullfile(src(m), mice(m), 'days', '*.mat'));
        daypth = daypth(dy);    
        load(fullfile(daypth.folder, daypth.name), 'VR') % load fall
    else
        load(fullfile(daypth.folder, daypth.name))
    end
    eps = find(VR.changeRewLoc>0);
    eps = [eps length(VR.changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    nbins = track_length/bin_size;
    ybinned = VR.ypos/VR.scalingFACTOR;
    rewlocs = VR.changeRewLoc(VR.changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    trialnum = VR.trialNum; % use VR variables
    rewards = VR.reward;
    licks = logical(VR.lick);
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_opto = success/total_trials;
        last8rng = (ismember(trialnum_,str)); % only do for failed trials
        lick_selectivity_opto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, rewsize);
        % get failed pre and post licks
        success=1;[com_success_opto] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize, success); % coms successful trials
        success=0;[com_fails_opto] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize,success); % coms failed trials
        % vs. previous epoch
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(2);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_ctrl = success/total_trials;
        last8rng = (ismember(trialnum_,str));
        lick_selectivity_ctrl = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc,rewsize);
        success=1;[com_success_ctrl] = get_com_licks(trialnum_, reward_, str, licks_, ybinned_, rewloc, ...
        rewsize,success); % coms successful trials
        success=0;[com_fails_ctrl] = get_com_licks(trialnum_, reward_, ftr, licks_, ybinned_, rewloc, ...
        rewsize,success); % coms failed trials
        % vs. next ep
        if length(eps)>4
            eprng = eps(4):eps(5);
            trialnum_ = trialnum(eprng);
            reward_ = rewards(eprng);
            licks_ = licks(eprng);
            ybinned_ = ybinned(eprng);
            rewloc = rewlocs(4);
            [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
            successrate_postopto = success/total_trials;
            last8rng = (ismember(trialnum_,str));
            lick_selectivity_postopto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc,rewsize);
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
        lick_selectivity_opto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc,  rewsize);
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
        lick_selectivity_ctrl = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc, rewsize);        
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
            lick_selectivity_postopto = get_lick_selectivity(licks_(last8rng), ybinned_(last8rng), bin_size, nbins, rewloc,rewsize);
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
% barplot with ctrl mice
% cond = ["ctrloff", "ctrlon", "vipoff", "vipon"];
figure;
means = [mean([rates{3}{4}' rates{4}{4}' rates{5}{4}' rates{6}{4}']) NaN ...
    mean([rates{3}{5}' rates{4}{5}' rates{5}{5}' rates{6}{5}']) NaN ...
    mean([rates{1}{4}' rates{2}{4}']) NaN...
    mean([rates{1}{5}' rates{2}{5}'])];
bar(means, 'FaceColor', 'w'); hold on
y1 = [rates{3}{4}' rates{4}{4}' rates{5}{4}' rates{6}{4}']; x = ones(1,size(y1,2));
swarmchart(x,y1,'k')
y2 = [rates{3}{5}' rates{4}{5}' rates{5}{5}' rates{6}{5}']; x = ones(1,size(y2,2))*3;
swarmchart(x,y2,'k')
y3 = [rates{1}{4}' rates{2}{4}']; x = ones(1,size(y3,2))*5;
swarmchart(x,y3,'k')
y4 = [rates{1}{5}' rates{2}{5}']; x = ones(1,size(y4,2))*7;
swarmchart(x,y4,'k')
yerr = {y1,NaN,y2,NaN,y3,NaN,y4};
err = [];
for i=1:length(yerr)
    err(i) =(std(yerr{i},'omitnan')/sqrt(size(yerr{i},2))); 
end
er = errorbar([1 NaN 3 NaN 5 NaN 7],means,err);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
ylabel('Success Rate')
xticklabels(["Control LED off", "", "Control LED on", "", "VIP stGtACR LED off", ...
    "", "VIP stGtACR LED on"])
% [h,p1,i,stats] = ttest2([rates{3}{5}' rates{4}{5}' rates{5}{5}' rates{6}{5}'], ....
%    [rates{1}{5}' rates{2}{5}']); % sig
% [h,p2,i,stats] = ttest([rates{1}{4}' rates{2}{4}'], ....
%    [rates{1}{5}' rates{2}{5}']); % sig
box off
condt = [repelem("Control LED off",length(y1)), repelem("Control LED on",length(y2)), repelem("VIP stGtACR LED off",length(y3)), ...
    repelem("VIP stGtACR LED on",length(y4))]';
tbl = table(condt,[rates{3}{4}' rates{4}{4}' rates{5}{4}' rates{6}{4}' rates{3}{5}' rates{4}{5}' rates{5}{5}'...
    rates{6}{5}' rates{1}{4}' rates{2}{4}'...
    rates{2}{5}' rates{2}{5}']',VariableNames=["Condition" "Success Rate"]);
aov = anova(tbl,'Success Rate');
multcompare(aov,'CriticalValueType',"bonferroni")
nctrl = length(y2);
title(sprintf('control n=4, %i sessions, opto n=2, 14 sessions \n p=%f b/wn led on control vs. opsin \n p=%f b/wn opsin led off vs. on ',nctrl,p1,p2))
%%
% mean of sessions
figure;
bar([mean(mean([rates{3}{4}' rates{4}{4}'])) ...
    mean(mean([rates{3}{5}' rates{4}{5}']))...
    mean(mean([rates{1}{4}' rates{2}{4}']))...
    mean(mean([rates{1}{5}' rates{2}{5}']))], 'FaceColor', 'w'); hold on
plot(1, [mean(rates{3}{4}') mean(rates{4}{4}')], 'ko')
plot(2, [mean(rates{3}{5}') mean(rates{4}{5}')], 'ko')
plot(3, [mean(rates{1}{4}') mean(rates{2}{4}')], 'ko')
plot(4, [mean(rates{1}{5}') mean(rates{2}{5}')], 'ko')
ylabel('Fraction of Successful Trials')
xlabel('Condition')
xticklabels(["Vector Control LED off", "Vector Control LED on", "stGtACR LED off", ...
    "stGtACR LED on"])
[h,p,i,stats] = ttest2([mean(rates{3}{5}') mean(rates{4}{5}')], ....
   [mean(rates{1}{5}') mean(rates{2}{5}')]); % sig
title(sprintf('control n=2, 8 sessions, opto n=2, 14 sessions \n p=%f b/wn led on control vs. opsin',p))
diffpow = mean(y4);
basepwr = mean(y2);
stdpow = std([y2]);
nout = sampsizepwr('t2',[basepwr, stdpow],diffpow,0.80);
pwrout = sampsizepwr('t2',[basepwr, stdpow],diffpow,[],7);
% if the effect is half
halfeffect = (basepwr-diffpow)/2;
nout_half = sampsizepwr('t2',[basepwr, stdpow],basepwr-halfeffect,0.80);
% need 4 animals to
% detect a difference between led on and off sessions and between vector
% ctrl led on and vector control led on and opsin led on
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
%%
% com fails
figure;
bar([mean([com_fail{3}{4}' com_fail{4}{4}' com_fail{5}{4}' com_fail{6}{4}'],'omitnan') NaN ...
    mean([com_fail{3}{5}' com_fail{4}{5}' com_fail{5}{5}' com_fail{6}{5}'],'omitnan') NaN ...
    mean([com_fail{1}{4}' com_fail{2}{4}'],'omitnan') NaN...
    mean([com_fail{1}{5}' com_fail{2}{5}'],'omitnan')], 'FaceColor', 'w'); hold on
y = [com_fail{3}{4}' com_fail{4}{4}' com_fail{5}{4}' com_fail{6}{4}']; x = ones(1,size(y,2));
swarmchart(x,y,'k')
y = [com_fail{3}{5}' com_fail{4}{5}' com_fail{5}{5}' com_fail{6}{5}']; x = ones(1,size(y,2))*3;
swarmchart(x,y,'k')
y = [com_fail{1}{4}' com_fail{2}{4}']; x = ones(1,size(y,2))*5;
swarmchart(x,y,'k')
y = [com_fail{1}{5}' com_fail{2}{5}']; x = ones(1,size(y,2))*7;
swarmchart(x,y,'k')
ylabel(['ypos COM of licks - rewloc start' newline '(excluding consumption licks)'])
xticklabels(["Control LED off", "", "Control LED on", "", "VIP stGtACR LED off", ...
    "", "VIP stGtACR LED on"])
title('failed trials')
[h,p,i,stats] = ttest2([com_fail{1}{4}' com_fail{2}{4}'], ....
   [com_fail{1}{5}' com_fail{2}{5}']); % sig