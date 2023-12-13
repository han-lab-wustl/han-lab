% calc opto epoch success and fails
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before reward
clear all; close all
mouse_name = "e218";
dys = [20,21,22,23,35,38,41,44];
opto_ep = [0,0,0,0,3 2 3,2];
src = "X:\vipcre";
epind = 1;
bin_size = 2; % cm bins for lick
rates_opto = []; rates_ctrl = []; rates_preopto = []; licks_opto = []; licks_ctrl = []; licks_preopto = [];
for dy=dys
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc', ...
        'ybinned', 'timedFF', 'VR');
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    track_length = 180/VR.scalingFACTOR;
    nbins = track_length/bin_size;
    ybinned = ybinned/VR.scalingFACTOR;
    rewlocs = changeRewLoc(changeRewLoc>0)/VR.scalingFACTOR;
    rewsize = VR.settings.rewardZone/VR.scalingFACTOR;
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_opto = success/total_trials;
        lick_selectivity_opto = get_lick_selectivity(licks_, ybinned_, bin_size, nbins, rewloc, ...
        rewsize);
        % vs. previous epoch
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(2);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_ctrl = success/total_trials;   
        lick_selectivity_ctrl = get_lick_selectivity(licks_, ybinned_, bin_size, nbins, rewloc, ...
        rewsize);
        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = lick_selectivity_opto;
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = lick_selectivity_ctrl;
    elseif opto_ep(epind)==2
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(opto_ep(epind));
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_opto = success/total_trials;
        lick_selectivity_opto = get_lick_selectivity(licks_, ybinned_, bin_size, nbins, rewloc, ...
        rewsize);
        % vs. previous epoch
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_ctrl = success/total_trials;   
        lick_selectivity_ctrl = get_lick_selectivity(licks_, ybinned_, bin_size, nbins, rewloc, ...
        rewsize);
        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = lick_selectivity_opto;
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = lick_selectivity_ctrl;
    else % just pre opto days
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        ybinned_ = ybinned(eprng);
        rewloc = rewlocs(1);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum,rewards);
        lick_selectivity_ctrl = get_lick_selectivity(licks_, ybinned_, bin_size, nbins, rewloc, ...
        rewsize);
        successrate = success/total_trials;
        rates_preopto(epind) = successrate;
        licks_preopto(epind) = lick_selectivity_ctrl;
    end    
    epind = epind+1;
end

%remove zeros from others
rates_opto = nonzeros(rates_opto);
rates_ctrl = nonzeros(rates_ctrl);
licks_opto = nonzeros(licks_opto);
licks_ctrl = nonzeros(licks_ctrl);
% barplot
figure; 
bar([mean(rates_preopto) mean(rates_opto) mean(rates_ctrl)], 'FaceColor', 'w'); hold on
plot(1, rates_preopto, 'ko')
plot(2, rates_opto, 'ko')
plot(3, rates_ctrl, 'ko')
ylabel('success rate')
xlabel('conditions')
xticklabels(["preopto days all ep", "opto ep", "previous ep"])
[h,p,i,stats] = ttest2(rates_opto, rates_ctrl); % sig

% lick rate

figure; 
bar([mean(licks_preopto) mean(licks_opto) mean(licks_ctrl)], 'FaceColor', 'w'); hold on
plot(1, licks_preopto, 'ko')
plot(2, licks_opto, 'ko')
plot(3, licks_ctrl, 'ko')
ylabel('lick selectivity (excluding consumption licks)')
xlabel('conditions')
xticklabels(["preopto days ep1", "opto ep", "previous ep"])
[h,p,i,stats] = ttest2(licks_preopto,licks_opto); % sig