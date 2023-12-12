% calc opto epoch success and fails
clear all; close all
mouse_name = "e218";
dys = [20,21,22,23,35,38,41,44];
opto_ep = [0,0,0,0,3 2 3,2];
src = "X:\vipcre";
epind = 1;
rates_opto = []; rates_ctrl = []; rates_preopto = []; licks_opto = []; licks_ctrl = []; licks_preopto = [];
for dy=dys
    daypth = dir(fullfile(src, mouse_name, string(dy), "**\*Fall.mat"));
    load(fullfile(daypth.folder,daypth.name), 'licks', 'trialnum', 'rewards', 'changeRewLoc');
    eps = find(changeRewLoc>0);
    eps = [eps length(changeRewLoc)];
    if opto_ep(epind)==3
        eprng = eps(3):eps(4);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        [exclude_frames_all] = get_frames_without_rewards_hrz(reward_);
        num_licks_nonreward_opto = sum(licks_(exclude_frames_all))/length(licks_(exclude_frames_all));
        successrate_opto = success/total_trials;
        % vs. previous epoch
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        successrate_ctrl = success/total_trials;   
        [exclude_frames_all] = get_frames_without_rewards_hrz(reward_);
        num_licks_nonreward_ctrl =  sum(licks_(exclude_frames_all))/length(licks_(exclude_frames_all));
        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = num_licks_nonreward_opto;
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = num_licks_nonreward_ctrl;
    elseif opto_ep(epind)==2
        eprng = eps(2):eps(3);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        [exclude_frames_all] = get_frames_without_rewards_hrz(reward_);
        num_licks_nonreward_opto =  sum(licks_(exclude_frames_all))/length(licks_(exclude_frames_all));
        successrate_opto = success/total_trials;
        % vs. previous epoch
        eprng = eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);        
        [exclude_frames_all] = get_frames_without_rewards_hrz(reward_);
        num_licks_nonreward_ctrl =  sum(licks_(exclude_frames_all))/length(licks_(exclude_frames_all));
        successrate_ctrl = success/total_trials;   
        rates_opto(epind) = successrate_opto;
        licks_opto(epind) = num_licks_nonreward_opto;
        rates_ctrl(epind) = successrate_ctrl;
        licks_ctrl(epind) = num_licks_nonreward_ctrl;
    else % just pre opto days
        eprng = 1:size(trialnum,2);%eps(1):eps(2);
        trialnum_ = trialnum(eprng);
        reward_ = rewards(eprng);
        licks_ = licks(eprng);
        [success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
        [exclude_frames_all] = get_frames_without_rewards_hrz(reward_);
        num_licks_nonreward_ctrl =  sum(licks_(exclude_frames_all))/length(licks_(exclude_frames_all));
        successrate = success/total_trials;
        rates_preopto(epind) = successrate;
        licks_preopto(epind) = num_licks_nonreward_ctrl;
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
[h,p,i,stats] = ttest2(rates_opto, rates_ctrl);

% lick rate

figure; 
bar([mean(licks_preopto) mean(licks_opto) mean(licks_ctrl)], 'FaceColor', 'w'); hold on
plot(1, licks_preopto, 'ko')
plot(2, licks_opto, 'ko')
plot(3, licks_ctrl, 'ko')
[h,p,i,stats] = ttest2(licks_opto, licks_ctrl);
ylabel('lick rate non reward')
xlabel('conditions')
xticklabels(["preopto days ep1", "opto ep", "previous ep"])
[h,p,i,stats] = ttest2(licks_opto, licks_ctrl);