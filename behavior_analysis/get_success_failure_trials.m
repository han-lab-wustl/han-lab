function [success,fail,total_trials] = get_success_failure_trials(trialnum,reward)
% Zahra
% basic behavior quantification for HRZ
% integrate with MasterHRZ?

% vrfl = 'Z:\sstcre_imaging\e201\28\behavior\vr\E201_28_Mar_2023_time(08_56_06).mat';
% if typefl == 'vrfile'
%     load(vrfl,'VR');
%     trialnum = VR.trialNum;
%     reward = VR.reward;
% else
%     fmat = load(vrfl);
%     trialnum = fmat.trialnum;
%     reward = fmat.rewards;
% end

success=0;fail=0;
for trial=unique(trialnum)
    if trial>=3 % trial < 3, probe trial
        if sum(reward(trialnum==trial)==1)>0 % if reward was found in the trial
            success=success+1;
        else
            fail=fail+1;
        end
    end
end

total_trials = sum(unique(trialnum)>=3);

end