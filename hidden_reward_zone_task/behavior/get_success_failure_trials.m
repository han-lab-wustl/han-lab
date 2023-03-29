function [success,fail,total_trials] = get_success_failure_trials(vrfl)
% Zahra
% basic behavior quantification for HRZ
% integrate with MasterHRZ?

% vrfl = 'Z:\sstcre_imaging\e201\28\behavior\vr\E201_28_Mar_2023_time(08_56_06).mat';
load(vrfl,'VR');

success=0;fail=0;
for trial=unique(VR.trialNum)
    if trial>3 % trial < 3, probe trial
        if sum(VR.reward(VR.trialNum==trial)==1)>0 % if reward was found in the trial
            success=success+1;
        else
            fail=fail+1;
        end
    end
end

total_trials = sum(unique(VR.trialNum)>3);

end