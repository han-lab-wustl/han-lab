function [speed_5trialsep, speed_ep, speed_5trialsep1, speed_ep1, speed_5trialsep2, ...s
    speed_ep2, speed_5trialsep3, speed_ep3, s,f,t, ur_lick_rate, ...
    r_lick_rate, rewzone] = get_opto_behavior(vrfl, condition)
%only relies on vr file
%   custom conditions
% s f t ordered by conditions
% 1- opto 5 trials; 2 - rest of opto epoch, 3 - 1st epoch 5 trials, 4 -
% rest of epoch 1, etc...
load(vrfl)
eps = find(VR.changeRewLoc>0);
eps = [eps length(VR.changeRewLoc)];
% get from epoch 3 to 4 if there are 4 or more epochs, IMPORTANT


speed = -0.013*VR.ROE(2:end)./diff(VR.time);
% get reward zone info
% same as what is coded in original COMgeneralviewF script
changeRewLoc=find(VR.changeRewLoc);
changeRewLoc = [changeRewLoc length(VR.changeRewLoc)];
for kk = 1:length(changeRewLoc)-1
    if (VR.changeRewLoc(changeRewLoc(kk)))<=86
        rewzone{kk} = 1; % rew zone 1
    elseif (VR.changeRewLoc(changeRewLoc(kk)))<=120 && (VR.changeRewLoc(changeRewLoc(kk)))>=101
        rewzone{kk} = 2;
    elseif (VR.changeRewLoc(changeRewLoc(kk)))>=135
        rewzone{kk} = 3;
    end
end
rewloc = VR.changeRewLoc(VR.changeRewLoc>0);
ep2rng = eps(2):eps(3);
ep3rng = eps(3):eps(4)-1; % to account for the different size of speed variable

if strcmp(condition,'ep2')
    % opto epoch
    mask = VR.trialNum(ep2rng)>=3 & VR.trialNum(ep2rng)<8; % skip probes = 0,1,2 
    trialnum = VR.trialNum(ep2rng); rew = VR.reward(ep2rng);
    %rewarded vs. unrewarded licks
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    mask, rewloc, 2);
    ur_lick_rate{1} = ur_lr;
    r_lick_rate{1} = r_lr;
    speed_ = speed(ep2rng);
    speed_5trialsep = mean(speed_(mask));
    [s{1},f{1},t{1}] = get_success_failure_trials(trialnum(mask),rew(mask));

    speed_ep = mean(speed_(~mask));
    [s{2},f{2},t{2}] = get_success_failure_trials(trialnum(~mask),rew(~mask));  
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    ~mask, rewloc, 2);
    ur_lick_rate{2} = ur_lr;
    r_lick_rate{2} = r_lr;
    % compute time spent in epoch (first 5 trials vs. rest of epoch)

    % ep2 - set NaN
    speed_5trialsep2 = NaN;
    speed_ep2 = NaN;       
    s{5} = NaN; f{5} = NaN; t{5} = NaN;
    s{6} = NaN; f{6} = NaN; t{6} = NaN;
    ur_lick_rate{5} = NaN; ur_lick_rate{6} = NaN;
    r_lick_rate{5} = NaN; r_lick_rate{6} = NaN;
    % ep3
    mask = VR.trialNum(ep3rng)>=3 & VR.trialNum(ep3rng)<8; % -1 for speed vector
    trialnum = VR.trialNum(ep3rng); rew = VR.reward(ep3rng);
    [s{7},f{7},t{7}] = get_success_failure_trials(trialnum(~mask),rew(~mask));   
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    mask, rewloc, 3);
    ur_lick_rate{7} = ur_lr;
    r_lick_rate{7} = r_lr;
    speed_ = speed(ep3rng);
    speed_5trialsep3 = mean(speed_(mask));

    speed_ep3 = mean(speed_(~mask));
    [s{8},f{8},t{8}] = get_success_failure_trials(trialnum(~mask),rew(~mask));            
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    ~mask, rewloc, 3);
    ur_lick_rate{8} = ur_lr;
    r_lick_rate{8} = r_lr;
            

elseif strcmp(condition,'ep3')
    % opto epoch
    mask = VR.trialNum(ep3rng)>=3 & VR.trialNum(ep3rng)<8; % skip probes
    trialnum = VR.trialNum(ep3rng); rew = VR.reward(ep3rng);
    [s{1},f{1},t{1}] = get_success_failure_trials(trialnum(mask),rew(mask));
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    mask, rewloc, 3);
    ur_lick_rate{1} = ur_lr;
    r_lick_rate{1} = r_lr;
    speed_ = speed(ep3rng); 
    speed_5trialsep = mean(speed_(mask));   

    speed_ep = mean(speed_(~mask)); 
    [s{2},f{2},t{2}] = get_success_failure_trials(trialnum(~mask),rew(~mask));    
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    ~mask, rewloc, 3);
    ur_lick_rate{2} = ur_lr;
    r_lick_rate{2} = r_lr;
    % ep2
    mask = VR.trialNum(ep2rng)>=3 & VR.trialNum(ep2rng)<8;
    trialnum = VR.trialNum(ep2rng); rew = VR.reward(ep2rng);
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    mask, rewloc, 2);
    ur_lick_rate{5} = ur_lr;
    r_lick_rate{5} = r_lr;
    [s{5},f{5},t{5}] = get_success_failure_trials(trialnum(mask),rew(mask));
    speed_ = speed(ep2rng); 
    speed_5trialsep2 = mean(speed_(mask));

    speed_ep2 = mean(speed_(~mask));
    [s{6},f{6},t{6}] = get_success_failure_trials(trialnum(~mask),rew(~mask));    
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    ~mask, rewloc, 2);
    ur_lick_rate{6} = ur_lr;
    r_lick_rate{6} = r_lr;
    % ep3 - set NaN
    speed_5trialsep3 = NaN;
    speed_ep3 = NaN;
    s{7} = NaN; f{7} = NaN; t{7} = NaN;
    s{8} = NaN; f{8} = NaN; t{8} = NaN;
    ur_lick_rate{7} = NaN; ur_lick_rate{8} = NaN;   
    r_lick_rate{7} = NaN; r_lick_rate{8} = NaN;   

elseif strcmp(condition,'control')
    % NaNs for opto epochs
    speed_5trialsep = NaN;
    speed_ep = NaN;
    s{1} = NaN; f{1} = NaN; t{1} = NaN;
    s{2} = NaN; f{2} = NaN; t{2} = NaN;
    r_lick_rate{1} = NaN; r_lick_rate{2} = NaN;
    ur_lick_rate{1} = NaN; ur_lick_rate{2} = NaN;   
    % ep2
    mask = VR.trialNum(ep2rng)>=3 & VR.trialNum(ep2rng)<8;
    trialnum = VR.trialNum(ep2rng); rew = VR.reward(ep2rng);
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    mask, rewloc, 2);
    ur_lick_rate{5} = ur_lr;
    r_lick_rate{5} = r_lr;
    [s{5},f{5},t{5}] =get_success_failure_trials(trialnum(mask),rew(mask));
    speed_= speed(ep2rng); 
    speed_5trialsep2 = mean(speed_(mask));

    speed_ep2 = mean(speed_(~mask));
    [s{6},f{6},t{6}] = get_success_failure_trials(trialnum(~mask),rew(~mask));    
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep2rng, ...
    ~mask, rewloc, 2);
    ur_lick_rate{6} = ur_lr;
    r_lick_rate{6} = r_lr;
    % ep3
    mask = VR.trialNum(ep3rng)>=3 & VR.trialNum(ep3rng)<8;
    trialnum = VR.trialNum(ep3rng); rew = VR.reward(ep3rng);
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    mask, rewloc, 3);

    [s{7},f{7},t{7}] = get_success_failure_trials(trialnum(mask),rew(mask));
    ur_lick_rate{7} = ur_lr;
    r_lick_rate{7} = r_lr;
    speed_ = speed(ep3rng); 
    speed_5trialsep3 = mean(speed_(mask));

    speed_ep3 = mean(speed_(~mask));
    [s{8},f{8},t{8}] = get_success_failure_trials(trialnum(~mask),rew(~mask)); 
    [ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep3rng, ...
    ~mask, rewloc, 3);
    ur_lick_rate{8} = ur_lr;
    r_lick_rate{8} = r_lr;
end

%ep1 - same in all conditions
mask = VR.trialNum(eps(1):eps(2))>=3 & VR.trialNum(eps(1):eps(2))<8;
trialnum = VR.trialNum(eps(1):eps(2)); rew = VR.reward(eps(1):eps(2));
ep1rng = eps(1):eps(2);
[s{3},f{3},t{3}] = get_success_failure_trials(trialnum(mask),rew(mask));
[ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep1rng, ...
    mask, rewloc, 1);
ur_lick_rate{3} = ur_lr;
r_lick_rate{3} = r_lr;
speed_ = speed(eps(1):eps(2)); 
speed_5trialsep1 = mean(speed_(mask));

speed_ep1 = mean(speed_(~mask));
[s{4},f{4},t{4}] = get_success_failure_trials(trialnum(~mask),rew(~mask));   
[ur_lr, r_lr] = get_rewarded_and_unrewarded_licks(VR, ep1rng, ...
   ~mask, rewloc, 1);

ur_lick_rate{4} = ur_lr;
r_lick_rate{4} = r_lr;

end
