function [speed_5trialsep, speed_ep, speed_ep1, speed_ep2, speed_ep3, ....
    speed_5trialsep2, speed_5trialsep3] = get_mean_speed(vrfl, condition)
%only relies on vr file
%   custom conditions
load(vrfl)
eps = find(VR.changeRewLoc>0);
speed = -0.013*VR.ROE(2:end)./diff(VR.time);

if strcmp(condition,'ep2')
    mask = VR.trialNum(eps(2):eps(3))>=3 & VR.trialNum(eps(2):eps(3))<8; % skip probes = 0,1,2 
    speed_5trialsep = mean(speed(mask));
    epex5trials = speed(eps(2):eps(3));
    speed_ep = mean(epex5trials(~mask));
    speed_ep3 = mean(speed(eps(3):end));
    speed_ep1 = mean(speed(eps(1):eps(2)));
    speed_ep2 = NaN;
    speed_5trialsep2 = NaN;
    speed_5trialsep3 = NaN;
elseif strcmp(condition,'ep3')
    % drop last frame because of speed vector
    mask = VR.trialNum(eps(3):end-1)>=3 & VR.trialNum(eps(3):end-1)<8; % skip probes
    speed_5trialsep = mean(speed(mask));
    epex5trials = speed(eps(3):end);
    speed_ep = mean(epex5trials(~mask));
    speed_ep2 = mean(speed(eps(2):eps(3)));
    speed_ep1 = mean(speed(eps(1):eps(2)));
    speed_ep3 = NaN;
    speed_5trialsep2 = NaN;
    speed_5trialsep3 = NaN;
elseif strcmp(condition,'control')
    speed_ep1 = mean(speed(eps(1):eps(2)));
    mask = VR.trialNum(eps(3):end-1)>3 & VR.trialNum(eps(3):end-1)<9; % skip probes
    speed_5trialsep2 = mean(speed(mask));
    epex5trials = speed(eps(3):end);
    speed_ep3 = mean(epex5trials(~mask));
    mask = VR.trialNum(eps(2):eps(3))>3 & VR.trialNum(eps(2):eps(3))<9; % skip probes 
    speed_5trialsep3 = mean(speed(mask));
    epex5trials = speed(eps(2):eps(3));
    speed_ep2 = mean(epex5trials(~mask));
    speed_5trialsep = NaN;
    speed_ep = NaN;
end
