
clear all; close all
load("Z:\VIP_intswanalysisv9.mat")
an = 'vip';
%%
dy = 2; % day
fall = VIP_ints(1).NR.day{1,dy}.all;
plns = fall.Falls;
planes = 3;
bin_size = 0.2; %s
range = 10; %s
changeRewLoc = fall.rewardLocation{1}(:,1)';
rewards = fall.rewards{1}(:,1)';
timedFF = fall.time{1}(:,1)';
forwardvel = fall.Forwards{1}(:,1)';
eps = find(changeRewLoc>0);
eps = [eps length(changeRewLoc)];
track_length = 180; %cm; TODO: import from VR instead
nbins = track_length/bin_size;
rewlocs = changeRewLoc(changeRewLoc>0);
grayColor = [.7 .7 .7]; 
pln = 1;
dff = fall.Falls{pln}; dff = dff; % invert for gm struct
trialnum = fall.trialNum{1}(:,1)';
for ep=1:length(eps)-1
    dff_ep = dff(eps(ep):eps(ep+1));
    rewloc = changeRewLoc(ep);
    ypos = fall.ybinned{1}(:,1); ypos = ypos(eps(ep):eps(ep+1));
    rewards_center = zeros(size(changeRewLoc(eps(ep):eps(ep+1))));
    rewards_center((ypos>rewloc-2) & (ypos<rewloc+2)) = 1;
    trialnum_ep = trialnum(eps(ep):eps(ep+1));
    time_ep = timedFF(eps(ep):eps(ep+1));
    % get only probes
    [binnedPerireward,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dff_ep(trialnum_ep<3),rewards_center(trialnum_ep<3), ...
    time_ep(trialnum_ep<3),range,bin_size);
    figure; imagesc(binnedPerireward)
    endBHBNZ

[binnedPerireward,allbins,rewdFF,normmeanrewdFF] = perirewardbinnedactivity(dff,rewards,timedFF, ...
    range,bin_size);
[binnedvel,~,rewvel] = perirewardbinnedvelocity(forwardvel,rewards,timedFF, ...
    range,bin_size);