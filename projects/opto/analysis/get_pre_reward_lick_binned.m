function [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, ep, rewlocs)
% only quants failed trials rn
bin_size = 2;
nbins = track_length/bin_size;  
ybinned_ = ybinned(eprng);
trialnum_ = trialnum(eprng);
reward_ = rewards(eprng);
licks_ = licks(eprng);
% filter dark time licks out
licks_(ybinned_<3) = 0;
rewloc = rewlocs(ep);
[success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);
frames = [1:length(ybinned_)];
frames = frames(ismember(trialnum_,str)); % successful trials
ypos = ybinned_(ismember(trialnum_,str));
time_in_bin = {};
for i = 1:nbins
    time_in_bin{i} = frames(ypos >= (i-1)*bin_size & ...
        ypos < i*bin_size);
end
% lick binning
lickbin = zeros(1,nbins);
for bin = 1:nbins
    lickbin(bin) = mean(licks_(time_in_bin{bin}));
end
lickbin_s = smoothdata(lickbin,"gaussian",2); % window of 3 (x2 =  cm)
frames = [1:length(ybinned_)];
frames = frames(ismember(trialnum_,ftr)); % successful trials
ypos = ybinned_(ismember(trialnum_,ftr));
time_in_bin = {};
for i = 1:nbins
    time_in_bin{i} = frames(ypos >= (i-1)*bin_size & ...
        ypos < i*bin_size);
end
% lick binning
lickbin = zeros(1,nbins);
for bin = 1:nbins
    lickbin(bin) = mean(licks_(time_in_bin{bin}));
end
lickbin_f = smoothdata(lickbin,"gaussian",2); % window of 3 (x2 =  cm)

rewloc_bin = rewloc/bin_size;
pertile = 0.25; % count licks w/in this percentile
prerewdist = rewloc_bin-(rewloc*pertile)/bin_size;
postrewdist = rewloc_bin+(rewloc*pertile)/bin_size;
prerewlickbin = mean(lickbin_s(ceil(prerewdist):ceil(rewloc_bin)),'omitnan');
% only success/failed trials
ypos = ybinned_(ismember(trialnum_,ttr));
flick = licks_(ismember(trialnum_,ttr));
prerewlicks = sum(flick((ypos>(rewloc-(rewloc*pertile))) & ypos<rewloc), 'omitnan');
% prerewlickbin_ratio = median(lickbin_f(ceil(prerewdist):ceil(rewloc_bin)),'omitnan');
% pre and post reward frames
rng = [1:ceil(rewloc_bin), ceil(postrewdist):length(lickbin_f)];
alllicks = sum(flick(ypos<(rewloc) | ypos>(rewloc+rewloc*pertile)),'omitnan');
% prerewlickbin_ratio = prerewlickbin/mean(lickbin_s(rng),'omitnan');
prerewlickbin_ratio = alllicks/total_trials;
end