function [lickbin_s,lickbin_f,prerewlickbin,prerewlickbin_ratio] = get_pre_reward_lick_binned(track_length, ...
    eprng, trialnum, rewards, licks, ybinned, ep, rewlocs, time)
bin_size = 2;
nbins = track_length/bin_size;  
time_ = time(eprng);
ybinned_ = ybinned(eprng);
trialnum_ = trialnum(eprng);
reward_ = rewards(eprng);
licks_ = licks(eprng);
% filter dark time licks out
licks_(ybinned_<3) = 0;
lickind = consecutive_stretch(find(licks_)); % get only first frame of lick bout
lickframes = cell2mat(cellfun(@(x) min(x), lickind, 'UniformOutput', false));
licks_con = zeros(1, length(licks_));
licks_con(lickframes) = 1;
rewloc = rewlocs(ep);
[success,fail,str, ftr, ttr, total_trials] = get_success_failure_trials(trialnum_,reward_);

rewloc_bin = rewloc/bin_size;
pertile = 0.25; % count licks w/in this percentile
prerewdist = rewloc_bin-(rewloc*pertile)/bin_size;
postrewdist = rewloc_bin+(rewloc*pertile)/bin_size;
% prerewlickbin = mean(lickbin_s(ceil(prerewdist):ceil(rewloc_bin)),'omitnan');
ypos = ybinned_(ismember(trialnum_,ttr));
flick = licks_(ismember(trialnum_,ttr));
prerewlicks = sum(flick((ypos>(rewloc-(rewloc*pertile))) & ypos<rewloc))/31.25;%/(sum(time_)/60)*30; % /s
% prerewlickbin_ratio = median(lickbin_f(ceil(prerewdist):ceil(rewloc_bin)),'omitnan');
% pre and post reward frames
% rng = [1:ceil(rewloc_bin), ceil(postrewdist):length(lickbin_f)];
alllicks = sum(flick(ypos<(rewloc) | ypos>(rewloc+rewloc*pertile)))/31.25;%;/(sum(time_)/60)*30; % /s
% prerewlickbin_ratio = prerewlickbin/mean(lickbin_s(rng),'omitnan');
prerewlickbin_ratio = alllicks/total_trials;
lickbin_s=0;lickbin_f=0;prerewlickbin=0; % legacy
end