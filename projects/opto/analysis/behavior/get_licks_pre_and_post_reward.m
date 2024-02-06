function [lick_pre, lick_post] = get_licks_pre_and_post_reward(licks, ybinned, bin_size, nbins, rewloc, ...
    rewsize)
% can do trial by trial as well
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before rewards\
% only do for last 10 trials?
frames = 1:length(ybinned);
for i = 1:nbins
    time_in_bin{i} = frames(ybinned >= (i-1)*bin_size & ...
        ybinned < i*bin_size);
end

% lick binning
lickbin = zeros(1,nbins);
for bin = 1:nbins
    lickbin(bin) = mean(licks(time_in_bin{bin}));
end
lickbin = smoothdata(lickbin,"gaussian",2); % window of 3 (x2 =  cm)

rewloc_bin = rewloc/bin_size;
rewsize_bin = rewsize/bin_size;
% get licks pre reward
lick_pre = lickbin(1:rewloc_bin-rewsize_bin/2);
lick_post = lickbin((rewloc_bin+rewsize_bin/2+20):end); % 20 cm after rew zone
end