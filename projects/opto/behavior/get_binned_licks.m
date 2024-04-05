function [outputArg1,outputArg2] = get_binned_licks(licks, ybinned, bin_size, nbins, rewloc, ...
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
pertile = 0.25; % count licks w/in this percentile
rewsize_bin = rewsize/bin_size;
opp_rewloc = nbins-rewloc_bin;
end