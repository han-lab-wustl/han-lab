function [lick_selectivity] = get_lick_selectivity(licks, ybinned, bin_size, nbins, rewloc, ...
    rewsize)
% can do trial by trial as well
% https://www.nature.com/articles/s41593-022-01050-4
% lick selectivity = smoothed licks in rew zone - smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
% rewzone = 10 cm before rewards\
% only do for last 10 trials?
rewzone_licks = 20; %cm 
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
opp_rewloc = nbins-rewloc_bin;
% get beginning of rewzone
% 20 cm before reward
lickrewzone = lickbin(ceil(rewloc_bin)-10:ceil(rewloc_bin-rewsize_bin/2));
try % if the opposite area is too close to the end
    lickoppzone = lickbin(ceil(opp_rewloc)-10:ceil(opp_rewloc-rewsize_bin/2));
catch
    lickoppzone = lickbin(ceil(opp_rewloc)-5:ceil(opp_rewloc-rewsize_bin/2)); % take only 10 cm before
end

lick_selectivity = (mean(lickrewzone, 'omitnan')-mean(lickoppzone, 'omitnan'))/(mean(lickrewzone, 'omitnan')+mean(lickoppzone, 'omitnan'));
% smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)

end