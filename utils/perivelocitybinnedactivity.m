function [binnedPerivelocity,allbins,rewvel] = perivelocitybinnedactivity(velocity,rewards,dff,timedFF, ...
    range,binsize,numplanes)
%%%
% dff aligned to stops
%%%
[moving_middle stop] = get_moving_time_V3(velocity,2,10,30);
% params for moving_time
% velocity - forwardvel
% thres - Threshold speed in cm/s below which something is considered a
% stop
% Fs - number of frames length minimum to be considered stopped.
% ftol - max number of frames between 2 'stops' to be considered as one
% stop instead
stop_idx = moving_middle(find(diff(moving_middle)>1)+1);
% find stops without reward
frame_rate=31.25/numplanes;
max_reward_stop = 10*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
rew_idx = find(rewards); 
rew_stop_idx = [];
frame_tol = 10; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
for r = 1:length(rew_idx)
    if ~isempty(find(stop_idx-rew_idx(r)>=0-frame_tol & stop_idx-rew_idx(r) <max_reward_stop,1))
    rew_stop_idx(r) = stop_idx(find(stop_idx-rew_idx(r)>=0-frame_tol & stop_idx-rew_idx(r) <max_reward_stop,1));
    else
        rew_stop_idx(r) = NaN;
    end
end
rew_stop_idx = rew_stop_idx(~isnan(rew_stop_idx));
non_rew_stops = setxor(rew_stop_idx,stop_idx);
non_rew_stops = non_rew_stops(~isnan(non_rew_stops));
% figure; plot(velocity); hold on; plot(rew_stop_idx, velocity(rew_stop_idx), 'r*'); 
% plot(find(rewards),velocity(rewards),'y*'); plot(non_rew_stops, velocity(non_rew_stops), 'b*')
% binsize = 0.1; %half a second bins
% range = 6; %seconds back and forward in time
rewvel = zeros(ceil(range*2/binsize),size(dff,2),length(non_rew_stops));
for rr = 1:length(non_rew_stops)
    rewtime = timedFF(non_rew_stops(rr));
    currentrewchecks = find(timedFF>rewtime-range & timedFF<=rewtime+range);
    currentrewcheckscell = consecutive_stretch(currentrewchecks);
    currentrewardlogical = cellfun(@(x) ismember(non_rew_stops(rr),x),currentrewcheckscell);
    
    for bin = 1:ceil(range*2/binsize)
        testbin(bin) = round(-range+bin*binsize-binsize,13); %round to nearest 13 so 0 = 0 and not 3.576e-16
        currentidxt = find(timedFF>rewtime-range+bin*binsize-binsize & timedFF<=rewtime-range+bin*binsize);
        checks = consecutive_stretch(currentidxt);
        if ~isempty(checks{1})
            currentidxlogical = cellfun(@(x) max(ismember(x,currentrewcheckscell{currentrewardlogical})),checks);
            if sum(currentidxlogical)>0
                checkidx = checks{currentidxlogical};
                
                rewvel(bin,:,rr) = mean(dff(checkidx,:),1,'omitnan');
            else
                rewvel(bin,:,rr) = NaN;
            end
        else
            rewvel(bin,:,rr) = NaN;
        end
    end
    
end

meanrewvel = mean(rewvel,3,'omitnan');

binnedPerivelocity = meanrewvel';
allbins = testbin;
end
