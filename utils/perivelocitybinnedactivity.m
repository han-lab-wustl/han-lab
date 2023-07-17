function [binnedPerivelocity,allbins,rewvel] = perivelocitybinnedactivity(velocity,rewards,timedFF,range,binsize)
%%%
%find velocity aligned by rewards
%%%
Rewindx = find(rewards);
% binsize = 0.1; %half a second bins
% range = 6; %seconds back and forward in time
rewvel = zeros(ceil(range*2/binsize),length(Rewindx));
for rr = 1:length(Rewindx)
    rewtime = timedFF(Rewindx(rr));
    currentrewchecks = find(timedFF>rewtime-range & timedFF<=rewtime+range);
    currentrewcheckscell = consecutive_stretch(currentrewchecks);
    currentrewardlogical = cellfun(@(x) ismember(Rewindx(rr),x),currentrewcheckscell);
    
    for bin = 1:ceil(range*2/binsize)
        testbin(bin) = round(-range+bin*binsize-binsize,13); %round to nearest 13 so 0 = 0 and not 3.576e-16
        currentidxt = find(timedFF>rewtime-range+bin*binsize-binsize & timedFF<=rewtime-range+bin*binsize);
        checks = consecutive_stretch(currentidxt);
        if ~isempty(checks{1})
            currentidxlogical = cellfun(@(x) max(ismember(x,currentrewcheckscell{currentrewardlogical})),checks);
            if sum(currentidxlogical)>0
                checkidx = checks{currentidxlogical};
                
                rewvel(bin,rr) = nanmean(velocity(checkidx));
            else
                rewvel(bin,rr) = NaN;
            end
        else
            rewvel(bin,rr) = NaN;
        end
    end
    
end

meanrewvel = nanmean(rewvel,2);

binnedPerivelocity = meanrewvel';
allbins = testbin;
end
