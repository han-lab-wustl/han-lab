function [binnedPerireward,allbins,rewdFF] = perirewardbinnedactivity(dFF,rewards,timedFF,range,binsize)
Rewindx = find(rewards);
% binsize = 0.1; %half a second bins
% range = 6; %seconds back and forward in time
rewdFF = zeros(ceil(range*2/binsize),size(dFF,2),length(Rewindx));
for rr = 1:length(Rewindx)
    rewtime = timedFF(Rewindx(rr));
    currentrewchecks = find(timedFF>rewtime-range & timedFF<=rewtime+range);
    currentrewcheckscell = consecutive_stretch(currentrewchecks);
    currentrewardlogical = cellfun(@(x) ismember(Rewindx(rr),x),currentrewcheckscell);
    
    for bin = 1:ceil(range*2/binsize)
        testbin(bin) = round(-range+bin*binsize-binsize,13); %round to nearest 13 so 0 = 0 and not 3.576e-16
        currentidxt = find(timedFF>rewtime-range+bin*binsize-binsize& timedFF<=rewtime-range+bin*binsize);
        checks = consecutive_stretch(currentidxt);
        if ~isempty(checks{1})
            currentidxlogical = cellfun(@(x) max(ismember(x,currentrewcheckscell{currentrewardlogical})),checks);
            if sum(currentidxlogical)>0
                checkidx = checks{currentidxlogical};
                
                rewdFF(bin,:,rr) = nanmean(dFF(checkidx,:),1); %ZD changed based on input structure 
                                                                                        % (per cell that is tracked over days)
            else
                rewdFF(bin,:,rr) = NaN;
            end
        else
            rewdFF(bin,:,rr) = NaN;
        end
    end
    
end

meanrewdFF = nanmean(rewdFF,3);
rewdFF = squeeze(rewdFF);
% figure()
y = 1:1:size(dFF,2);
x = -1*range:binsize:range-binsize;
% subplot(2,1,1)
% imagesc(x,y,meanrewdFF')
% subplot(2,1,2)
% plot(x,mean(meanrewdFF,2),'b')
% hold on
% plot(x,mean(meanrewdFF,2)-(std(meanrewdFF,0,2)/sqrt(size(meanrewdFF,2))),'b:')
% plot(x,mean(meanrewdFF,2)+(std(meanrewdFF,0,2)/sqrt(size(meanrewdFF,2))),'b:')

for n = 1:size(dFF,2)
    normmeanrewdFF(:,n) = rescale(meanrewdFF(:,n),0,1);
end
% figure()
% subplot(2,1,1)
% imagesc(x,y,normmeanrewdFF')
%
%
% title('Normalized')
% subplot(2,1,2)
% plot(x,nanmean(normmeanrewdFF,2),'b')
% hold on
% plot(x,mean(normmeanrewdFF,2)-(std(normmeanrewdFF,0,2)/sqrt(size(normmeanrewdFF,2))),'b:')
% plot(x,mean(normmeanrewdFF,2)+(std(normmeanrewdFF,0,2)/sqrt(size(normmeanrewdFF,2))),'b:')

binnedPerireward = meanrewdFF';
allbins = testbin;
end
