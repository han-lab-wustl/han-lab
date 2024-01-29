figure; 
combinedspeed = [];
combineddFF = [];
corrs = [];
for n = 1:length(single_lick_idx)
    currtime = find(timedFF>= utimedFF(single_lick_idx(n)),1):find(timedFF>= utimedFF(single_lick_idx(n))+5,1);
    combinedspeed = [combinedspeed smoothdata(forwardvel(currtime),'gaussian',5)];
    combineddFF = [combineddFF params.roibasemean3{1}(currtime)];
end
plot(combineddFF)
yyaxis right
plot(combinedspeed)