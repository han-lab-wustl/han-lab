function [com] = get_com_licks(trialnum, rewards, trials, licks, ybinned, rewloc, ...
    rewsize, success)
% center of mass location of licks - rewloc (start)
% trials = unique(trialnum>2);
% success = 0 or 1, if testing successful or failed trials
% filter out dark time licks
licks(ybinned<3)=0; % dark time is typically 1.5
com = zeros(1,length(trials));
for tr=1:length(trials)
    lick = licks(trialnum==trials(tr));
    ypos = ybinned(trialnum==trials(tr));
    rew = rewards(trialnum==trials(tr));
        % get position of rew
    if success==1
        yposafterrew = ypos(find(rew==1,1));
        yposlick = ypos(lick(ypos<=yposafterrew));
        avypos = mean(yposlick, 'omitnan');
        com(tr) = avypos-(rewloc-rewsize/2);
    elseif sum(lick)>0 % get trials when they lick
        yposlick = ypos(lick); % for failed trials, no need to filter consumption licks
        avypos = mean(yposlick, 'omitnan');
        com(tr) = avypos-(rewloc-rewsize/2);
    else 
        com(tr) = NaN;
    end
    
end

% smoothed licks in opp
% zone / (smoothed licks in rew zone + smoothed licks in opp zone)
end