function [ur_lick_rate,r_lick_rate] = get_rewarded_and_unrewarded_licks(VR, eprng, ...
    mask, rewloc, ep)
% function to get rewarded and unrewarded licks for opto behavior analysis
%rewarded vs. unrewarded licks
% 5 cm before rew and 10 cm after rew are considered rewarded licks
% mask is for filtering between trials
trialnum = VR.trialNum(eprng);
pos = VR.ypos(eprng); pos=pos(mask);
rewarded = find((pos>=rewloc(ep)-5) & (pos<rewloc(ep)+10)); % indices
unrewarded = setdiff([1:length(pos)], rewarded);
lick = VR.lick(eprng); lick=lick(mask);
ur_lick = lick(unrewarded);
r_lick = lick(rewarded);
ur_lick_rate = sum(ur_lick)/size(unique(trialnum(mask)),2);
r_lick_rate = sum(r_lick)/size(unique(trialnum(mask)),2);

end