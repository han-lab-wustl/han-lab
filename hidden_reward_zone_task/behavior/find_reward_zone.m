function [rewzone] = find_reward_zone(rewloc)
% based on 7/14 hrz version
if rewloc <= 86 
    rewzone = 1;
elseif rewloc >= 101 && rewloc <=120
    rewzone = 2;
elseif rewloc >=135
    rewzone = 3;
end
end