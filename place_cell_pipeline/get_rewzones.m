function [rewzonenum] = get_rewzones(rewlocs, gainf)
rewzonenum = zeros(1,length(rewlocs)); % get rew zone identity too:  a=[{67:86} {101:120} {135:154}];
for kk = 1:length(rewlocs)
if rewlocs(kk)<=86*gainf
    rewzonenum(kk) = 1; % rew zone 1
elseif rewlocs(kk)<=120*gainf && rewlocs(kk)>=101*gainf
    rewzonenum(kk) = 2;
elseif rewlocs(kk)>=135*gainf
    rewzonenum(kk) = 3;
end
end