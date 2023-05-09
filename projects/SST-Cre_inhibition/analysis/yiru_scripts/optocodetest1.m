Fc0 = F(find(iscell(:,1)),:);
Fnc0 = Fneu(find(iscell(:,1)),:);
% dFF = redo_dFF(Fc0',31.25,20,Fnc0');
VRselectstartendsplit;

optostartpoints = [10077 13123 19735 21792 23211 24520 26331 33739 35144 36477 38941 40461 41740 44018];
optostoppoints = NaN(size(optostartpoints));
optomeandFF = NaN(size(dFF,2),length(optostartpoints));
for o = 1:length(optostartpoints)
    temptime = timedFF(optostartpoints(o));
    optostoppoints(o) = find(timedFF>temptime+5,1);
    currenttime = optostartpoints(o):optostoppoints(o);
    currenttime(F(395,currenttime)>1000) =[];
    optomeandFF(:,o)=nanmean(dFF(currenttime,:));
end
optocurrent = [6 10 15 6 10 15 6 10 15 4 4 4 4 4];
nopowerstart = [14436 19091 28302];
nopowerstop = NaN(size(nopowerstart));
nopoweroptomeandFF = NaN(size(dFF,2),length(nopowerstart));
for o = 1:length(nopowerstart)
    temptime = timedFF(nopowerstart(o));
    nopowerstop(o) = find(timedFF>temptime+5,1);
    currenttime = nopowerstart(o):nopowerstop(o);
    currenttime(1:2:end) = [];
    nopoweroptomeandFF(:,o) = nanmean(dFF(currenttime,:));
end
x = [0 0 0 optocurrent];
y = [nopoweroptomeandFF optomeandFF];
figure;

for c = [1 2 4 5 6]
    scatter(x+(c/10-3/10),y(c,:),"filled")
    hold on
    mn = accumarray((x+1)',y(c,:)',[],@mean);
    mn(mn==0) = NaN;

    std = accumarray((x+1)',y(c,:)',[],@std)./sqrt([3 0 0 0 5 0 3 0 0 0 3 0 0 0 0 3]');
    scatter((0:15)+(c/10-3/10),mn,20,'kd')
    errorbar((0:15)+(c/10-3/10),mn,std,'k-','Capsize',0)
end