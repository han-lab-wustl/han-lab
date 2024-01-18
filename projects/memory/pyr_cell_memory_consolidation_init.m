clear all; 
% load('Y:\hrz_consolidation\e217\10\240118_ZD_000_001\suite2p\plane0\Fall.mat')
load('Y:\hrz_consolidation\e228\9\240116_ZD_000_003\suite2p\plane0\Fall.mat')
figure; 
F_ = F(logical(iscell(:,1)),:);
plot(ybinned); hold on
plot(find(licks), ybinned(licks), 'r.')
plot(find(rewards), ybinned(logical(rewards)), 'b.')
yyaxis right 
plot(dFF(:,randi([1 size(dFF,2)]),:),'g');
% plot(F_(randi([1 size(F_,1)]),:),'g');