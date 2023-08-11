% plotting with gerardo's vip structure
% 5 days of HRZ
clear all; close all;
load("Z:\VIP_intswanalysisv9.mat")
%%
dy = 1; % day
fall = VIP_ints(1).NR.day{1,dy}.all;

figure; subplot(3,1,1); 
imagesc(rescale(fall.Falls{1},0,1)'); 
%probes
rectangle('position',[find(trialnum(rng)<3, 1) 0 ...
    length(find(trialnum(rng)<3)) cells], ... % just picked max for visualization
                'EdgeColor',[0 0 0 0],'FaceColor',[0 .5 .5 0.3])
subplot(3,1,2); 
ybin = fall.ybinned{1}(:,1);
plot(ybin, 'k'); xlim([0 length(ybin)])
hold on; 
plot(find(fall.rewards{1}(:,1)==1), ybin(fall.rewards{1}(:,1)==1), 'go')
plot(find(fall.licks{1}(:,1)), ybin(logical(fall.licks{1}(:,1))), 'r.'); xlim([0 length(ybin)])

subplot(3,1,3); plot(fall.Forwards{1}(:,1)); xlim([0 length(ybin)])

% per cell
for cell=1:size(fall.Falls{1},2)
    figure; subplot(3,1,1); 
    plot(fall.Falls{1}(:,cell),'g'); xlim([0 length(fall.Falls{1}(:,1))])
    subplot(3,1,2); 
    ybin = fall.ybinned{1}(:,1);
    plot(ybin, 'k'); xlim([0 length(ybin)])
    hold on; 
    plot(find(fall.rewards{1}(:,1)==1), ybin(fall.rewards{1}(:,1)==1), 'go')
    plot(find(fall.licks{1}(:,1)), ybin(logical(fall.licks{1}(:,1))), 'r.')
    
    subplot(3,1,3); plot(fall.Forwards{1}(:,1)); xlim([0 length(fall.Falls{1}(:,1))])
end