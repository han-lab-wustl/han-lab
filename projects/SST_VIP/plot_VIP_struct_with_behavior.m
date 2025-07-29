% plotting with gerardo's vip structure
% 5 days of HRZ
clear all; close all;
load("\\storage1.ris.wustl.edu\ebhan\Active\dzahra\old_stuff\gerardo_VIP_rawdata_intswanalysisv9.mat")
%%
dy = 1; % day
fall = VIP_ints(1).NR.day{1,dy}.all;
pln = 1; c = 9;
figure; axes(1) = subplot(3,1,1); 
plt = [fall.Falls{:}]';  % 13333 x (12+10+10) = 13333 x 32
% for cc=1:size(plt,1)
%     plot(plt(cc,:)); hold on;
% end

imagesc(plt(:,:)); hold on;

axes(2) = subplot(3,1,2); 
ybin = fall.ybinned{1}(:,c);
plot(ybin, 'k'); xlim([0 length(ybin)])
hold on; 
plot(find(fall.rewards{1}(:,1)==1), ybin(fall.rewards{1}(:,1)==1), 'go')
plot(find(fall.licks{1}(:,1)), ybin(logical(fall.licks{1}(:,1))), 'r.'); xlim([0 length(ybin)])

axes(3) = subplot(3,1,3); plot(fall.Forwards{1}(:,1)); xlim([0 length(ybin)])
linkaxes(axes, 'x')
%%
% export per day to mat file
savepath = 'Z:\vip_gcamp_data_reformatted\';

for dy = 1:length(VIP_ints(1).NR.day)
    fall = VIP_ints(1).NR.day{1,dy}.all;

    % DFF matrix
    plt = [fall.Falls{:}]';
    save(fullfile(savepath, sprintf('dff_day%d_hrz.mat', dy)), 'plt')

    % Y position
    ybin = fall.ybinned{1}(:,9);  % Change column index if needed
    save(fullfile(savepath, sprintf('ypos_day%d_hrz.mat', dy)), 'ybin')

    % Reward binary vector
    rew = fall.rewards{1}(:,1);
    save(fullfile(savepath, sprintf('rew_day%d_hrz.mat', dy)), 'rew')

    % Lick binary vector
    lick = fall.licks{1}(:,1);
    save(fullfile(savepath, sprintf('lick_day%d_hrz.mat', dy)), 'lick')

    % Velocity
    vel = fall.Forwards{1}(:,1);
    save(fullfile(savepath, sprintf('vel_day%d_hrz.mat', dy)), 'vel')

    % Trial number
    tr = fall.trialNum{1}(:,1);
    save(fullfile(savepath, sprintf('trialnum_day%d_hrz.mat', dy)), 'tr')

    % Reward location change indicator
    changerewloc = fall.rewardLocation{1}(:,1);
    save(fullfile(savepath, sprintf('changerewloc_day%d_hrz.mat', dy)), 'changerewloc')

    % Time
    time = fall.time{1}(:,1);
    save(fullfile(savepath, sprintf('time_day%d_hrz.mat', dy)), 'time')
end

%%
% plot activity in dark time
plt=plt';
mask = ybin<20;
dt = plt(:,mask);
postrew = plt(:,ybin>150);
figure; imagesc(normalize([dt postrew],1)); hold on
xline(size(dt,2), 'k--', 'LineWidth',5)