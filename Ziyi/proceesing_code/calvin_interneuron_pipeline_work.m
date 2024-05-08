%% Calvin trying to image signal with behavior
% using interneuron pipeline

clear
main_folder = 'C:\Users\workstation5\Documents\calvin pc\calvin scripts';
ris_folder = 'E:\Ziyi\Data\';

% so there are 10k frames per plane. each plane is recorded at 7.8 fps
% so that means it takes 1282.05 seconds for the whole recording

% to look at 1 sec before and after reward, that would be 7.8*2 samples

% throw out first ten rewards for 240227

recording_date = 240501;
planenum = 4;
save_name = 'e227_240501_roi_means_rewardcentered_plane4.png';
data_folder = fullfile(ris_folder, 'E227\240501_ZH\240501_ZH_000_000\suite2p');
plane0_folder = fullfile(data_folder,'\plane3\reg_tif');
cd(plane0_folder)
load('file000_cha_XC_plane_4_roibyclick_F')
cd(main_folder)


pre_post_rew = zeros(size(rewards,2),1);
reward_indices = find(rewards);
mod_rewards = rewards;
% mod_rewards(reward_indices(1:10)) = 0;

for i = 1:size(mod_rewards,2)
    if mod_rewards(1,i) == 1
        pre_post_rew(i-50:i+50,1) = 1;
    end
end

ideal_tile_size = sqrt(size(masks,1));
tile_size = ceil(ideal_tile_size);
figure('position',[381 287 1058 622]) 
t = tiledlayout(tile_size,tile_size);
for roi_num = 1:size(masks,1)
    roi_dff = dFF(:,roi_num);
    % R = corrcoef(roi_dff, forwardvel);

    USIndexes = find(mod_rewards==1); % water rewards
    numSplits = numel(USIndexes);
    splitVectors = zeros(numSplits, 101);
    for split=1:numSplits
        splitVectors(split,:) = ...
            roi_dff((USIndexes(split)-50):(USIndexes(split)+50));
    end
    mean_signal_rew_centered = mean(splitVectors,1);
    CSIndexes = find(solenoid2==1); % sound happens 4 samples before?

    nexttile
    plot(mean_signal_rew_centered)
    hold on
    xline(51,'color','k')
    xline(47, 'color', 'r')
    title(sprintf('ROI %d',roi_num))
end
specific_name = sprintf('ROI avg over %d rewards in plane%d e227 sparse DA from %d',numSplits,planenum,recording_date);
title(t,specific_name)
ylabel(t,'DeltaF/F')
xlabel(t,'50 samples before and after reward (US black line, CS red line)')
saveas(t,save_name)
close




%% other plotting stuff

% plot ROIs (masks)
for i = 1:size(masks,1)
    figure
    imagesc(squeeze(masks(i,:,:)))
end

% plot the dFF of each ROI
for i = 1:size(masks,1)
    plot(dFF(:,i)+(i*5))
    hold on
end


for i = 1:size(masks,1)
    figure
    imagesc(squeeze(nega_masks(i,:,:)))
end

% lets look at ROI 1

figure
plot(dFF(:,1)./10,'k')
hold on
plot(lickVoltage+.2,'r')
plot(forwardvel./300+.2,'b')
plot(rewards/10+0.4,'g')
title('roi1 in SLM e227 sparse DA from 240227')
ylabel('arbitrary units')
xlabel('samples (10k frames per plane)')
legend('dFF','licks','running speed','rewards')



%% For ed - look at larger ROI for sparse DA data


clear
main_folder = 'E:\Ziyi\results';
ris_folder = 'E:\Ziyi\Data\';
% so there are 10k frames per plane. each plane is recorded at 7.8 fps
% so that means it takes 1282.05 seconds for the whole recording

% to look at 1 sec before and after reward, that would be 7.8*2 samples

% throw out first ten rewards for 240227

recording_date = 240326;
planenum = 1;
mouse_num = 231;
save_name = 'e227_240503_single_roi_all_planes.png';
data_folder = fullfile(ris_folder, 'E227\240503_ZH\240503_ZH_000_000\suite2p');


% for first plane
plane0_folder = fullfile(data_folder,'\plane0\reg_tif');
cd(plane0_folder)
load('params')
cd(main_folder)

pre_post_rew = zeros(size(rewards,2),1);
mod_rewards = rewards;

for i = 1:size(mod_rewards,2)
    if mod_rewards(1,i) == 1
        pre_post_rew(i-50:i+50,1) = 1;
    end
end

roi_dff = params.roibasemean3{1,1};

USIndexes = find(mod_rewards==1); % water rewards
numSplits = numel(USIndexes);
splitVectors = zeros(numSplits, 101);
for split=1:numSplits
    splitVectors(split,:) = ...
        roi_dff((USIndexes(split)-50):(USIndexes(split)+50));
end
mean_signal_rew_centered_pl1 = mean(splitVectors,1);


% for second plane
plane1_folder = fullfile(data_folder,'\plane1\reg_tif');
cd(plane1_folder)
load('params')
cd(main_folder)

pre_post_rew = zeros(size(rewards,2),1);
mod_rewards = rewards;

for i = 1:size(mod_rewards,2)
    if mod_rewards(1,i) == 1
        pre_post_rew(i-50:i+50,1) = 1;
    end
end

roi_dff = params.roibasemean3{1,1};

USIndexes = find(mod_rewards==1); % water rewards
numSplits = numel(USIndexes);
splitVectors = zeros(numSplits, 101);
for split=1:numSplits
    splitVectors(split,:) = ...
        roi_dff((USIndexes(split)-50):(USIndexes(split)+50));
end
mean_signal_rew_centered_pl2 = mean(splitVectors,1);


% for third plane
plane2_folder = fullfile(data_folder,'\plane2\reg_tif');
cd(plane2_folder)
load('params')
cd(main_folder)

pre_post_rew = zeros(size(rewards,2),1);
mod_rewards = rewards;

for i = 1:size(mod_rewards,2)
    if mod_rewards(1,i) == 1
        pre_post_rew(i-50:i+50,1) = 1;
    end
end

roi_dff = params.roibasemean3{1,1};

USIndexes = find(mod_rewards==1); % water rewards
numSplits = numel(USIndexes);
splitVectors = zeros(numSplits, 101);
for split=1:numSplits
    splitVectors(split,:) = ...
        roi_dff((USIndexes(split)-50):(USIndexes(split)+50));
end
mean_signal_rew_centered_pl3 = mean(splitVectors,1);


% for fourth plane
plane3_folder = fullfile(data_folder,'\plane3\reg_tif');
cd(plane3_folder)
load('params')
cd(main_folder)

pre_post_rew = zeros(size(rewards,2),1);
reward_indices = find(rewards);
mod_rewards = rewards;

for i = 1:size(mod_rewards,2)
    if mod_rewards(1,i) == 1
        pre_post_rew(i-50:i+50,1) = 1;
    end
end

roi_dff = params.roibasemean3{1,1};

USIndexes = find(mod_rewards==1); % water rewards
numSplits = numel(USIndexes);
splitVectors = zeros(numSplits, 101);
for split=1:numSplits
    splitVectors(split,:) = ...
        roi_dff((USIndexes(split)-50):(USIndexes(split)+50));
end
mean_signal_rew_centered_pl4 = mean(splitVectors,1);
CSIndexes = find(solenoid2==1); % sound happens 4 samples before?




figure('position',[381 287 1058 622]);
t = tiledlayout(2,2);

nexttile
plot(mean_signal_rew_centered_pl1)
hold on
xline(51,'color','k')
xline(47, 'color', 'r')
title(sprintf('ROI avg over %d rewards in SLM (plane1)',numSplits))

ylabel('DeltaF/F')
xlabel('50 samples before and after reward (US black, CS red)')

nexttile
plot(mean_signal_rew_centered_pl2)
hold on
xline(51,'color','k')
xline(47, 'color', 'r')
title(sprintf('ROI avg over %d rewards in SR (plane2)',numSplits))

ylabel('DeltaF/F')
xlabel('50 samples before and after reward (US black, CS red)')

nexttile
plot(mean_signal_rew_centered_pl3)
hold on
xline(51,'color','k')
xline(47, 'color', 'r')
title(sprintf('ROI avg over %d rewards in SP (plane3)',numSplits))

ylabel('DeltaF/F')
xlabel('50 samples before and after reward (US black, CS red)')

nexttile
plot(mean_signal_rew_centered_pl4)
hold on
xline(51,'color','k')
xline(47, 'color', 'r')
title(sprintf('ROI avg over %d rewards in SO (plane4)',numSplits))

ylabel('DeltaF/F')
xlabel('50 samples before and after reward (US black, CS red)')

title(t,sprintf('e%d from %d',mouse_num,recording_date))

saveas(gcf,save_name)
close

