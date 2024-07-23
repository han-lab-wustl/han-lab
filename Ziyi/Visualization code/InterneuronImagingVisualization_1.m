close all
clear
mouse_id = 228;
% addon = '_dark_reward';
% mov_corr = [];
% stop_corr = [];
% mov_stop = [];
% mov_corr_success = [];
% stop_corr_success = [];
% mov_corr_prob = [];
% stop_corr_prob = [];
% mov_corr_fail = [];
% stop_corr_fail = [];
% cnt = 0;

pr_dir0 = uipickfiles;

allROI_data = cell(1, 4); % Initialize a cell array to collect all ROI data for each plane

% Constants
frame_rate = 7.8; % Frame rate in Hz
num_seconds = 5; % Number of seconds to display from -5 to +5 around the reward
pre_win = 5;
post_win = 5;
pre_win_frames = round(pre_win * frame_rate);
post_win_frames = round(post_win * frame_rate);

% Compute frame range for plotting
frame_step = frame_rate * num_seconds; % Number of frames that correspond to 5 seconds

% Assuming the middle point (0 seconds) is at frame 40 of the extracted -39 to +39 frames
mid_point = 40; % This is frame 0 on your original scale
frame_ticks = mid_point + (-num_seconds:num_seconds) * frame_rate; % Frame positions for each second from -5 to +5
time_ticks = -5:1:5; % Time labels from -5 to +5


for alldays = 1:length(pr_dir0)%
    planeroii = 0;
    for allplanes = 1:4 %%change plane info
        plane = allplanes;
        Day = alldays;
        pr_dir1 = pr_dir0{Day};
        pr_dir = strcat(pr_dir1,'\suite2p', '\plane', num2str(plane-1), '\reg_tif\');

        if exist(pr_dir, 'dir')

            cd(pr_dir)

            load(['file0000_XC_plane_' num2str(allplanes) '_roibyclick_F.mat'])
            %load(['file0000_cha_XC_plane_' num2str(allplanes) '_roibyclick_F.mat'])

            if ~exist('lickVoltage')
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end

            % Variable definitions here for clarity
            %frame_rate = 31.25;

            R = find(rewards);
            %R = find(solenoid2);
            temp = consecutive_stretch(R);
            reward_indices = cellfun(@(x) x(1), temp,'UniformOutput',1);

            df_f = mat2cell(dFF, size(dFF, 1), ones(1, size(dFF, 2)))';

            figure('position', [100 100 800 600])
            t = tiledlayout(ceil(sqrt(size(df_f,1))), ceil(sqrt(size(df_f,1))), 'TileSpacing', 'compact', 'Padding', 'compact');
            title(t, sprintf('All ROIs for Plane %d', plane));

            % Collecting and plotting data centered on reward location
            for roi_num = 1:size(df_f, 1)
                roi_data = df_f{roi_num};
                roi_event_data = [];
                for rew_idx = 1:length(reward_indices)
                    center_frame = reward_indices(rew_idx);
                    frame_range = max(1, center_frame - 39):min(size(roi_data, 1), center_frame + 39);
                    if size(frame_range,2)  ==  pre_win_frames + post_win_frames +1
                        roi_event_data = [roi_event_data; roi_data(frame_range)'];
                    end                    
                end
                norm_roi_event_data=roi_event_data./mean(roi_event_data(:,1:pre_win_frames),2);
                %norm_roi_event_data=roi_event_data./mean(roi_event_data(1:pre_win_frames),2);
                allROI_data{plane} = [allROI_data{plane}; roi_event_data]; % Collect data

                nexttile
                plot(-pre_win_frames:post_win_frames, smoothdata(mean(norm_roi_event_data, 1), 'gaussian',5))
                hold on
                xline(0, 'color', 'k', 'LineWidth', 2) % Vertical line at reward
                % Set x-axis ticks to show seconds instead of frames
                xticks(frame_ticks - mid_point) % Shift ticks to align with the plotted data range
                xticklabels(time_ticks) % Label ticks from -5 to +5 seconds
                title(sprintf('ROI %d', roi_num))
                xlabel('Seconds from Reward (-5 to 5)')
                ylabel('DeltaF/F')
            end

            ylabel(t, 'DeltaF/F')
            xlabel(t, 'Seconds from Reward (-5 to 5)')
        end
    end
end

grey = [0.5, 0.5, 0.5];  % Grey color
pink = [1, 0.4, 0.6];    % Pink color
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
planenames = {"SLM","SR","SP","SO"};

% planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
% roiplaneidx = cellfun(@(x) str2num(x(7)),reg_name,'UniformOutput',1);
% [v, w] = unique( roiplaneidx, 'stable' );
% duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
% color = planecolors(roiplaneidx);
% color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)

figure
% Plotting the mean DeltaF/F for each plane
for p = 1:4
    mean_data = mean(allROI_data{p}, 1);
    if isrow(mean_data)
        mean_data = mean_data';
    end
    norm_mean_data = mean_data./mean(mean_data(1:pre_win_frames,:));
    plot(-pre_win_frames:post_win_frames, smoothdata(norm_mean_data,"gaussian",5), 'Color', planecolors{p})

    hold on
   
end
    xticks(frame_ticks - mid_point) % Shift ticks to align with the plotted data range
    xticklabels(time_ticks) % Label ticks from -5 to +5 seconds

    %title(sprintf('Mean DeltaF/F across all ROIs for Plane %d', p))
    xlabel('Seconds relative to reward (-5 to 5)')
    ylabel('DeltaF/F')
    xline(0, 'color', 'k', 'LineWidth', 1) % Vertical line at reward
    legend(planenames)
