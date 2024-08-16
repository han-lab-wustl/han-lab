clear all; % Clear workspace variables
%close all; % Close all figure windows

% Mouse information and file selection
mouse_id = 227;
addon = '_dark_reward';
pr_dir0 = uipickfiles; % User selects directories
oldbatch = 0;

% Initialize variables for correlation and event data
mov_corr = []; 
stop_corr = []; 
mov_stop = [];
mov_corr_success = []; 
stop_corr_success = [];
mov_corr_prob = []; 
stop_corr_prob = [];
mov_corr_fail = [];
stop_corr_fail = [];
cnt = 0;

% Initialize variables for averaging across days
all_roinorm_single_tracesCS = cell(1, 4);
all_roinorm_single_traces_roesmthCS = cell(1, 4);
speed_event_data_acrossdays = [];

% Constants
frame_rate = 7.8; % Frame rate in Hz
num_seconds = 5; % Number of seconds to display from -5 to +5 around the reward
pre_win = 5; % Pre-reward window in seconds
post_win = 5; % Post-reward window in seconds
pre_win_frames = round(pre_win * frame_rate);
post_win_frames = round(post_win * frame_rate);

% Compute frame range for plotting
frame_step = frame_rate * num_seconds; % Number of frames that correspond to 5 seconds
mid_point = 40; % Middle point (0 seconds) in extracted frames
frame_ticks = mid_point + (-num_seconds:num_seconds) * frame_rate; % Frame positions for each second from -5 to +5
time_ticks = -5:1:5; % Time labels from -5 to +5
reg_name = {'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

% Plane colors for plotting
planecolors = {[0 0 1], [0 1 0], [204 164 61]/256, [231 84 128]/256};

% Determine plane indices
roiplaneidx = cellfun(@(x) str2num(x(7)), reg_name, 'UniformOutput', 1);
[v, w] = unique(roiplaneidx, 'stable');
duplicate_indices = setdiff(1:numel(roiplaneidx), w);
color = planecolors(roiplaneidx);
color(duplicate_indices) = cellfun(@(x) x/2, color(duplicate_indices), 'UniformOutput', false);

% Loop over all selected days
for alldays = 1:length(pr_dir0)
    planeroii = 0;
    speed_event_data = [];
    for allplanes = 1:4
        plane = allplanes;
        Day = alldays;
        pr_dir1 = strcat(pr_dir0{Day}, '\suite2p');
        pr_dir = strcat(pr_dir1, '\plane', num2str(plane-1), '\reg_tif\', '');

        if exist(pr_dir, 'dir')
            cd(pr_dir);
            load('params.mat');

            % Check if base_mean exists in params
            if isfield(params, 'base_mean')
                base_mean = params.base_mean;
            else
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name);
                if ~exist('forwardvel', 'var')
                    forwardvel = speed_binned;
                end
            end

            if ~exist('lickVoltage', 'var')
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name);
                if ~exist('forwardvel', 'var')
                    forwardvel = speed_binned;
                end
            end

            % Parameters for analysis
            numplanes = 4;
            gauss_win = 5;
            frame_rate = 31.25;
            lickThresh = -0.085;
            rew_thresh = 0.001;
            sol2_thresh = 1.5;
            num_rew_win_sec = 5;
            rew_lick_win = 10;
            pre_win = 5;
            post_win = 5;
            exclusion_win = 10;
            speed_thresh = 5;
            Stopped_frame = 15;
            max_reward_stop = 5 * frame_rate;
            frame_tol = 5;
            CSUStimelag = 0.5;
            frame_time = 1 / frame_rate;
            num_rew_win_frames = round(num_rew_win_sec / frame_time);
            rew_lick_win_frames = round(rew_lick_win / frame_time);
            post_win_frames = round(post_win / frame_time / numplanes);
            post_win_framesALL = round(post_win / frame_time);
            pre_win_framesALL = round(pre_win / frame_time);
            pre_win_frames = round(pre_win / frame_time / numplanes);
            exclusion_win_frames = round(exclusion_win / frame_time);
            CSUSframelag_win_frames = round(CSUStimelag / frame_time);
            speedftol = 10;

            mean_base_mean = mean(base_mean);
            norm_base_mean = base_mean;

            if exist('forwardvelALL', 'var')
                speed_binned = forwardvelALL;
            end
            reward_binned = rewardsALL;
            temp = find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;

            % Smooth speed data
            roi_speed_data_smth = smoothdata(speed_binned, 'gaussian', 1)';

            if mod(length(roi_speed_data_smth), 4) ~= 0
                error('Data length must be divisible by 4.');
            end

            % Reshape speed data into 4-row matrix and compute mean for each column
            reshaped_data = reshape(roi_speed_data_smth, 4, []);
            binned_speed = mean(reshaped_data, 1)';
            dop_smth = smoothdata(norm_base_mean, 'gaussian', gauss_win);

            % Select the appropriate df_f based on batch
            if oldbatch == 1
                df_f = params.roibasemean2;
            else
                df_f = params.roibasemean3;
            end

            % Process each ROI
            for roii = 1:size(df_f, 1)
                planeroii = planeroii + 1;
                roibase_mean = df_f{roii, 1};
                roimean_base_mean = mean(df_f{roii, 1});
                roinorm_base_mean = roibase_mean / roimean_base_mean;
                roidop_smth = smoothdata(roinorm_base_mean, 'gaussian', gauss_win);

                % Find epochs with at least 3 consecutive successful trials
                epoch_start = find_first_consecutive_numbers(find(trialnum == 0));
                epoch_start_frame = [1 epoch_start length(trialnum)];
                epoch_period = cell(1, length(epoch_start_frame) - 1);

                for i = 1:length(epoch_period)
                    epoch_period{i} = [epoch_start_frame(i) epoch_start_frame(i+1) - 1];
                end

                consecutive_success_csIndx = [];

                for i = 1:length(epoch_period)
                    single_epoch_start = epoch_period{i}(1);
                    single_epoch_end = epoch_period{i}(2);
                    single_epoch_trialnum = trialnum(single_epoch_start:single_epoch_end);
                    single_epoch_cs = solenoid2(single_epoch_start:single_epoch_end);
                    csNum = find(single_epoch_cs);
                    consecutive_success_indx = find_indices_with_consecutive_before(single_epoch_trialnum(csNum));
                    single_epoch_consecutive_success_csIndx = csNum(consecutive_success_indx) + single_epoch_start - 1;
                    consecutive_success_csIndx = [consecutive_success_csIndx single_epoch_consecutive_success_csIndx];
                end

                singlerew = consecutive_success_csIndx;
                roi_event_data = [];

                % Extract event data around rewards
                for rew_idx = 1:length(singlerew)
                    center_frame = singlerew(rew_idx);
                    frame_range = max(1, center_frame - 39):min(size(roibase_mean, 1), center_frame + 39);
                    if size(frame_range,2) == pre_win_frames + post_win_frames + 1
                        roi_event_data = [roi_event_data; roibase_mean(frame_range)'];
                        if allplanes == 1
                            speed_event_data = [speed_event_data; binned_speed(frame_range)'];
                        end
                    end
                end

                roi_event_data = roi_event_data';
                roi_event_data_normed = roi_event_data ./ mean(roi_event_data(1:pre_win_frames, :));
                roi_event_data_smth = smoothdata(roi_event_data_normed, 'gaussian', 5);
                speed_event_data_smth = smoothdata(speed_event_data, 'gaussian', 1);

                % Aggregate data across days for each plane
                all_roinorm_single_tracesCS{plane} = [all_roinorm_single_tracesCS{plane}, roi_event_data_smth];
            end
        end
    end

    single_day_mean_speed = mean(speed_event_data_smth, 1);
    speed_event_data_acrossdays = [speed_event_data_acrossdays, single_day_mean_speed'];
end

mean_speed = mean(speed_event_data_acrossdays, 2);

% Plot results for each plane
figure('Position', [100, 100, 800, 500]);
for p = 1:4
    subplot(2,2,p);
    hold on;
    title(reg_name{p});
    xlabel('seconds from CS');
    ylabel('dF/F');

    mean_data = mean(all_roinorm_single_tracesCS{p}, 2);
    norm_mean_data = mean_data ./ mean(mean_data(1:pre_win_frames));
    yax = norm_mean_data;
    se_yax = std(all_roinorm_single_tracesCS{p}, [], 2) ./ sqrt(size(all_roinorm_single_tracesCS{p}, 2))';
    xax = -pre_win_frames:post_win_frames;
    h10 = shadedErrorBar(xax', yax, se_yax, [], 1);

    xt = [-3 * ones(1, length(reg_name))];
    yt = [0.992:0.001:1];
    
    xticks(frame_ticks - mid_point); % Shift ticks to align with the plotted data range
    xticklabels(time_ticks); % Label ticks from -5 to +5 seconds

    if sum(isnan(se_yax)) ~= length(se_yax)
        h10.patch.FaceColor = planecolors{p}; 
        h10.mainLine.Color = planecolors{p}; 
        h10.edge(1).Color = planecolors{p};
        h10.edge(2).Color = planecolors{p};
        %text(xt(p), yt(p), reg_name{p}, 'Color', planecolors{p}, 'FontSize', 10);
    end
    
    plot(xax, rescale(mean_speed, 0.996, 0.998), 'k', 'LineWidth', 2);

    ylims = ylim;
    pls = plot([0 0], ylims, '--k', 'LineWidth', 1);
    ylim(ylims);
    pls.Color(4) = 0.5;  
end
