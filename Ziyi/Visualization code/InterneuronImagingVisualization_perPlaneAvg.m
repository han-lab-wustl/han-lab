close all
mouse_id = 228;

pr_dir0 = uipickfiles;

allROI_data = cell(1, 4); % Initialize a cell array to collect all ROI data for each plane
all_roinorm_single_traces_roesmthCS = cell(1, 4);

% Constants
numplanes = 4;
gauss_win = 5;
frame_rate = 7.8; % Frame rate in Hz
num_seconds = 5; % Number of seconds to display from -5 to +5 around the reward
pre_win = 5;
post_win = 5;
pre_win_frames = round(pre_win * frame_rate);
post_win_frames = round(post_win * frame_rate);
rew_thresh = 0.001;
frame_time = 1 / frame_rate;
CSUStimelag = 0.5;
CSUSframelag_win_frames = round(CSUStimelag / frame_time);
pre_win_framesALL = round(pre_win / frame_time);
post_win_framesALL = round(post_win / frame_time);

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
        %pr_dir = strcat(pr_dir1, '\plane', num2str(plane-1), '\');
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
            
            if exist('forwardvelALL')
                speed_binned = forwardvelALL;
            end

            reward_binned = rewardsALL;
            temp = find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;

            speed_smth_1 = smoothdata(speed_binned, 'gaussian', gauss_win)';

            % Variable definitions here for clarity
            %frame_rate = 31.25;

            R = find(rewards);
            %R = find(solenoid2);
            temp = consecutive_stretch(R);
            reward_indices = cellfun(@(x) x(1), temp,'UniformOutput',1);

            df_f = mat2cell(dFF, size(dFF, 1), ones(1, size(dFF, 2)))';

            roibase_mean = df_f{allplanes, 1};
            R = bwlabel(reward_binned > rew_thresh);
            rew_idx = find(R);
            rew_idx_diff = diff(rew_idx);
            temp = consecutive_stretch(rew_idx);
            rew_idx = cellfun(@(x) x(1), temp, 'UniformOutput', 1);

            reward_binned = rewardsALL;
            temp = find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;


            short = (reward_binned == 1);
            short(rew_idx(find(rew_idx_diff < pre_win_frames))) = 0;
            short(rew_idx(find(rew_idx_diff < pre_win_frames) + 1)) = 0;

            single_rew = find(short);
            singlerew = single_rew(find(single_rew > pre_win_frames & single_rew < length(licksALL) - post_win_frames)) - CSUSframelag_win_frames;
            rm_idz = [];
            for i = 1:length(singlerew)
                currentnrrewCSidxperplane = find(timedFF >= utimedFF(singlerew(i)), 1);
                idz = currentnrrewCSidxperplane - pre_win_frames:currentnrrewCSidxperplane + post_win_frames;
                if idz(1) <= 0 || idz(end) > length(roibase_mean)
                    rm_idz = [rm_idz i];
                end
            end
            singlerew(rm_idz) = [];

            %roisingle_tracesCS = zeros(pre_win_frames + post_win_frames + 1, length(singlerew));
            roisingle_traces_roesmthCS = zeros(pre_win_framesALL + post_win_framesALL + 1, length(singlerew));
            for i = 1:length(singlerew)
                currentnrrewCSidxperplane = find(timedFF >= utimedFF(singlerew(i)), 1);
                %roisingle_tracesCS(:, i) = roibase_mean(currentnrrewCSidxperplane - pre_win_frames:currentnrrewCSidxperplane + post_win_frames)';
                roisingle_traces_roesmthCS(:, i) = speed_smth_1(singlerew(i) - pre_win_framesALL:singlerew(i) + post_win_framesALL)';
            end

            %roinorm_single_tracesCS = roisingle_tracesCS ./ mean(roisingle_tracesCS(1:pre_win_frames, :));
            roinorm_single_traces_roesmthCS = roisingle_traces_roesmthCS;

            % Aggregate data across days for each plane
            %all_roinorm_single_tracesCS{plane} = [all_roinorm_single_tracesCS{plane}, roinorm_single_tracesCS];
            all_roinorm_single_traces_roesmthCS{plane} = [all_roinorm_single_traces_roesmthCS{plane}, roinorm_single_traces_roesmthCS];

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
                all_roinorm_single_traces_roesmthCS{plane} = [all_roinorm_single_traces_roesmthCS{plane}, roinorm_single_traces_roesmthCS];
                % nexttile
                % plot(-pre_win_frames:post_win_frames, smoothdata(mean(norm_roi_event_data, 1), 'gaussian',5))
                % hold on
                % xline(0, 'color', 'k', 'LineWidth', 2) % Vertical line at reward
                % % Set x-axis ticks to show seconds instead of frames
                % xticks(frame_ticks - mid_point) % Shift ticks to align with the plotted data range
                % xticklabels(time_ticks) % Label ticks from -5 to +5 seconds
                % title(sprintf('ROI %d', roi_num))
                % xlabel('Seconds from Reward (-5 to 5)')
                % ylabel('DeltaF/F')
            end

            % ylabel(t, 'DeltaF/F')
            % xlabel(t, 'Seconds from Reward (-5 to 5)')
        end
    end
end

% grey = [0.5, 0.5, 0.5];  % Grey color
% pink = [1, 0.4, 0.6];    % Pink color
% planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
% planenames = {"SLM","SR","SP","SO"};

reg_name={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
roiplaneidx = cellfun(@(x) str2num(x(7)),reg_name,'UniformOutput',1);
[v, w] = unique( roiplaneidx, 'stable' );
duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
color = planecolors(roiplaneidx);
color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)

figure
% Plotting the mean DeltaF/F for each plane
avg_roinorm_single_traces_roesmthCS = cellfun(@(x) nanmean(x, 2), all_roinorm_single_traces_roesmthCS, 'UniformOutput', false);
for p = 1:4
    % mean_data = mean(allROI_data{p}, 1);
    % if isrow(mean_data)
    %     mean_data = mean_data';
    % end
    % norm_mean_data = mean_data./mean(mean_data(1:pre_win_frames,:));
    subplot(2,2,p)
    %plot(-pre_win_frames:post_win_frames, smoothdata(norm_mean_data,"gaussian",5), 'Color', planecolors{p})



    subplot(2, 2, p);
    xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames;
    xaxSpeed=frame_time*(-pre_win_framesALL):frame_time:frame_time*post_win_framesALL;
    yax = mean(allROI_data{p},1);
    se_yax = std(allROI_data{p}, [], 1) / sqrt(size(allROI_data{p}, 1));
    h10 = shadedErrorBar(xax, yax, se_yax, [], 1);
    hold on;
    plot(xaxSpeed, rescale(avg_roinorm_single_traces_roesmthCS{p}, 0.99, 0.995), 'k', 'LineWidth', 2);
    xt=[-3*ones(1,length(reg_name))];
    yt=[0.992:0.001:1];
    title(['Plane ', num2str(p)]);
    xlabel('Seconds from CS');
    ylabel('dF/F');
    %legend({'Licks', '2% dF/F', 'Speed', 'Reward'});

    if sum(isnan(se_yax))~=length(se_yax)
        h10.patch.FaceColor = color{p}; h10.mainLine.Color = color{p}; h10.edge(1).Color = color{p};
        h10.edge(2).Color=color{p};
        text(xt(p),yt(p),reg_name{p},'Color',color{p},'Fontsize',10)
    end

    ylims = ylim;
    pls = plot([0 0],ylims,'--k','Linewidth',1);
    ylim(ylims)
    pls.Color(4) = 0.5;
    hold on

end
xticks(frame_ticks - mid_point) % Shift ticks to align with the plotted data range
xticklabels(time_ticks) % Label ticks from -5 to +5 seconds

%title(sprintf('Mean DeltaF/F across all ROIs for Plane %d', p))
xlabel('Seconds relative to reward (-5 to 5)')
ylabel('DeltaF/F')
xline(0, 'color', 'k', 'LineWidth', 1) % Vertical line at reward
%legend(planenames)
