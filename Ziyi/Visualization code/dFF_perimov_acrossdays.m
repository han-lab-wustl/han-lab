
load('E:\Ziyi\Data\E247_Ach_GrabDA_red\Pavlovian\Dopamine\247_DA_workspace.mat')
%dop_alldays_planes_success_mov

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

reg_name={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

num_days = size(dop_alldays_planes_success_mov,2); % Number of days

% Initialize the 1x4 result cell array
normdFF_perimove = cell(1, size(dop_alldays_planes_success_mov, 2));
normdFF_perimove = cell(1, size(roe_alldays_planes_success_stop, 2));
roe_perimove = cell(1, size(roe_alldays_planes_success_mov, 2));

pre_win_frames = 39;

for j = 1:size(dop_alldays_planes_success_mov, 2) % Iterate over columns
    aggregatedMatrix = []; % Temporary storage for 4x79 matrix
    aggregatedMatrix2 = [];
    for i = 1:size(dop_alldays_planes_success_mov, 1) % Iterate over rows
        currentMatrix = dop_alldays_planes_success_mov{i, j}; % Access the matrix in the cell
        %norm_currentMatrix = currentMatrix ./ mean(currentMatrix(1:pre_win_frames, :), 1); % Normalize
        norm_currentMatrix = currentMatrix ./ mean(currentMatrix(:,:), 2); % Normalize
        aggregatedMatrix = [aggregatedMatrix; mean(norm_currentMatrix, 1)]; % Append mean row

        currentMatrix2 = roe_alldays_planes_success_mov{i, j}; % Access the matrix in the cell
        %norm_currentMatrix2 = currentMatrix2 ./ mean(currentMatrix2(1:pre_win_frames, :), 1); % Normalize
        norm_currentMatrix2 = currentMatrix2;
        aggregatedMatrix2 = [aggregatedMatrix2; mean(norm_currentMatrix2, 1)]; % Append mean row

    end
    normdFF_perimove{1, j} = aggregatedMatrix; % Store the aggregated matrix in the 1x4 cell
    roe_perimove{1, j} = aggregatedMatrix2;
end
% roinorm_single_tracesCS = roisingle_tracesCS ./ mean(roisingle_tracesCS(1:pre_win_frames, :));
% roinorm_single_tracesCS_smth = smoothdata(roinorm_single_tracesCS,'gaussian',5);
% roinorm_single_traces_roesmthCS = roisingle_traces_roesmthCS;
% roinorm_single_tracesCS_each_day= mean(roinorm_single_tracesCS_smth,2)

% Initialize the 1x4 result cell array
normdFF_peristop = cell(1, size(dop_alldays_planes_success_stop, 2));
roe_peristop = cell(1, size(roe_alldays_planes_success_stop, 2));

pre_win_frames = 39;

for j = 1:size(dop_alldays_planes_success_stop, 2) % Iterate over columns
    aggregatedMatrix = []; % Temporary storage for 4x79 matrix
    aggregatedMatrix2 = [];
    for i = 1:size(dop_alldays_planes_success_stop, 1) % Iterate over rows
        currentMatrix = dop_alldays_planes_success_stop{i, j}; % Access the matrix in the cell
        %norm_currentMatrix = currentMatrix ./ mean(currentMatrix(1:pre_win_frames, :), 1); % Normalize
        norm_currentMatrix = currentMatrix ./ mean(currentMatrix(:,:), 2); % Normalize
        aggregatedMatrix = [aggregatedMatrix; mean(norm_currentMatrix, 1)]; % Append mean row

        currentMatrix2 = roe_alldays_planes_success_stop{i, j}; % Access the matrix in the cell
        norm_currentMatrix2 = currentMatrix2; % Normalize
        %norm_currentMatrix2 = currentMatrix2 ./ mean(currentMatrix2(:,:), 1);
        aggregatedMatrix2 = [aggregatedMatrix2; mean(norm_currentMatrix2, 1)]; % Append mean row

    end
    normdFF_peristop{1, j} = aggregatedMatrix; % Store the aggregated matrix in the 1x4 cell
    roe_peristop{1, j} = aggregatedMatrix2;
end

% stop no reward
% Initialize the 1x4 result cell array
normdFF_peristop_no_reward = cell(1, size(dop_alldays_planes_success_stop_no_reward, 2));
roe_peristop_no_reward = cell(1, size(roe_alldays_planes_success_stop_no_reward, 2));

pre_win_frames = 39;

for j = 1:size(dop_alldays_planes_success_stop_no_reward, 2) % Iterate over columns
    aggregatedMatrix = []; % Temporary storage for 4x79 matrix
    aggregatedMatrix2 = [];
    for i = 1:size(dop_alldays_planes_success_stop_no_reward, 1) % Iterate over rows
        currentMatrix = dop_alldays_planes_success_stop_no_reward{i, j}; % Access the matrix in the cell
        %norm_currentMatrix = currentMatrix ./ mean(currentMatrix(1:pre_win_frames, :), 1); % Normalize
        norm_currentMatrix = currentMatrix ./ mean(currentMatrix(:,:), 2); % Normalize
        aggregatedMatrix = [aggregatedMatrix; mean(norm_currentMatrix, 1)]; % Append mean row

        currentMatrix2 = roe_alldays_planes_success_stop_no_reward{i, j}; % Access the matrix in the cell
        norm_currentMatrix2 = currentMatrix2; % Normalize
        %norm_currentMatrix2 = currentMatrix2 ./ mean(currentMatrix2(:,:), 1);
        aggregatedMatrix2 = [aggregatedMatrix2; mean(norm_currentMatrix2, 1)]; % Append mean row

    end
    normdFF_peristop_no_reward{1, j} = aggregatedMatrix; % Store the aggregated matrix in the 1x4 cell
    roe_peristop_no_reward{1, j} = aggregatedMatrix2;
end


% stop with reward
normdFF_peristop_reward = cell(1, size(dop_alldays_planes_success_stop_reward, 2));
roe_peristop_reward = cell(1, size(roe_alldays_planes_success_stop_reward, 2));

pre_win_frames = 39;

for j = 1:size(dop_alldays_planes_success_stop_reward, 2) % Iterate over columns
    aggregatedMatrix = []; % Temporary storage for 4x79 matrix
    aggregatedMatrix2 = [];
    for i = 1:size(dop_alldays_planes_success_stop_reward, 1) % Iterate over rows
        currentMatrix = dop_alldays_planes_success_stop_reward{i, j}; % Access the matrix in the cell
        %norm_currentMatrix = currentMatrix ./ mean(currentMatrix(1:pre_win_frames, :), 1); % Normalize
        norm_currentMatrix = currentMatrix ./ mean(currentMatrix(:,:), 2); % Normalize
        aggregatedMatrix = [aggregatedMatrix; mean(norm_currentMatrix, 1)]; % Append mean row

        currentMatrix2 = roe_alldays_planes_success_stop_reward{i, j}; % Access the matrix in the cell
        norm_currentMatrix2 = currentMatrix2; % Normalize
        %norm_currentMatrix2 = currentMatrix2 ./ mean(currentMatrix2(:,:), 1);
        aggregatedMatrix2 = [aggregatedMatrix2; mean(norm_currentMatrix2, 1)]; % Append mean row

    end
    normdFF_peristop_reward{1, j} = aggregatedMatrix; % Store the aggregated matrix in the 1x4 cell
    roe_peristop_reward{1, j} = aggregatedMatrix2;
end






for plane = 1:4

    % find_figure('Avg_dop')
    % subplot(2, 2, plane);
    xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames;

    fig1=find_figure('dFF_Days_peristarts')
    %figure
    fig1.Position = [100, 100, 600, 600]
    subplot(4,1,plane)

    imagesc(xax,1:num_days,normdFF_perimove{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    hold on
    title(reg_name(plane))


    fig2 = find_figure('roe_Days_peristart')
    fig2.Position = [100, 100, 600, 150]
    %figure
if plane ==1
    imagesc(xax,1:num_days,roe_perimove{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    cb2 = colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values
    colormap("gray")


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    ylabel(cb2,'Speed (cm/s)','FontSize',12,'Rotation',90)
end

    fig3 = find_figure('dFF_Days_peristop')
    fig3.Position = [100, 100, 600, 600]
    %figure
    subplot(4,1,plane)

    imagesc(xax,1:num_days,normdFF_peristop{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    hold on
    title(reg_name(plane))


    fig4 = find_figure('roe_Days_peristop')
    fig4.Position = [100, 100, 600, 150]
    %figure
   if plane==1
    imagesc(xax,1:num_days,roe_peristop{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    cb4 = colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values
    colormap("gray")


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    ylabel(cb4,'Speed (cm/s)','FontSize',12,'Rotation',90)
end

    fig5 = find_figure('dFF_Days_peristop_no_reward')
    fig5.Position = [100, 100, 600, 600]
    %figure
    subplot(4,1,plane)

    imagesc(xax,1:num_days,normdFF_peristop_no_reward{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    hold on
    title(reg_name(plane))


    fig6 = find_figure('roe_Days_peristop_no_reward')
    fig6.Position = [100, 100, 600, 150]
    %figure
    %subplot(4,1,plane)
    
    if plane==1
    imagesc(xax,1:num_days,roe_peristop_no_reward{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    cb6 = colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values
    colormap("gray")


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    ylabel(cb6,'Speed (cm/s)','FontSize',12,'Rotation',90)
    end


    fig7 = find_figure('dFF_Days_peristop_reward')
    fig7.Position = [100, 100, 600, 600]
    %figure
    subplot(4,1,plane)

    imagesc(xax,1:num_days,normdFF_peristop_reward{plane});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    colorbar
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    hold on
    title(reg_name(plane))


    fig8 = find_figure('roe_Days_peristop_with_reward')
    fig8.Position = [100, 100, 600, 150]
    %figure
    %subplot(4,1,plane)
    
    if plane==1 
    imagesc(xax,1:num_days,roe_peristop_reward{1});
    %imagesc(all_day_roinorm_single_tracesCS{plane}')
    cb8 = colorbar;
    y_tick_positions = 0.5:num_days-0.5;
    y_tick_labels = 1:num_days;
    %yticks(1:num_days)
    % Adjust the y-axis to represent days

    %colorbar; % Add a colorbar to show the scaling
    set(gca, 'YTick', y_tick_positions); % Set x-ticks at positions calculated
    set(gca, 'YTickLabel', y_tick_labels); % Label these ticks with corresponding time values
    colormap("gray")
    ylabel(cb8,'Speed (cm/s)','FontSize',12,'Rotation',90)


    % Add labels and title
    xlabel('Time (seconds)');
    ylabel('Day');
    title('Daily Data Representation Over Time');
    %hold on
    %title(reg_name(plane))
    end 





end
