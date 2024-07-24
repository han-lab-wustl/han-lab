clear all
%close all

mouse_id = 227;
addon = '_dark_reward';
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

pr_dir0 = uipickfiles;

oldbatch = 0; %input('if oldbatch press 1 else 0=');

% Initialize variables for averaging across days
all_roinorm_single_tracesCS = cell(1, 4);
all_roinorm_single_traces_roesmthCS = cell(1, 4);

 reg_name={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};

planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
roiplaneidx = cellfun(@(x) str2num(x(7)),reg_name,'UniformOutput',1);
[v, w] = unique( roiplaneidx, 'stable' );
duplicate_indices = setdiff( 1:numel(roiplaneidx), w )
color = planecolors(roiplaneidx);
color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false)

for alldays = 1:length(pr_dir0)
    planeroii = 0;
    for allplanes = 1:4
        plane = allplanes;
        Day = alldays;
        pr_dir1 = strcat(pr_dir0{Day}, '\suite2p');
        pr_dir = strcat(pr_dir1, '\plane', num2str(plane-1), '\reg_tif\', '');
        
        if exist(pr_dir, 'dir')
            cd(pr_dir)
            load('params.mat')
            
            if isfield(params, 'base_mean')
                base_mean = params.base_mean;
            else
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end
            
            if ~exist('lickVoltage')
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end
            
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
            
            if exist('forwardvelALL')
                speed_binned = forwardvelALL;
            end
            reward_binned = rewardsALL;
            temp = find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;
            
            speed_smth_1 = smoothdata(speed_binned, 'gaussian', gauss_win)';
            dop_smth = smoothdata(norm_base_mean, 'gaussian', gauss_win);
            
            if oldbatch == 1
                df_f = params.roibasemean2;
            else
                df_f = params.roibasemean3;
            end
            
            for roii = 1:size(df_f, 1)
                planeroii = planeroii + 1;
                
                roibase_mean = df_f{roii, 1};
                roimean_base_mean = mean(df_f{roii, 1});
                roinorm_base_mean = roibase_mean / roimean_base_mean;
                roidop_smth = smoothdata(roinorm_base_mean, 'gaussian', gauss_win);
                
                R = bwlabel(reward_binned > rew_thresh);
                rew_idx = find(R);
                rew_idx_diff = diff(rew_idx);
                temp = consecutive_stretch(rew_idx);
                rew_idx = cellfun(@(x) x(1), temp, 'UniformOutput', 1);
                
                short = (reward_binned == 1);
                short(rew_idx(find(rew_idx_diff < num_rew_win_frames))) = 0;
                short(rew_idx(find(rew_idx_diff < num_rew_win_frames) + 1)) = 0;
                
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
                
                roisingle_tracesCS = zeros(pre_win_frames + post_win_frames + 1, length(singlerew));
                roisingle_traces_roesmthCS = zeros(pre_win_framesALL + post_win_framesALL + 1, length(singlerew));
                for i = 1:length(singlerew)
                    currentnrrewCSidxperplane = find(timedFF >= utimedFF(singlerew(i)), 1);
                    roisingle_tracesCS(:, i) = roibase_mean(currentnrrewCSidxperplane - pre_win_frames:currentnrrewCSidxperplane + post_win_frames)';
                    roisingle_traces_roesmthCS(:, i) = speed_smth_1(singlerew(i) - pre_win_framesALL:singlerew(i) + post_win_framesALL)';
                end
                
                roinorm_single_tracesCS = roisingle_tracesCS ./ mean(roisingle_tracesCS(1:pre_win_frames, :));
                roinorm_single_tracesCS_smth = smoothdata(roinorm_single_tracesCS,'gaussian',5);
                roinorm_single_traces_roesmthCS = roisingle_traces_roesmthCS;
                
                % Aggregate data across days for each plane
                all_roinorm_single_tracesCS{plane} = [all_roinorm_single_tracesCS{plane}, roinorm_single_tracesCS_smth];
                all_roinorm_single_traces_roesmthCS{plane} = [all_roinorm_single_traces_roesmthCS{plane}, roinorm_single_traces_roesmthCS];
            end
        end
    end
end

% Average results over days for each plane
avg_roinorm_single_tracesCS = cellfun(@(x) mean(x, 2), all_roinorm_single_tracesCS, 'UniformOutput', false);
avg_roinorm_single_traces_roesmthCS = cellfun(@(x) nanmean(x, 2), all_roinorm_single_traces_roesmthCS, 'UniformOutput', false);

% Plot results for each plane
figure('Position', [100, 100, 800, 500]);
for plane = 1:4
    subplot(2, 2, plane);
    xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames;
    xaxSpeed=frame_time*(-pre_win_framesALL):frame_time:frame_time*post_win_framesALL;
    yax = avg_roinorm_single_tracesCS{plane}';
    se_yax = std(all_roinorm_single_tracesCS{plane}, [], 2) / sqrt(size(all_roinorm_single_tracesCS{plane}, 2));
    h10 = shadedErrorBar(xax, yax, se_yax, [], 1);
    hold on;
    plot(xaxSpeed, rescale(avg_roinorm_single_traces_roesmthCS{plane}, 0.99, 0.995), 'k', 'LineWidth', 2);
    %plot(xaxSpeed, rescale(avg_roinorm_single_traces_roesmthCS{plane}, 0.975, 0.98), 'k', 'LineWidth', 2);
    xt=[-3*ones(1,length(reg_name))];
    yt=[0.992:0.001:1];
    title(['Plane ', num2str(plane)]);
    xlabel('Seconds from CS');
    ylabel('dF/F');
    %legend({'Licks', '2% dF/F', 'Speed', 'Reward'});

    if sum(isnan(se_yax))~=length(se_yax)
        h10.patch.FaceColor = color{plane}; h10.mainLine.Color = color{plane}; h10.edge(1).Color = color{plane};
        h10.edge(2).Color=color{plane};
        text(xt(plane),yt(plane),reg_name{plane},'Color',color{plane},'Fontsize',10)
    end

    ylims = ylim;
    pls = plot([0 0],ylims,'--k','Linewidth',1);
    ylim(ylims)
    pls.Color(4) = 0.5;

end
