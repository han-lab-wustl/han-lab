clear all; close all
addon = '_dark_reward';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;

pr_dir0 = uipickfiles;


for alldays = 1:length(pr_dir0)%
    planeroii=0;
    for allplanes=1:4 %%change plane info
        plane=allplanes;
        Day=alldays;
        pr_dir1 = pr_dir0{Day};
        pr_dir=strcat(pr_dir1,'\suite2p', '\plane',num2str(plane-1),'\', 'reg_tif\')


        if exist( pr_dir, 'dir')

            cd (pr_dir)

            load(['file0000_XC_plane_' num2str(allplanes) '_roibyclick_F.mat'])

            if ~exist('lickVoltage')
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end

            numplanes=4;
            gauss_win=5;
            frame_rate=31.25;
            lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
            rew_thresh=0.001;
            sol2_thresh=1.5;
            num_rew_win_sec=5;%window in seconds for looking for multiple rewards
            rew_lick_win=10;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
            pre_win=5;%pre window in s for rewarded lick average
            post_win=5;%post window in s for rewarded lick average
            exclusion_win=10;%exclusion window pre and post rew lick to look for non-rewarded licks
            speed_thresh = 5; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 5; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            CSUStimelag = 0.5; %seconds between
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time/numplanes);
            post_win_framesALL=round(post_win/frame_time);
            pre_win_framesALL = round(pre_win/frame_time);
            pre_win_frames=round(pre_win/frame_time/numplanes);
            exclusion_win_frames=round(exclusion_win/frame_time);
            CSUSframelag_win_frames=round(CSUStimelag/frame_time);
            speedftol=10;


            if exist('forwardvelALL')
                speed_binned=forwardvelALL;
            end
            reward_binned=rewardsALL;
            % temporary artefact check and remove
            temp= find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';

            df_f=mat2cell(dFF,size(dFF,1),ones(1,size(dFF,2)))';


            find_figure('mean_image')
            subplot(2,2,allplanes),imagesc(frame)
            colormap("gray")
            hold on
            title(strcat('Plane',num2str(allplanes)))
            reg_name = {};
            for roii =1:size(df_f,1)
                reg_name{roii} = ['Plane ' num2str(allplanes) 'ROI ' num2str(roii)];
            end
            pre_post_rew = zeros(size(rewards,2),1);
            reward_indices = find(rewards);
            mod_rewards = rewards;
            % mod_rewards(reward_indices(1:10)) = 0;

            for i = 1:size(mod_rewards,2)
                if mod_rewards(1,i) == 1
                    pre_post_rew(i-pre_win_frames:i+post_win_frames,1) = 1;
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
                splitVectors = zeros(numSplits, pre_win_frames+post_win_frames+1);
                for split=1:numSplits
                    splitVectors(split,:) = ...
                        roi_dff((USIndexes(split)-pre_win_frames):(USIndexes(split)+post_win_frames));
                end
                mean_signal_rew_centered = mean(splitVectors,1);
                CSIndexes = find(solenoid2==1); % sound happens 4 samples before?

                nexttile
                plot(mean_signal_rew_centered)
                original_length = 80;  % Length of original data
                new_scale_length = 11; % Length from -5 to 5
                points_per_unit = original_length / new_scale_length;
                center_index = 40;
                % Generate indices for ticks from -5 to 5
                tick_indices = round(center_index + (-5:5) * points_per_unit);

                % Set the xticks and xticklabels
                xticks(tick_indices)
                xticklabels({'-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'})
                %xticks(-:20:40)
                %xlabel(round(-40/(frame_rate/numplanes)):2.5:round(40/(frame_rate/numplanes)))
                hold on
                xline(40,'color','k')
                xline(36, 'color', 'r')
                title(sprintf('ROI %d',roi_num))
            end
%            specific_name = sprintf('ROI avg over %d rewards in plane%d e227 sparse DA from %d',numSplits,planenum,recording_date);
         %   title(t,specific_name)
            ylabel(t,'DeltaF/F')
            xlabel(t,'50 samples before and after reward (US black line, CS red line)')
            


        end
    end
end





 