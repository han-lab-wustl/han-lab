clear all
close all
mouse_id=227;
addon = '_dark_reward';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;

pr_dir0 = uipickfiles;

oldbatch=0;%input('if oldbatch press 1 else 0=');

for alldays = 1:length(pr_dir0)
    planeroii=0;
    for allplanes=1:4
        plane=allplanes;
        Day=alldays;
        pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
        pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','');
        
        if exist( pr_dir, 'dir')
            
            cd (pr_dir)
            
            load('params.mat')
            if isfield(params,'base_mean')
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
            
            mean_base_mean=mean(base_mean);
            norm_base_mean=base_mean;
            
            if exist('forwardvelALL')
                speed_binned=forwardvelALL;
            end
            reward_binned=rewardsALL;
            temp= find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0;
            
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
            dop_smth=smoothdata(norm_base_mean,'gaussian',gauss_win);
            
            if oldbatch==1
                df_f=params.roibasemean2;
            else
                df_f=params.roibasemean3;
            end
            
            mean_image = find_figure('mean_image')
            subplot(2,2,allplanes),imagesc(params.mimg)
            colormap("gray")
            hold on
            title(strcat('Plane',num2str(allplanes)))
            
            for roii = 1:size(df_f,1)
                planeroii=planeroii+1;
                
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                
                raw_figure = find_figure('raw_figure')
                reg_name={'Plane 1 SLM','Plane 2 SR','Plane 3 SP','Plane 4 SO'};
                
                planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
                roiplaneidx = cellfun(@(x) str2num(x(7)),reg_name,'UniformOutput',1);
                [v, w] = unique( roiplaneidx, 'stable' );
                duplicate_indices = setdiff( 1:numel(roiplaneidx), w );
                color = planecolors(roiplaneidx);
                color(duplicate_indices)=cellfun(@(x) x/2 ,color(duplicate_indices) ,'UniformOutput' ,false);
                
                if planeroii == 1
                    plot(utimedFF,rescale(speed_smth_1,-1,0),'k','LineWidth',1.5)
                end
                
                hold on
                plot(timedFF,rescale(roidop_smth,planeroii-1,planeroii),'color',color{planeroii},'LineWidth',1.5)
                plot([1090 1090]+5,[planeroii-0.5 (0.02)/(range(roidop_smth))+planeroii-0.5],'k-','LineWidth',2)
                reward_binned(find(reward_binned))=1;
                if planeroii == length(reg_name)
                    plot(utimedFF,rescale(reward_binned,0,length(reg_name)),'LineWidth',1.5)
                    plot(utimedFF,rescale(solenoid2ALL,0,length(reg_name)),'LineWidth',1.5)
                    plot(utimedFF,rescale(licksALL,length(reg_name),length(reg_name)+0.5),'color',[.7 .7 .7],'LineWidth',1.5)
                    if exist("urew_solenoidALL",'var')
                        unrew_single=find(urew_solenoidALL);
                        unrew_single(find(diff(unrew_single) == 1)) = 0;
                        unrew_single=unrew_single(find(unrew_single));
                        nunrew=zeros(1,length(reward_binned));
                        nunrew(unrew_single)=1;
                        plot(utimedFF,rescale(nunrew,0,length(reg_name)),'k','LineWidth',1.5)
                    end
                end
                
                legend({'Licks','Plane1 SLM','2% dF/F','Plane 1 SR','2% dF/F','Plane3 SP','2% dF/F','Plane4 SO','Speed','Reward'})
                ylabel('dF/F')
                
               
                
                find_figure('mean_image')
                
                if roii == 1
                    subplot(2,2,allplanes),plot(params.newroicoords{roii}(:,1),params.newroicoords{roii}(:,2),'w-','LineWidth',1.5)
                else
                    subplot(2,2,allplanes), plot(params.newroicoords{roii}(:,1),params.newroicoords{roii}(:,2),'w--','LineWidth',1.5)
                end
                text(mean(params.newroicoords{roii}(:,1)), mean(params.newroicoords{roii}(:,2)),reg_name{planeroii},'Color', 'w', 'FontSize', 12)
                

                
                % Compute CS for all ROI's
                R = bwlabel(reward_binned>rew_thresh);%label rewards, ascending
                rew_idx=find(R);%get indexes of all rewards
                rew_idx_diff=diff(rew_idx);%difference in reward index from last
                temp = consecutive_stretch(rew_idx);
                rew_idx = cellfun(@(x) x(1), temp,'UniformOutput',1); %If the threshold over counts the same reward
               
                short= (reward_binned == 1);%logical for rewards that happen less than x frames from last reward. 0 = single rew.
                short(rew_idx(find(rew_idx_diff<num_rew_win_frames))) = 0;
                short(rew_idx(find(rew_idx_diff<num_rew_win_frames)+1)) = 0;
               
                % Single rewards
                single_rew=find(short);
                single_idx=[];single_lick_idx=[]; single_lick_gap = [];
                singlerew = single_rew(find(single_rew>pre_win_frames&single_rew<length(licksALL)-post_win_frames))-CSUSframelag_win_frames;
                
                rm_idz=[];
                for i=1:length(singlerew)
                    currentnrrewCSidxperplane = find(timedFF>=utimedFF(singlerew(i)),1);
                    idz=currentnrrewCSidxperplane-pre_win_frames:currentnrrewCSidxperplane+post_win_frames;
                    if idz(1)<=0 || idz(end)>length(roibase_mean)
                        rm_idz=[rm_idz i];
                    end
                end
                singlerew(rm_idz)=[];
                
                roisingle_tracesCS=zeros(pre_win_frames+post_win_frames+1,length(singlerew));
                roisingle_traces_roesmthCS=zeros(pre_win_framesALL+post_win_framesALL+1,length(singlerew));
                for i=1:length(singlerew)
                    currentnrrewCSidxperplane = find(timedFF>=utimedFF(singlerew(i)),1);
                    roisingle_tracesCS(:,i)=roibase_mean(currentnrrewCSidxperplane-pre_win_frames:currentnrrewCSidxperplane+post_win_frames)';%lick at pre_win_frames+1
                    roisingle_traces_roesmthCS(:,i)=speed_smth_1(singlerew(i)-pre_win_framesALL:singlerew(i)+post_win_framesALL)';
                end
                
                roinorm_single_tracesCS=roisingle_tracesCS./mean(roisingle_tracesCS(1:pre_win_frames,:));
                roinorm_single_traces_roesmthCS=roisingle_traces_roesmthCS;
                
                save('params','roinorm_single_tracesCS','roinorm_single_traces_roesmthCS','-append')
                
                % Plot for single reward CS
                samePlaneOverlay = find_figure('same plane ROIs overlay');
                subplot(2,2,allplanes),
                hold on;
                title(['Plane',num2str(allplanes), ' Single rewards CS']);
                xlabel('seconds from CS')
                ylabel('dF/F')
                xax=frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames;
                yax=mean(roinorm_single_tracesCS,2)';
                se_yax=std(roinorm_single_tracesCS,[],2)./sqrt(size(roinorm_single_tracesCS,2))';
                h10=shadedErrorBar(xax,yax,se_yax,[],1);
                xax=frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes;
                plot(xax,rescale(mean(roinorm_single_traces_roesmthCS,2),0.99,0.995),'k','LineWidth',2);
                xt=[-3*ones(1,length(reg_name))];
                yt=[0.992:0.001:1];
                
                if sum(isnan(se_yax))~=length(se_yax)
                    h10.patch.FaceColor = color{planeroii}; h10.mainLine.Color = color{planeroii}; h10.edge(1).Color = color{planeroii};
                    h10.edge(2).Color=color{planeroii};
                    text(xt(planeroii),yt(planeroii),reg_name{planeroii},'Color',color{planeroii},'Fontsize',10)
                end
                
                ylims = ylim;
                pls = plot([0 0],ylims,'--k','Linewidth',1);
                ylim(ylims)
                pls.Color(4) = 0.5;
                % 
                % % Save the same plane ROIs overlay
                % filename = sprintf('Mouse%d_Day%d_Plane%d_ROI%d_same_plane_overlay.jpg', mouse_id, Day, allplanes, roii);
                % saveas(gcf, filename)
            end
              % Save the raw_figure
                filename = sprintf('Mouse%d_Day%d_Plane%d_ROI%d_Fl_Speed_Reward.jpg', mouse_id, Day, allplanes, roii);
                saveas(mean_image, filename)
             % Save the same plane ROIs overlay
                filename = sprintf('Mouse%d_Day%d_Plane%d_ROI%d_same_plane_overlay.jpg', mouse_id, Day, allplanes, roii);
                saveas(samePlaneOverlay, filename)
                
        end
    end
end
