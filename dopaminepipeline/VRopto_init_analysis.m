e% clear all
close all
clearvars -except pr_dir0
mouse_id=195;
addon = '_opto';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
pr_dir0 = uipickfiles;

% oldbatch=input('if oldbatch press 1 else 0=');
oldbatch = 0;
% dop_allsuc_stop_no_reward = NaN(length(pr_dir0),4,79);
for alldays = 1:length(pr_dir0)%[3:12 14:19]%[3:12 13:19]%[3 5:1]%[5:12 14]%[5:12 14]%1:5%26:30%1:33%31:32%30%27:30%21%[1:21]%[1:2 4:5 12:22]%[8:21]%[2:4 6:11]%[1:2 4:5 12:20]%%[1:21]%%
%     clearvars -except alldays addon mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr ...
%         mov_corr_success stop_corr_success mov_corr_prob stop_corr_prob mov_corr_fail stop_corr_fail...
%         c  alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr dop_suc_movint dop_suc_stopint roe_suc_movint roe_suc_stopint ...
%         dop_allsuc_mov dop_allsuc_stop roe_allsuc_mov roe_allsuc_stop ...
%         dop_allfail_mov dop_allfail_stop roe_allfail_mov roe_allfail_stop...
%         dop_alldays_planes_success_mov dop_alldays_planes_fail_mov dop_alldays_planes_success_stop dop_alldays_planes_fail_stop...
%         roe_alldays_planes_success_mov roe_alldays_planes_fail_mov roe_alldays_planes_success_stop roe_alldays_planes_fail_stop...
%         subp days_check cnt dop_alldays_planes_perireward roe_alldays_planes_perireward dop_allsuc_perireward roe_allsuc_perireward...
%         pr_dir0 dop_alldays_planes_success_stop_no_reward dop_alldays_planes_success_stop_reward...
%         dop_allsuc_stop_no_reward dop_allsuc_stop_reward roe_allsuc_stop_no_reward roe_allsuc_stop_reward...
%         dop_alldays_planes_double_0 roe_alldays_planes_double_0 dop_alldays_planes_perireward_double roe_alldays_planes_perireward_double...
%         dop_allsuc_perireward_double roe_allsuc_perireward_double dop_allsuc_perireward_double_se roe_allsuc_perireward_double_se...
%         day_labels roi_dop_alldays_planes_success_mov roi_dop_alldays_planes_success_stop roi_dop_alldays_planes_success_stop_no_reward...
%         roi_dop_alldays_planes_success_stop_reward roi_dop_allsuc_mov roi_dop_allsuc_stop roi_dop_allsuc_stop_no_reward roi_dop_allsuc_stop_reward...
%         roi_dop_alldays_planes_perireward_0 roi_dop_alldays_planes_perireward_double_0 roi_dop_alldays_planes_perireward roi_dop_alldays_planes_perireward_double...
%         roi_dop_allsuc_perireward roi_dop_allsuc_perireward_double roi_dop_allsuc_perireward_se roi_dop_allsuc_perireward_double_se...
%         roe_alldays_planes_perireward_0 roe_alldays_planes_success_stop_no_reward roe_alldays_planes_stop_reward roe_alldays_planes_perireward_double_0...
%         dop_alldays_planes_perireward_double_0 dop_alldays_planes_perireward_0 roe_alldays_planes_success_stop_reward...
%         roi_dop_alldays_planes_periCS roi_dop_alldays_planes_peridoubleCS roi_roe_alldays_planes_periCS roi_roe_alldays_planes_peridoubleCS roi_dop_allsuc_perirewardCS...
%         roi_dop_allsuc_perireward_doubleCS roi_roe_allsuc_perirewardCS roi_roe_allsuc_perireward_doubleCS...
%         roi_dop_alldays_planes_perinrlicks_0  roi_dop_alldays_planes_perinrlicks...
%         roi_dop_alldays_planes_periUS roi_dop_alldays_planes_peridoubleUS roi_dop_allsuc_perirewardUS roi_dop_allsuc_perireward_doubleUS...
%         roi_roe_alldays_planes_periUS roi_roe_alldays_planes_peridoubleUS roi_roe_allsuc_perirewardUS roi_roe_allsuc_perireward_doubleUS...
%         roi_dop_alldays_planes_unreward_single_0 roi_dop_alldays_planes_unreward_single roi_dop_allsuc_unreward_single roi_dop_allsuc_unreward_single_se...
%         roi_roe_alldays_planes_unrewardCS roi_roe_allsuc_perireward_unrewardCS oldbatch  roi_dop_alldays_planes_rewusonly_single_0...
%         roi_dop_alldays_planes_rewusonly_single_0 roi_dop_alldays_planes_rewusonly_single roi_dop_allsuc_rewusonly_single   roi_dop_allsuc_rewusonly_single_se....
%         roi_roe_alldays_planes_rewusonly roi_roe_allsuc_rewusonly roi_dop_alldays_planes_perireward_usonly_0   roi_dop_alldays_planes_perireward_usonly...
%         roi_dop_allsuc_perireward_usonly  roi_dop_allsuc_perireward_usonly_se roi_dop_alldays_planes_perirewarddoublemUS_CS_0    roi_dop_alldays_planes_perirewarddoublemUS_CS...
%          roi_dop_allsuc_perirewarddoublemUS_CS  roi_dop_allsuc_perirewarddoublemUS_CS_se roi_roe_alldays_planes_perirewarddoublemUS_CS    roi_roe_allsuc_perirewarddoublemUS_CS
    
    clearvars lickVoltage forwardvel
%     
    
    close all
    dir_s2p = struct2cell(dir([pr_dir0{alldays} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
    
    planeroicount = 0;
    cnt=cnt+1;
    Day=alldays;
    for allplanes=1:size(planefolders,2)
        plane=allplanes;
        
%         pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
%         pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','')
        pr_dir=strcat(planefolders{2,allplanes},'\plane',num2str(plane-1),'\reg_tif\','')
        
        
        
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
%             if allplanes == 1 %%% CHANGE 1 FOR DEFAULT
%                 forwardvel1= forwardvel;
%                 rewards1 = rewards;
%             else
%                 forwardvel = forwardvel1;
%                 rewards = rewards1;
%             end
            %%%%%%%%%%%%%%%
       numplanes=size(planefolders,2);
            gauss_win=5;
            frame_rate=31.25;
            lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
            rew_thresh=0.001;
            sol2_thresh=1.5;
            num_rew_win_sec=5;%window in seconds for looking for multiple rewards
            rew_lick_win=10;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
            pre_win=5;%pre window in s for rewarded lick average
            post_win=12;%post window in s for rewarded lick average
            exclusion_win=10;%exclusion window pre and post rew lick to look for non-rewarded licks
            speed_thresh = 12; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 14; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
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

            
            
            mean_base_mean=mean(base_mean);
            
            norm_base_mean=base_mean;
            
            if exist('forwardvelALL')
                speed_binned=forwardvelALL;
            end
            reward_binned=rewardsALL;
            % temporary artefact check and remove
            temp= find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0; 
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
            dop_smth=smoothdata(norm_base_mean,'gaussian',gauss_win);
            
            if length(timedFF)>length(dop_smth)
                timedFF = timedFF(1:end-1);
            end
            
            
            
            figure;
            plot(utimedFF,rescale(speed_smth_1,min(dop_smth)-range(dop_smth),min(dop_smth)),'LineWidth',1.5)
            hold on
            plot(timedFF,dop_smth,'LineWidth',1.5)
            plot(utimedFF,rescale(reward_binned,min(dop_smth),max(dop_smth)),'LineWidth',1.5)
            legend({'Reward','Speed','Dopamine'})
            ylabel('dF/F')
            savefig('Fl_Speed_Reward.fig')
            
            %%%%%%%%%%%%%%%%%%%calculate single traces
            %%%%%%rewarded licks
            
            
           
                supraLick=licksALL;

  
            %%%%%%%%%%%%%%
            %non-rewarded licks

            
            nr_lick=bwlabel(supraLick');

            nr_lick(1:exclusion_win_frames,1)=0;%get rid of non-rewarded licks at start, otherwise will crash when grab traces
            nr_lick(end-exclusion_win_frames:end,1)=0;%same at end
            nr_lick=bwlabel(nr_lick);
            nr_lick_idx=[];
            for i=1:max(nr_lick)
                nr_lick_idx(i)= (find(nr_lick==i,1,'first'));%
            end
            
            
            save('params','nr_lick_idx','nr_lick','-append');

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                
                if size(forwardvelALL,2) == 1
                    moving_middle=get_moving_time_V2(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)),speed_thresh,Stopped_frame);
                else
                    moving_middle=get_moving_time_V2(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5))',speed_thresh,Stopped_frame);
                end
                mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                
                
                
                
                idx_rm=(mov_success_tmpts- pre_win_framesALL)<=0;
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                
                
                idx_rm=(mov_success_tmpts+post_win_framesALL)>length(utimedFF);
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                allmov_success=NaN(1,length(norm_base_mean));
                allmov_success(moving_middle) = 1;
                dop_success_perimov=[]; roe_success_perimov=[];
                for stamps=1:length(mov_success_tmpts)
                    currentnrstartidxperplane = find(timedFF>=utimedFF(mov_success_tmpts(stamps)),1);
                    dop_success_perimov(stamps,:)= norm_base_mean(currentnrstartidxperplane-pre_win_frames:currentnrstartidxperplane+post_win_frames);
                    roe_success_perimov(stamps,:)= forwardvelALL(mov_success_tmpts(stamps)-pre_win_framesALL:mov_success_tmpts(stamps)+post_win_framesALL);
                    
                end
                
                
                %%%%stopping success trials
                
                stop_success_tmpts =  moving_middle(find(diff(moving_middle)>1))+1;
                
                
                idx_rm= (stop_success_tmpts - pre_win_framesALL) <=0;
                rm_idx=find(idx_rm==1);
                stop_success_tmpts(rm_idx)=[];
                
                
                
                idx_rm=(stop_success_tmpts+post_win_framesALL)>length(utimedFF);
                rm_idx=find(idx_rm==1)
                stop_success_tmpts(rm_idx)=[];
                
                
                
                save('params','stop_success_tmpts','mov_success_tmpts','-append')
                
                
                
           dop_success_peristop=[]; roe_success_peristop=[];
                allstop_success=NaN(1,length(base_mean));
                allstop_success(setxor(1:length(base_mean),moving_middle)) = 1;
                for stamps=1:length(stop_success_tmpts)
                    currentnrrewstopidxperplane = find(timedFF>=utimedFF(stop_success_tmpts(stamps)),1);
                    dop_success_peristop(stamps,:)= base_mean(currentnrrewstopidxperplane-pre_win_frames:currentnrrewstopidxperplane+post_win_frames);
                    roe_success_peristop(stamps,:)= forwardvelALL(stop_success_tmpts(stamps)-pre_win_framesALL:stop_success_tmpts(stamps)+post_win_framesALL);
                end
                
                %save all peri locomotion variables
                save('params','dop_success_perimov','dop_success_peristop','roe_success_perimov','roe_success_peristop','-append')
                
                
                alldays=alldays;
                dop_alldays_planes_success_mov{alldays,allplanes}=dop_success_perimov;
                dop_alldays_planes_success_stop{alldays,allplanes}=dop_success_peristop;
                
                roe_alldays_planes_success_mov{alldays,allplanes}=roe_success_perimov;
                roe_alldays_planes_success_stop{alldays,allplanes}=roe_success_peristop;
                
                dop_allsuc_mov(alldays,allplanes,:)=mean(dop_success_perimov);
                dop_allsuc_stop(alldays,allplanes,:)=mean(dop_success_peristop);
                
                roe_allsuc_mov(alldays,allplanes,:)=mean(roe_success_perimov,1);
                roe_allsuc_stop(alldays,allplanes,:)=mean(roe_success_peristop,1);

                %%%%% moving
                find_figure(strcat(mouse_id,'_perilocomotion moving','_plane',num2str(allplanes)));
                subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),
                ylabel('dF/F')
                
                hold on
                subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,dop_success_perimov);
                hold on
                subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(dop_success_perimov,1),'k','LineWidth',2);
                
                legend(['n = ',num2str(size(dop_success_perimov,1))])
                hold on
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes,rescale(mean(roe_success_perimov,1),0.985,0.99),'k--','LineWidth',2)
                
                text(-5, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
                text(5, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
      
            %%%%%%%%%%%%%%%%%%%%%%%%%%% for regions of interest
            
            if oldbatch==1
                
                df_f=params.roibasemean2;
            else
                df_f=params.roibasemean3;
            end
            
            for roii = 1:size(df_f,1)
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                
                figure;
                plot(utimedFF,rescale(speed_smth_1,min(roidop_smth)-range(roidop_smth),min(roidop_smth)),'LineWidth',1.5)
                hold on
                plot(timedFF,roidop_smth,'LineWidth',1.5)
                plot(utimedFF,rescale(reward_binned,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
                legend({'Reward','Speed','Dopamine'})
                ylabel('dF/F')
                savefig(['ROI_' num2str(roii) 'Fl_Speed_Reward.fig'])

                %%%%%%%%%%%%%%
                %%
                %non-rewarded licks
                
                nr_lick=bwlabel(supraLick);

                nr_lick(1:exclusion_win_frames)=0;%get rid of non-rewarded licks at start, otherwise will crash when grab traces
                nr_lick(end-exclusion_win_frames:end)=0;%same at end
                nr_lick=bwlabel(nr_lick);
                for i=1:max(nr_lick)
                    nr_lick_idx(i)= (find(nr_lick==i,1,'first'));%
                end
                roinr_traces=zeros(pre_win_frames+post_win_frames+1,length(nr_lick_idx));
                roenr_traces=zeros(pre_win_framesALL+post_win_framesALL+1,length(nr_lick_idx));
                for i=1:length(nr_lick_idx)
                    currentidx = find(timedFF>=utimedFF(nr_lick_idx(i)),1);
                    roinr_traces(:,i)=roibase_mean(currentidx-pre_win_frames:currentidx+post_win_frames)';%lick at pre_win_frames+1
                    roenr_traces(:,i) =   forwardvelALL(nr_lick_idx(i)-pre_win_framesALL:nr_lick_idx(i)+post_win_framesALL);
                end
                roinorm_nr_traces=roinr_traces./mean(roinr_traces(1:pre_win_frames,:));
                
                figure;
                hold on;
                title(strcat('ROI',num2str(roii),'_Non-rewarded licks'));
                xlabel('seconds from non-rewarded lick')
                ylabel('dF/F')
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_nr_traces,'Color',[.8 .8 .8]);
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,nanmean(roinorm_nr_traces,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_nr_traces,2))])
                
                
                
                currfile=strcat(['ROI_' num2str(roii) '_non_rew_licks.fig']);
                savefig(currfile)
                
                
                
                save('params','roinorm_nr_traces','roinr_traces','nr_lick_idx','nr_lick','-append');
                    
                    if size(forwardvelALL,2) == 1
                        moving_middle=get_moving_time_V2(forwardvelALL,speed_thresh,Stopped_frame);
                    else
                        moving_middle=get_moving_time_V2(forwardvelALL',speed_thresh,Stopped_frame);
                    end
                    mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                    idx_rm=(mov_success_tmpts- pre_win_framesALL)<=0;
                    rm_idx=find(idx_rm==1)
                    mov_success_tmpts(rm_idx)=[];
                    
                    idx_rm=(mov_success_tmpts+post_win_framesALL)>length(forwardvelALL);
                    rm_idx=find(idx_rm==1)
                    mov_success_tmpts(rm_idx)=[];
                    
                    roiallmov_success=NaN(1,length(forwardvelALL));
                    roiallmov_success(moving_middle) = 1;
                    roidop_success_perimov=[]; roe_success_perimov=[];
                    for stamps=1:length(mov_success_tmpts)
                        currentidx = find(timedFF>=utimedFF(mov_success_tmpts(stamps)),1);
                        roidop_success_perimov(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                    end
                    
                    
                    %%%%stopping success trials
                    %
                    stop_success_tmpts =  moving_middle(find(diff(moving_middle)>1))+1;
                    
                    
                    idx_rm= (stop_success_tmpts - pre_win_framesALL) <=0;
                    rm_idx=find(idx_rm==1);
                    stop_success_tmpts(rm_idx)=[];
                    
                    idx_rm=(stop_success_tmpts+post_win_framesALL)>length(forwardvelALL);
                    rm_idx=find(idx_rm==1)
                    stop_success_tmpts(rm_idx)=[];

                    roidop_success_peristop=[];
                    roiallstop_success=NaN(1,length(forwardvelALL));
                    roiallstop_success(setxor(1:length(forwardvelALL),moving_middle)) = 1;
                    for stamps=1:length(stop_success_tmpts)
                        currentidx =  find(timedFF>=utimedFF(stop_success_tmpts(stamps)),1);
                        roidop_success_peristop(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                    end
                    
                    
                    % Peri Stimulation 
                stim_idx = find(stims);
                    
                                    roistim_traces=zeros(pre_win_frames+post_win_frames+1,length(stim_idx));
                                    roe_success_peristim=zeros(pre_win_framesALL+post_win_framesALL+1,length(stim_idx));
                                    lick_success_peristim = zeros(pre_win_framesALL+post_win_framesALL+1,length(stim_idx));
                for i=1:length(stim_idx)
                    currentidx = find(timedFF>=utimedFF(stim_idx(i)),1);
                    if currentidx+post_win_frames<=length(roibase_mean)
                    roistim_traces(:,i)=roibase_mean(currentidx-pre_win_frames:currentidx+post_win_frames)';%lick at pre_win_frames+1
                    roe_success_peristim(:,i)= forwardvelALL(stim_idx(i)-pre_win_framesALL:stim_idx(i)+post_win_framesALL);
                    lick_success_peristim(:,i) = licksALL(stim_idx(i)-pre_win_framesALL:stim_idx(i)+post_win_framesALL);
                    else
                        roistim_traces(:,i)=[roibase_mean(currentidx-pre_win_frames:length(roibase_mean));NaN(size(roistim_traces,1)-length(currentidx-pre_win_frames:length(roibase_mean)),1)]';%lick at pre_win_frames+1
                        roe_success_peristim(:,i)= [forwardvelALL(stim_idx(i)-pre_win_framesALL:length(forwardvelALL)) NaN(1,size(roe_success_peristim,1)-length(stim_idx(i)-pre_win_framesALL:length(forwardvelALL)))];
                       lick_success_peristim(:,i) = [licksALL(stim_idx(i)-pre_win_framesALL:length(forwardvelALL)) NaN(1,size(roe_success_peristim,1)-length(stim_idx(i)-pre_win_framesALL:length(forwardvelALL)))];
                    end
                end
                roinorm_stim_traces=roistim_traces./mean(roistim_traces(1:pre_win_frames,:));
                
                figure;
                hold on;
                title(strcat('ROI',num2str(roii),'_Stim'));
                xlabel('seconds from non-rewarded lick')
                ylabel('dF/F')
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_stim_traces,'Color',[.8 .8 .8]);
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,nanmean(roinorm_stim_traces,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_stim_traces,2))])
                
                
                
                currfile=strcat(['ROI_' num2str(roii) '_stim.fig']);
                savefig(currfile)
                    
                save('params','roinorm_stim_traces','roistim_traces','roe_success_peristim','lick_success_peristim','stim_idx','-append');
                    %% SAVING SECTION
                    
                    
                    roi_nr_traces{roii} = roinr_traces;
                    roi_norm_nr_traces{roii} = roinorm_nr_traces;
                    roi_stim_traces{roii} = roistim_traces;
                    roi_norm_stim_traces{roii} = roinorm_stim_traces;
                    roi_dop_success_perimov{roii} = roidop_success_perimov;
                    roi_dop_success_peristop{roii} = roidop_success_peristop;
                    roi_roe_success_peristim{roii} = roe_success_peristim;
                    roi_lick_success_peristim{roii} = lick_success_peristim;
                    
                    save('params','roi_nr_traces','roi_norm_nr_traces','roi_stim_traces','roi_norm_stim_traces','roi_dop_success_perimov','roi_dop_success_peristop','roi_roe_success_peristim','-append')
                    %save allDAYS peri locomotion dopamine variables
                    planeroicount = planeroicount + 1;
                    planeroicount
                    alldays=alldays;
                    roi_dop_alldays_planes_success_mov{alldays,planeroicount}=roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2);
                    roi_dop_alldays_planes_success_stop{alldays,planeroicount}=roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2);
                    
                    roi_dop_allsuc_mov(alldays,planeroicount,:)=mean(roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2),1);
                    roi_dop_allsuc_stop(alldays,planeroicount,:)=mean(roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2),1);
                    
                    roi_dop_alldays_planes_perinrlicks_0{alldays,planeroicount}=roinr_traces;
                    
                    roi_dop_alldays_planes_perinrlicks{alldays,planeroicount}=roinorm_nr_traces;
                    
                    roi_roe_alldays_planes_perinrlicks{alldays,planeroicount}=roenr_traces;
                    
                    roi_dop_alldays_planes_peristim_0{alldays,planeroicount}=roistim_traces;
                    
                    roi_dop_alldays_planes_peristim{alldays,planeroicount}=roinorm_stim_traces;
                    
                    roi_roe_alldays_planes_peristim{alldays,planeroicount}=roe_success_peristim;
                    
                    roi_lick_alldays_planes_peristim{alldays,planeroicount} = lick_success_peristim;

                
            end
            
            % SAVING ALL ROI VARIABLES FOR ONE DAY

            save('params','roi_dop_success_perimov','roi_dop_success_peristop','-append')
           
        end
        
        
    end
    
end

% SAVING WHO ACCROSS DAY WORKSPACE
k = strfind(pr_dir0{1},'\');
allsavedir = pr_dir0{1}(1:k(end));
filenamestrip = [num2str(mouse_id) addon];
allsavefilename = [filenamestrip '_workspace'];
filetype = '.mat';
startCount = 0;
numDigits = '2';
if exist([allsavedir allsavefilename filetype],'file') == 2 %file exists already, check for alternative
    checker = true; %check for alternate file names
    Cnt = startCount; %counter for file name
    
    while checker
        testPath = [allsavedir allsavefilename '_' num2str(Cnt, ['%0' numDigits 'i']) filetype];
        
        if exist(testPath,'file') == 2
            Cnt = Cnt + 1; %increase counter until a non-existing file name is found
        else
            checker = false;
        end
        
        if Cnt == 10^numDigits-1 && checker
            numDigits = numDigits+1;
            warning(['No unused file found at given number of digits. Number of digits increased to ' num2str(numDigits) '.']);
        end
    end
    outFile = testPath;
    
else
    outFile = [allsavedir allsavefilename filetype];
end
close all
save(outFile)


