clear all
close all
mouse_id=158;
addon = '_HRZ';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
pr_dir0 = uipickfiles;
dop_allsuc_stop_no_reward = NaN(length(pr_dir0),4,79);
%%
for alldays = 1:length(pr_dir0)%[3:12 14:19]%[3:12 13:19]%[3 5:1]%[5:12 14]%[5:12 14]%1:5%26:30%1:33%31:32%30%27:30%21%[1:21]%[1:2 4:5 12:22]%[8:21]%[2:4 6:11]%[1:2 4:5 12:20]%%[1:21]%%
    clearvars -except alldays addon mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr ...
        mov_corr_success stop_corr_success mov_corr_prob stop_corr_prob mov_corr_fail stop_corr_fail...
        c  alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr dop_suc_movint dop_suc_stopint roe_suc_movint roe_suc_stopint ...
        dop_allsuc_mov dop_allsuc_stop roe_allsuc_mov roe_allsuc_stop ...
        dop_allfail_mov dop_allfail_stop roe_allfail_mov roe_allfail_stop...
        dop_alldays_planes_success_mov dop_alldays_planes_fail_mov dop_alldays_planes_success_stop dop_alldays_planes_fail_stop...
        roe_alldays_planes_success_mov roe_alldays_planes_fail_mov roe_alldays_planes_success_stop roe_alldays_planes_fail_stop...
        subp days_check cnt dop_alldays_planes_perireward roe_alldays_planes_perireward dop_allsuc_perireward roe_allsuc_perireward...
        pr_dir0 dop_alldays_planes_success_stop_no_reward dop_alldays_planes_success_stop_reward...
        dop_allsuc_stop_no_reward dop_allsuc_stop_reward roe_allsuc_stop_no_reward roe_allsuc_stop_reward...
        dop_alldays_planes_double_0 roe_alldays_planes_double_0 dop_alldays_planes_perireward_double roe_alldays_planes_perireward_double...
        dop_allsuc_perireward_double roe_allsuc_perireward_double dop_allsuc_perireward_double_se roe_allsuc_perireward_double_se...
        day_labels roi_dop_alldays_planes_success_mov roi_dop_alldays_planes_success_stop roi_dop_alldays_planes_success_stop_no_reward...
        roi_dop_alldays_planes_success_stop_reward roi_dop_allsuc_mov roi_dop_allsuc_stop roi_dop_allsuc_stop_no_reward roi_dop_allsuc_stop_reward...
        roi_dop_alldays_planes_perireward_0 roi_dop_alldays_planes_perireward_double_0 roi_dop_alldays_planes_perireward roi_dop_alldays_planes_perireward_double...
        roi_dop_allsuc_perireward roi_dop_allsuc_perireward_double roi_dop_allsuc_perireward_se roi_dop_allsuc_perireward_double_se...
        roe_alldays_planes_perireward_0 roe_alldays_planes_success_stop_no_reward roe_alldays_planes_stop_reward roe_alldays_planes_perireward_double_0...
        dop_alldays_planes_perireward_double_0 dop_alldays_planes_perireward_0 roe_alldays_planes_success_stop_reward...
        roi_dop_alldays_planes_periCS roi_dop_alldays_planes_peridoubleCS roi_roe_alldays_planes_periCS roi_roe_alldays_planes_peridoubleCS roi_dop_allsuc_perirewardCS...
        roi_dop_allsuc_perireward_doubleCS roi_roe_allsuc_perirewardCS roi_roe_allsuc_perireward_doubleCS...
        roi_dop_alldays_planes_first3 roi_dop_alldays_planes_last3 roe_dop_alldays_planes_first3 roe_dop_alldays_planes_last3 roi_dop_alldays_planes_probetrace...
        roe_dop_alldays_planes_probetrace roi_dop_allsuc_first3 roi_dop_allsuc_last3 roe_dop_allsuc_first3 roe_dop_allsuc_last3 roi_dop_allsuc_probetrace roe_dop_allsuc_probetrace
    
    close all
    

    planeroicount = 0;
    cnt=cnt+1;
    Day=alldays;
    for allplanes=1:4
        plane=allplanes;

        pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
        pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','')
        

        
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
            if allplanes == 1
                forwardvel1= forwardvel;
                rewards1 = rewards;
            else
                forwardvel = forwardvel1;
                rewards = rewards1;
            end
            %%%%%%%%%%%%%%%
            numplanes=4;
            gauss_win=5;
            frame_rate=31.25/numplanes;
            lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
            rew_thresh=0.001;
            sol2_thresh=1.5;
            num_rew_win_sec=5;%window in seconds for looking for multiple rewards
            rew_lick_win=20;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
            pre_win=5;%pre window in s for rewarded lick average
            post_win=5;%post window in s for rewarded lick average
            exclusion_win=20;%exclusion window pre and post rew lick to look for non-rewarded licks
            speed_thresh = 5; %cm/s cut off for stopped
            Stopped_frame = frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 4; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time);
            pre_win_frames=round(pre_win/frame_time);
            exclusion_win_frames=round(exclusion_win/frame_time);
            

            mean_base_mean=mean(base_mean);
          
            norm_base_mean=base_mean;
            
            if exist('forwardvel')
            speed_binned=forwardvel;
            end
            reward_binned=rewards;
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
            dop_smth=smoothdata(norm_base_mean,'gaussian',gauss_win);
            
            
            
            
            figure;
            plot(rescale(reward_binned,min(dop_smth),max(dop_smth)),'LineWidth',1.5)
            hold on
            plot(rescale(speed_smth_1,min(dop_smth),max(dop_smth)),'LineWidth',1.5)
            plot(dop_smth,'LineWidth',1.5)
            legend({'Reward','Speed','Dopamine'})
             ylabel('dF/F')
             savefig('Fl_Speed_Reward.fig')
           
            %%%%%%%%%%%%%%%%%%%calculate single traces
            %%%%%%rewarded licks
            
                      
            %double rewards
            
            R = bwlabel(reward_binned>rew_thresh);%label rewards, ascending
            rew_idx=find(R);%get indexes of all rewards
            rew_idx_diff=diff(rew_idx);%difference in reward index from last
            short= (rewards == 2);%logical for rewards that happen less than x frames from last reward. 0 = single rew.
            short(rew_idx(find(rew_idx_diff<num_rew_win_frames))) = 1;
            if ~exist('licks')
                licks = lick_binned;
            end
            if sum(licks==0)+sum(licks == 1) ~= length(licks)
                licks = licks<lickThresh;
            end
            supraLick = licks;
            
%             double_rew=find(short);%double events have ysize=0
%             double_lick_idx = [];
%             double_lick_gap = [];
%             for i=1:length(double_rew)
%                 double_idx(i)=double_rew(i);
%                 if double_idx(i)+rew_lick_win_frames < length(supraLick)%if window to search for lick after rew is past length of supraLick, doesn't update single_lick_idx, but single_idx is
%                     if sum(supraLick(double_idx(i):double_idx(i)+rew_lick_win_frames))>0
%                         double_lick_idx(i) = (find(supraLick(double_idx(i):double_idx(i)+rew_lick_win_frames),1,'first'))+double_idx(i)-1;
%                         double_lick_gap(i) = double_lick_idx(i)-double_idx(i);
%                     else
% 
%                     warning('no lick after double reward was delivered!!!');
%                     double_lick_idx(i) = NaN;
%                     double_lick_gap(i) = NaN;
%                     end
%                 end
%             end
            
%             didntlickdouble = find(isnan(double_lick_idx));
%             double_lick_idx(didntlickdouble) = [];
%             %save variables in params
%             save('params','didntlickdouble','double_lick_idx','double_lick_gap','-append')
            
             R = bwlabel(reward_binned>rew_thresh);%label rewards, ascending
            rew_idx=find(R);%get indexes of all rewards
            rew_idx_diff=diff(rew_idx);%difference in reward index from last
            short= (rewards == 1);%logical for rewards that happen less than x frames from last reward. 0 = single rew.
%             short(rew_idx(find(rew_idx_diff<num_rew_win_frames))) = 0;
%             short(rew_idx(find(rew_idx_diff<num_rew_win_frames)+1)) = 0;
            

            
            %single rewards

            
            single_rew=find(short);
            single_idx=[];single_lick_idx=[]; single_lick_gap = [];
            for i=1:length(single_rew)
                %single_idx(i)=rew_idx(i); %orig but doesn't eliminate doubles
                single_idx=single_rew(i);
                if single_idx+rew_lick_win_frames < length(supraLick)%if window to search for lick after rew is past length of supraLick, doesn't update single_lick_idx, but single_idx is
                    %                     single_lick_idx(i) = (find(supraLick(single_idx(i):single_idx(i)+rew_lick_win_frames),1,'first'))+single_idx(i)-1;
                    %                     %looks for first lick after rew with window =exclusion_win_frames
                    %                     %however first lick can be much further in naive animals
                    %                 end
                    
                    lick_exist=(find(supraLick(single_idx:single_idx+rew_lick_win_frames),1,'first'))+single_idx-1;
                    if isempty(lick_exist)~=1
                        
                        single_lick_idx(i) = (find(supraLick(single_idx:single_idx+rew_lick_win_frames),1,'first'))+single_idx-1;
                        %looks for first lick after rew with window =exclusion_win_frames
                        %however first lick can be much further in naive animals
                        single_lick_gap(i) = single_lick_idx(i)-single_idx;
                    else
                        warning('no lick after reward was delivered!!!');
                        single_lick_idx(i)= NaN;
                        single_lick_gap(i) = NaN;
                    end
                end
            end
            didntlicksingle = find(isnan(single_lick_idx));
            single_lick_idx(didntlicksingle) = [];
            if ~isempty(single_lick_idx)
                if single_lick_idx(1) - pre_win_frames <0%remove events too early
                    single_lick_idx(1)=[];
                end
                if single_lick_idx(end) + post_win_frames > length(base_mean)%remove events too late
                    single_lick_idx(end)=[];
                end
            end
            
                        %save variables in params
            save('params','didntlicksingle','single_lick_idx','single_lick_gap','-append')
            
            
            %%%for
            %%%% if no lick after reward strange though!!
            
            [r c]=find(single_lick_idx==0);
            single_lick_idx(c)=[];
            single_traces=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
            
            
            single_traces_roesmth=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
            coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
            for i=1:length(single_lick_idx)
                
                single_traces(:,i)=base_mean(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';%lick at pre_win_frames+1
                single_traces_roesmth(:,i)=speed_smth_1(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';
                [rho, pval]=corrcoef(single_traces(:,i),single_traces_roesmth(:,i));
                coeff_rewarded_licks(i,:)=[rho(1,2) pval(1,2)];
                
                %%% compute cross-corr and find the lag
                s2=single_traces(:,i);
                s1=single_traces_roesmth(:,i);
                [C21,lag21] = xcorr(s2,s1);
                C21 = C21/max(C21);
                [M21,I21] = max(C21);
                t21 = lag21(I21);
                lags_single_traces(1,i)=t21;
            end
            norm_single_traces=single_traces./mean(single_traces(1:pre_win_frames,:));
            norm_single_traces_roesmth=single_traces_roesmth./mean(single_traces_roesmth(1:pre_win_frames,:));
             %save variables in params
             save('params','single_traces','single_traces_roesmth','norm_single_traces','norm_single_traces_roesmth','-append')
             
             %plot for single reward
             
              figure;
             hold on;
             title('Single rewards');
             xlabel('seconds from reward lick')
             ylabel('dF/F')
             plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces);%auto color
             plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(norm_single_traces,2),'k','LineWidth',2);
             legend(['n = ',num2str(size(norm_single_traces,2))])%n=
             
             
             currfile=strcat('PeriReward_Fl_single_rew.fig');
             savefig(currfile)
             
             
             
            
             %for doubles
%              double_traces=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
            
            
%             double_traces_roesmth=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
%             coeff_doublerewarded_licks=[]; coeff_norm_doublerewarded_licks=[];  lags_double_traces=[];
%             if ~isempty(double_lick_idx)
%             for i=1:length(double_lick_idx)
%                 
%                 
%                 double_traces(:,i)=base_mean(double_lick_idx(i)-pre_win_frames:double_lick_idx(i)+post_win_frames)';%lick at pre_win_frames+1
%                 double_traces_roesmth(:,i)=speed_smth_1(double_lick_idx(i)-pre_win_frames:double_lick_idx(i)+post_win_frames)';
%                 [rho, pval]=corrcoef(double_traces(:,i),double_traces_roesmth(:,i));
%                 coeff_doublerewarded_licks(i,:)=[rho(1,2) pval(1,2)];
                
                %%% compute cross-corr and find the lag
%                 s2=double_traces(:,i);
%                 s1=double_traces_roesmth(:,i);
%                 [C21,lag21] = xcorr(s2,s1);
%                 C21 = C21/max(C21);
%                 [M21,I21] = max(C21);
%                 t21 = lag21(I21);
%                 lags_double_traces(1,i)=t21;
%             end
%             else
%                 double_traces = NaN(length(-pre_win_frames:post_win_frames));
%                 double_traces_roesmth = NaN(length(-pre_win_frames:post_win_frames));
%             end
%             norm_double_traces=double_traces./mean(double_traces(1:pre_win_frames,:));
%             norm_double_traces_roesmth=double_traces_roesmth./mean(double_traces_roesmth(1:pre_win_frames,:));
%              %save variables in params
%              save('params','double_traces','double_traces_roesmth','norm_double_traces','norm_double_traces_roesmth','-append')
             
             %plot doubles and save figure
             
%              figure;
%              hold on;
%              title('Double rewards');
%              xlabel('seconds from reward lick')
%              ylabel('dF/F')
%              plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_double_traces);%auto color
%              plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(norm_double_traces,2),'k','LineWidth',2);
%              legend(['n = ',num2str(size(norm_double_traces,2))])%n=
%              %legend()
%              
%              
%              currfile=strcat('PeriReward_Fl_double_rew.fig');
%              savefig(currfile)

            
            

            
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
           if exist('single_traces','var')
%                 find_figure(strcat(mouse_id,'_perireward','_plane',num2str(allplanes)))
%                 subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),
%                 xlabel('seconds from first reward lick')
%                 ylabel('dF/F')
                striptitleindx = strfind(pr_dir0{Day},'\');
%                 title(pr_dir0{Day}(striptitleindx(end)+1:end))
                day_labels{Day} = pr_dir0{Day}(striptitleindx(end)+1:end);
%                 hold on
%                           subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces);
%                 hold on
%                  subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(mean(norm_single_traces,2),'k','LineWidth',2);
%                 legend(['n = ',num2str(size(norm_single_traces,2))])
%                 
%                 plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(norm_single_traces_roesmth,0.99,1))
%  
                
                %%%%moving and stop activity
                
                
%                 find_figure('dop_forwardvel_control_binned_activity');clf; 
%                 subplot(4,1,allplanes),plot(norm_base_mean)
%                 hold on
                forwardvel=smoothdata(speed_binned,'gaussian',gauss_win)';
%                 plot(rescale(forwardvel,0.999,1.05),'k')
                
        
                
                
                if size(forwardvel,1) == 1
                    moving_middle=get_moving_time_V2(forwardvel',speed_thresh,Stopped_frame);
                else
                    moving_middle=get_moving_time_V2(forwardvel,speed_thresh,Stopped_frame);
                end
                mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                
                

                
                idx_rm=(mov_success_tmpts- pre_win_frames)<=0;
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                

                idx_rm=(mov_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                allmov_success=NaN(1,length(norm_base_mean));
                allmov_success(moving_middle) = 1;
                dop_success_perimov=[]; roe_success_perimov=[];
                for stamps=1:length(mov_success_tmpts)
                    dop_success_perimov(stamps,:)= norm_base_mean(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
                    roe_success_perimov(stamps,:)= forwardvel(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);

                end
                
                
                %%%%stopping success trials
                
                stop_success_tmpts =  moving_middle(find(diff(moving_middle)>1))+1;
                
                                
                idx_rm= (stop_success_tmpts - pre_win_frames) <0;
                rm_idx=find(idx_rm==1);
                stop_success_tmpts(rm_idx)=[];
                
                
                
                idx_rm=(stop_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm==1)
                stop_success_tmpts(rm_idx)=[];
                
                rew_idx = find(rewards);
                rew_stop_success_tmpts = [];
                for r = 1:length(rew_idx)
                    if ~isempty(find(stop_success_tmpts-rew_idx(r)>=0-frame_tol & stop_success_tmpts-rew_idx(r) <max_reward_stop,1))
                    rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts-rew_idx(r)>=0-frame_tol & stop_success_tmpts-rew_idx(r) <max_reward_stop,1));
                    else
                        rew_stop_success_tmpts(r) = NaN;
                    end
                end
                didntstoprew = find(isnan(rew_stop_success_tmpts));
                rew_stop_success_tmpts(isnan(rew_stop_success_tmpts)) = [];
                nonrew_stop_success_tmpts = setxor(rew_stop_success_tmpts,stop_success_tmpts);
                
                save('params','didntstoprew','rew_stop_success_tmpts','nonrew_stop_success_tmpts','stop_success_tmpts','mov_success_tmpts','-append')
                
                
                
                dop_success_peristop=[]; roe_success_peristop=[];
                allstop_success=NaN(1,length(norm_base_mean));
                allstop_success(setxor(1:length(norm_base_mean),moving_middle)) = 1;
                for stamps=1:length(stop_success_tmpts)
                    dop_success_peristop(stamps,:)= norm_base_mean(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
                    roe_success_peristop(stamps,:)= forwardvel(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
                end
                
                %  non rewarded stops
                 dop_success_peristop_no_reward=[]; roe_success_peristop_no_reward=[];
                for stamps=1:length(nonrew_stop_success_tmpts)
                    dop_success_peristop_no_reward(stamps,:)= norm_base_mean(nonrew_stop_success_tmpts(stamps)-pre_win_frames:nonrew_stop_success_tmpts(stamps)+post_win_frames);
                    roe_success_peristop_no_reward(stamps,:)= forwardvel(nonrew_stop_success_tmpts(stamps)-pre_win_frames:nonrew_stop_success_tmpts(stamps)+post_win_frames);
                end
                
                % rewarded stops
                 dop_success_peristop_reward=[]; roe_success_peristop_reward=[];

                for stamps=1:length(rew_stop_success_tmpts)
                    dop_success_peristop_reward(stamps,:)= norm_base_mean(rew_stop_success_tmpts(stamps)-pre_win_frames:rew_stop_success_tmpts(stamps)+post_win_frames);
                    roe_success_peristop_reward(stamps,:)= forwardvel(rew_stop_success_tmpts(stamps)-pre_win_frames:rew_stop_success_tmpts(stamps)+post_win_frames);

                end
                
                
                
                %save all peri locomotion variables
                save('params','roe_success_peristop_no_reward','dop_success_peristop_no_reward','roe_success_peristop_reward','dop_success_peristop_reward','dop_success_perimov','dop_success_peristop','roe_success_perimov','roe_success_peristop','-append')
                
                
                alldays=alldays;
                dop_alldays_planes_success_mov{alldays,allplanes}=dop_success_perimov;
                dop_alldays_planes_success_stop{alldays,allplanes}=dop_success_peristop;
                
                dop_alldays_planes_success_stop_no_reward{alldays,allplanes}= dop_success_peristop_no_reward;
                dop_alldays_planes_success_stop_reward{alldays,allplanes}= dop_success_peristop_reward;
                
                roe_alldays_planes_success_mov{alldays,allplanes}=roe_success_perimov;
                roe_alldays_planes_success_stop{alldays,allplanes}=roe_success_peristop;
                
                roe_alldays_planes_success_stop_no_reward{alldays,allplanes}= roe_success_peristop_no_reward;
                roe_alldays_planes_success_stop_reward{alldays,allplanes}= roe_success_peristop_reward;
                
                                
                dop_allsuc_mov(alldays,allplanes,:)=mean(dop_success_perimov);
                dop_allsuc_stop(alldays,allplanes,:)=mean(dop_success_peristop);
                
                dop_allsuc_stop_no_reward(alldays,allplanes,:)=mean(dop_success_peristop_no_reward);
                dop_allsuc_stop_reward(alldays,allplanes,:)=mean(dop_success_peristop_reward);
                
                if ~isempty(roe_success_perimov)
                roe_allsuc_mov(alldays,allplanes,:)=mean(roe_success_perimov,1);
                else
                    roe_allsuc_mov(alldays,allplanes,:)= NaN(1,79);
                end
                if ~isempty(roe_success_perimov)
                roe_allsuc_stop(alldays,allplanes,:)=mean(roe_success_peristop,1);
                  else
                    roe_allsuc_stop(alldays,allplanes,:)= NaN(1,79);
                end
                
                if ~isempty(roe_success_peristop_no_reward)
                roe_allsuc_stop_no_reward(alldays,allplanes,:)=mean(roe_success_peristop_no_reward,1);
                else
                    roe_allsuc_stop_no_reward(alldays,allplanes,:)=NaN(1,1,79);
                end
                if ~isempty(roe_success_peristop_reward)
                roe_allsuc_stop_reward(alldays,allplanes,:)=mean(roe_success_peristop_reward,1);
                else
                    roe_allsuc_stop_reward(alldays,allplanes,:) = NaN(1,1,79);
                end
                
                %%%%% perireward
                dop_alldays_planes_perireward_0{alldays,allplanes}=single_traces;
                roe_alldays_planes_perireward_0{alldays,allplanes}=single_traces_roesmth;
                
%                 dop_alldays_planes_perireward_double_0{alldays,allplanes} = double_traces;
%                 roe_alldays_planes_perireward_double_0{alldays,allplanes} = double_traces_roesmth;
%                 
                dop_alldays_planes_perireward{alldays,allplanes}=norm_single_traces;
                roe_alldays_planes_perireward{alldays,allplanes}=norm_single_traces_roesmth;
                
%                 dop_alldays_planes_perireward_double{alldays,allplanes} = norm_double_traces;
%                 roe_alldays_planes_perireward_double{alldays,allplanes} = norm_double_traces_roesmth;
                
                dop_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces,2);
                roe_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces_roesmth,2);
                
%                 dop_allsuc_perireward_double(alldays,allplanes,:) = mean(norm_double_traces,2);
%                 roe_allsuc_perireward_double(alldays,allplanes,:) = mean(norm_double_traces_roesmth,2);
%                 
                dop_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces,[],2)./sqrt(size(norm_single_traces,2));
                roe_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces_roesmth,[],2)./sqrt(size(norm_single_traces_roesmth,2));
%                 
%                 dop_allsuc_perireward_double_se(alldays,allplanes,:)=std(norm_double_traces,[],2)./sqrt(size(norm_double_traces,2));
%                 roe_allsuc_perireward_double_se(alldays,allplanes,:)=std(norm_double_traces_roesmth,[],2)./sqrt(size(norm_double_traces_roesmth,2));
%                 
                
                %%%%% moving
%                 find_figure(strcat(mouse_id,'_perilocomotion moving','_plane',num2str(allplanes)));
%                   subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),
%                 ylabel('dF/F')
%                 
%                 
%                 
%                 hold on
%                             subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov);
%                 hold on
%                  subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(dop_success_perimov,1),'k','LineWidth',2);
%                 
%                 legend(['n = ',num2str(size(dop_success_perimov,1))])
%                 hold on
%                 plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(mean(roe_success_perimov,1),0.985,0.99),'k--','LineWidth',2)
%                 
%                 text(-5, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
%                 text(5, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
                
                
                
                
                
            
  
            end
            
                                        %%%%%%%%%%%%%%%%%%%%%%%%%%5 for regions of interest
            
            for roii = 1:length(params.roibasemean2)
                roibase_mean = params.roibasemean2{roii};
                roimean_base_mean=mean(params.roibasemean2{roii});

            
            roinorm_base_mean=roibase_mean/roimean_base_mean;
            
            roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                figure;
            plot(rescale(reward_binned,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
            hold on
            plot(rescale(speed_smth_1,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
            plot(roidop_smth,'LineWidth',1.5)
            legend({'Reward','Speed','Dopamine'})
             ylabel('dF/F')
             savefig(['ROI_' num2str(roii) 'Fl_Speed_Reward.fig'])



            roisingle_traces=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
            
            
            single_traces_roesmth=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
            coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
            for i=1:length(single_lick_idx)
                
                roisingle_traces(:,i)=roibase_mean(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';%lick at pre_win_frames+1
                single_traces_roesmth(:,i)=speed_smth_1(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';
     
            end
            roinorm_single_traces=roisingle_traces./mean(roisingle_traces(1:pre_win_frames,:));
              norm_single_traces_roesmth=single_traces_roesmth./mean(single_traces_roesmth(1:pre_win_frames,:));

             %plot for single reward
             
              figure;
             hold on;
             title(['ROI ' num2str(roii) ' Single rewards']);
             xlabel('seconds from reward lick')
             ylabel('dF/F')
             plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces);%auto color
             plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(norm_single_traces,2),'k','LineWidth',2);
             legend(['n = ',num2str(size(roinorm_single_traces,2))])%n=
             
             currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_rew.fig']);
             savefig(currfile)
             
             
             %split into first 3 success vs last 3 success of epoch 1
             rewards(find([0 diff(rewards) == 0])) = 0;
             rew_idx = find(rewards);
             rewards(rew_idx(find(rewards)+rew_lick_win_frames>length(rewards))) = 0;
             if length(find(changeRewLoc))>1
             numrewardsinepoch1 = length(find(rewards(1:find(changeRewLoc(2:end),1))));
             else
                 numrewardsinepoch1 = length(find(rewards));
             end
             if numrewardsinepoch1 > 10
             roifirst3rewardlick = roinorm_single_traces(:,1:3);
             roilast3rewardlick = roinorm_single_traces(:,numrewardsinepoch1-2:numrewardsinepoch1);
             
             roefirst3rewardlick =single_traces_roesmth(:,1:3);
             roelast3rewardlick = single_traces_roesmth(:,numrewardsinepoch1-2:numrewardsinepoch1);
             else
                 roifirst3rewardlick = NaN(79,1);
                 roilast3rewardlick = NaN(79,1);
                 roefirst3rewardlick = NaN(79,1);
                 roelast3rewardlick = NaN(79,1);
                 
             end
             
             %add a periprobed lick for epoch 1
%              epoch1rewardzone = changeRewLoc(1);
%              probeidx = find(trialnum(find(trialnum == 3,1):end) == 0,1):find(trialnum(find(trialnum == 3,1):end) == 1,1)-1;
%              if ~isempty(probeidx)
%              probelickindx = find(licks(find(ybinned(probeidx)>epoch1rewardzone-7.5 & ybinned(probeidx)<epoch1rewardzone+7.5)+probeidx(1)),1)+probeidx(1);
%              else
%                  probelickindx = [];
%              end
%              if ~isempty(probelickindx)
%                  roiprobetrace = roibase_mean(probelickindx-pre_win_frames:probelickindx+post_win_frames)';
%                  roeprobetrace = speed_smth_1(probelickindx-pre_win_frames:probelickindx+post_win_frames)';
%              else
%                  roiprobetrace = NaN(1,79);
%                  roeprobetrace = NaN(1,79);
%              end
             
             
             
             % for single rew CS
             singlerew = single_rew(find(single_rew>pre_win_frames&single_rew<length(licks)-post_win_frames));
             roisingle_tracesCS=zeros(pre_win_frames+post_win_frames+1,length(singlerew));
            
            
            roisingle_traces_roesmthCS=zeros(pre_win_frames+post_win_frames+1,length(singlerew));
            for i=1:length(singlerew)
                
                roisingle_tracesCS(:,i)=roibase_mean(singlerew(i)-pre_win_frames:singlerew(i)+post_win_frames)';%lick at pre_win_frames+1
                roisingle_traces_roesmthCS(:,i)=speed_smth_1(singlerew(i)-pre_win_frames:singlerew(i)+post_win_frames)';
     
            end
            roinorm_single_tracesCS=roisingle_tracesCS./mean(roisingle_tracesCS(1:pre_win_frames,:));
              roinorm_single_traces_roesmthCS=roisingle_traces_roesmthCS./mean(roisingle_traces_roesmthCS(1:pre_win_frames,:));
             
              save('params','roinorm_single_tracesCS','roinorm_single_traces_roesmthCS','-append')
            
%              %for doubles
%              roidouble_traces=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
%             
%             
%             double_traces_roesmth=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
%             if ~isempty(double_lick_idx)
%             for i=1:length(double_lick_idx)
%                 
%                 
%                 roidouble_traces(:,i)=roibase_mean(double_lick_idx(i)-pre_win_frames:double_lick_idx(i)+post_win_frames)';%lick at pre_win_frames+1
%                 double_traces_roesmth(:,i)=speed_smth_1(double_lick_idx(i)-pre_win_frames:double_lick_idx(i)+post_win_frames)';
% 
%             end
%             else
%                 roidouble_traces = NaN(length(-pre_win_frames:post_win_frames));
%                 roidouble_traces_roesmth = NaN(length(-pre_win_frames:post_win_frames));
%             end
%             roinorm_double_traces=roidouble_traces./mean(roidouble_traces(1:pre_win_frames,:));
%             norm_double_traces_roesmth=double_traces_roesmth./mean(double_traces_roesmth(1:pre_win_frames,:));
 

%              %plot doubles and save figure
%              
%              figure;
%              hold on;
%              title(['ROI ' num2str(roii) ' Double rewards']);
%              xlabel('seconds from reward lick')
%              ylabel('dF/F')
%               plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_double_traces);%auto color
%              plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(norm_double_traces,2),'k','LineWidth',2);
%              legend(['n = ',num2str(size(norm_double_traces,2))])%n=
%                
%              
%              currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_double_rew.fig']);
%              savefig(currfile)
%              % for double rew CS
%              doublerew = double_rew(find(double_rew>pre_win_frames&double_rew<length(licks)-post_win_frames));
%              roidouble_tracesCS=zeros(pre_win_frames+post_win_frames+1,length(doublerew));


%              roidouble_traces_roesmthCS=zeros(pre_win_frames+post_win_frames+1,length(doublerew));
%              for i=1:length(doublerew)
% 
%                  roidouble_tracesCS(:,i)=roibase_mean(doublerew(i)-pre_win_frames:doublerew(i)+post_win_frames)';%lick at pre_win_frames+1
%                  roidouble_traces_roesmthCS(:,i)=speed_smth_1(doublerew(i)-pre_win_frames:doublerew(i)+post_win_frames)';
% 
%              end
%              roinorm_double_tracesCS=roidouble_tracesCS./mean(roidouble_tracesCS(1:pre_win_frames,:));
%              roinorm_double_traces_roesmthCS=roidouble_traces_roesmthCS./mean(roidouble_traces_roesmthCS(1:pre_win_frames,:));
%             save('params','roinorm_double_tracesCS','roinorm_double_traces_roesmthCS','-append')
%              
             if exist('roisingle_traces','var')
             
                 if size(forwardvel,1) == 1
                     moving_middle=get_moving_time_V2(forwardvel',speed_thresh,Stopped_frame);
                 else
                     moving_middle=get_moving_time_V2(forwardvel,speed_thresh,Stopped_frame);
                 end
                mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                idx_rm=(mov_success_tmpts- pre_win_frames)<=0;
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];

                idx_rm=(mov_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                
                roiallmov_success=NaN(1,length(roinorm_base_mean));
                roiallmov_success(moving_middle) = 1;
                roidop_success_perimov=[]; roe_success_perimov=[];
                for stamps=1:length(mov_success_tmpts)
                    roidop_success_perimov(stamps,:)= roinorm_base_mean(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
                end
                
                
                %%%%stopping success trials
                %
                stop_success_tmpts =  moving_middle(find(diff(moving_middle)>1))+1;
                

                idx_rm= (stop_success_tmpts - pre_win_frames) <0;
                rm_idx=find(idx_rm==1);
                stop_success_tmpts(rm_idx)=[];
                
                idx_rm=(stop_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm==1)
                stop_success_tmpts(rm_idx)=[];
                
                rew_idx = find(rewards);
                rew_stop_success_tmpts = [];
                for r = 1:length(rew_idx)
                    if ~isempty(find(stop_success_tmpts-rew_idx(r)>=0-frame_tol & stop_success_tmpts-rew_idx(r) <max_reward_stop,1))
                    rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts-rew_idx(r)>=0-frame_tol & stop_success_tmpts-rew_idx(r) <max_reward_stop,1));
                    else
                        rew_stop_success_tmpts(r) = NaN;
                    end
                end
                didntstoprew = find(isnan(rew_stop_success_tmpts));
                rew_stop_success_tmpts(isnan(rew_stop_success_tmpts)) = [];
                nonrew_stop_success_tmpts = setxor(rew_stop_success_tmpts,stop_success_tmpts);
                
                
                roidop_success_peristop=[]; 
                roiallstop_success=NaN(1,length(roinorm_base_mean));
                roiallstop_success(setxor(1:length(roinorm_base_mean),moving_middle)) = 1;
                for stamps=1:length(stop_success_tmpts)
                    roidop_success_peristop(stamps,:)= roinorm_base_mean(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
                   
                end
                
                %  non rewarded stops
                 roidop_success_peristop_no_reward=[]; 
                for stamps=1:length(nonrew_stop_success_tmpts)
                    roidop_success_peristop_no_reward(stamps,:)= roinorm_base_mean(nonrew_stop_success_tmpts(stamps)-pre_win_frames:nonrew_stop_success_tmpts(stamps)+post_win_frames);
                    
                end
                
                % rewarded stops
                 roidop_success_peristop_reward=[];

                for stamps=1:length(rew_stop_success_tmpts)
                    roidop_success_peristop_reward(stamps,:)= roinorm_base_mean(rew_stop_success_tmpts(stamps)-pre_win_frames:rew_stop_success_tmpts(stamps)+post_win_frames);
                end
                
                %save per day roi reward
                roi_single_traces{roii} = roisingle_traces;
                roi_first3_traces{roii} = roifirst3rewardlick;
                roi_last3_traces{roii} = roilast3rewardlick;
%                 roi_probe_trace{roii} = roiprobetrace/mean(roiprobetrace(1:pre_win_frames));
                
                roe_first3_traces{roii} = roefirst3rewardlick;
                roe_last3_traces{roii} = roelast3rewardlick;
%                 roe_probe_traces{roii} = roeprobetrace;
                
                
                
                roi_norm_single_traces{roii} = roinorm_single_traces;
               
%                 roi_double_traces{roii} = roidouble_traces;
%                 roi_norm_double_traces{roii} = roinorm_double_traces;
                roi_dop_success_peristop_no_reward{roii} = roidop_success_peristop_no_reward;
                roi_dop_success_peristop_reward{roii} = roidop_success_peristop_reward;
                roi_dop_success_perimov{roii} = roidop_success_perimov;
                roi_dop_success_peristop{roii} = roidop_success_peristop;
                
                %save all peri locomotion variables
                 planeroicount = planeroicount + 1;
                alldays=alldays;
                
                
%                                 roi_first3_traces{roii} = roifirst3rewardlick;
%                 roi_last3_traces{roii} = roilast3rewardlick;
%                 roi_probe_trace{roii} = roiprobetrace/mean(roiprobetrace(1:pre_win_frames));
%                 
%                 roe_first3_traces{roii} = roefirst3rewardlick;
%                 roe_last3_traces{roii} = roelast3rewardlick;
%                 roe_probe_traces{roii} = roeprobetrace;
                
                roi_dop_alldays_planes_first3{alldays,planeroicount}= roifirst3rewardlick;
                roi_dop_alldays_planes_last3{alldays,planeroicount}= roilast3rewardlick;
                roe_dop_alldays_planes_first3{alldays,planeroicount}= roefirst3rewardlick;
                roe_dop_alldays_planes_last3{alldays,planeroicount}= roelast3rewardlick;
                
%                 roi_dop_alldays_planes_probetrace{alldays,planeroicount}= roiprobetrace/mean(roiprobetrace(1:pre_win_frames));
%                 roe_dop_alldays_planes_probetrace{alldays,planeroicount}= roeprobetrace;
                
                roi_dop_allsuc_first3(alldays,planeroicount,:) = mean(roifirst3rewardlick,2);
                roi_dop_allsuc_last3(alldays,planeroicount,:) = mean(roilast3rewardlick,2);
                roe_dop_allsuc_first3(alldays,planeroicount,:) = mean(roefirst3rewardlick,2);
                roe_dop_allsuc_last3(alldays,planeroicount,:) = mean(roelast3rewardlick,2);
%                 roi_dop_allsuc_probetrace(alldays,planeroicount,:) = roiprobetrace/mean(roiprobetrace(1:pre_win_frames));
%                 roe_dop_allsuc_probetrace(alldays,planeroicount,:) = roeprobetrace;
                
                
                if ~isempty(roidop_success_perimov)
                roi_dop_alldays_planes_success_mov{alldays,planeroicount}=roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2);
                else
                     roi_dop_alldays_planes_success_mov{alldays,planeroicount}= NaN(1,79);
                end
                if ~isempty(roidop_success_peristop)
                roi_dop_alldays_planes_success_stop{alldays,planeroicount}=roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2);
                else
                    roi_dop_alldays_planes_success_stop{alldays,planeroicount}=NaN(1,79);
                end
                if ~isempty(roidop_success_peristop_no_reward) 
                roi_dop_alldays_planes_success_stop_no_reward{alldays,planeroicount}= roidop_success_peristop_no_reward./mean(roidop_success_peristop_no_reward(:,1:pre_win_frames),2);
                else
                    roi_dop_alldays_planes_success_stop_no_reward{alldays,planeroicount}= NaN(1,size(roidop_success_perimov,2));
                end
                if ~isempty(roidop_success_peristop_reward)
                roi_dop_alldays_planes_success_stop_reward{alldays,planeroicount}= roidop_success_peristop_reward./mean(roidop_success_peristop_reward(:,1:pre_win_frames),2);
                else
                 roi_dop_alldays_planes_success_stop_reward{alldays,planeroicount} = NaN(1,size(roidop_success_perimov,2));
                end
                     
                if ~isempty(roidop_success_perimov)                
                roi_dop_allsuc_mov(alldays,planeroicount,:)=mean(roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2),1);
                else
                     roi_dop_allsuc_mov(alldays,planeroicount,:)= NaN(1,79);
                end
                if ~isempty(roidop_success_peristop)
                roi_dop_allsuc_stop(alldays,planeroicount,:)=mean(roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2),1);
                else
                    roi_dop_allsuc_stop(alldays,planeroicount,:)= NaN(1,79);
                end
                
                if ~isempty(roidop_success_peristop_no_reward) 
                roi_dop_allsuc_stop_no_reward(alldays,planeroicount,:)=mean(roidop_success_peristop_no_reward./mean(roidop_success_peristop_no_reward(:,1:pre_win_frames),2),1);
                else
                    roi_dop_allsuc_stop_no_reward(alldays,planeroicount,:) = NaN(1,79);
                end
                if ~isempty(roidop_success_peristop_reward)
                roi_dop_allsuc_stop_reward(alldays,planeroicount,:)=mean(roidop_success_peristop_reward./mean(roidop_success_peristop_reward(:,1:pre_win_frames),2),1);
                else
                    roi_dop_allsuc_stop_reward(alldays,planeroicount,:) = NaN(1,79);
                end
                
               
                %%%%% perireward
                roi_dop_alldays_planes_perireward_0{alldays,planeroicount}=roisingle_traces;
                  
%                 roi_dop_alldays_planes_perireward_double_0{alldays,planeroicount} = roidouble_traces;
                  
                roi_dop_alldays_planes_perireward{alldays,planeroicount}=roinorm_single_traces;
                 
%                 roi_dop_alldays_planes_perireward_double{alldays,planeroicount} = roinorm_double_traces;
                
                roi_dop_allsuc_perireward(alldays,planeroicount,:)=mean(roinorm_single_traces,2);
                 
%                 roi_dop_allsuc_perireward_double(alldays,planeroicount,:) = mean(roinorm_double_traces,2);
                
                roi_dop_allsuc_perireward_se(alldays,planeroicount,:)=std(roinorm_single_traces,[],2)./sqrt(size(roinorm_single_traces,2));
                 
%                 roi_dop_allsuc_perireward_double_se(alldays,planeroicount,:)=std(roinorm_double_traces,[],2)./sqrt(size(roinorm_double_traces,2));

                roi_dop_alldays_planes_periCS{alldays,planeroicount} = roinorm_single_tracesCS;

%                 roi_dop_alldays_planes_peridoubleCS{alldays,planeroicount} = roinorm_double_tracesCS;

                roi_roe_alldays_planes_periCS{alldays,planeroicount} = roisingle_traces_roesmthCS;

%                 roi_roe_alldays_planes_peridoubleCS{alldays,planeroicount} = roidouble_traces_roesmthCS;
                
                roi_dop_allsuc_perirewardCS(alldays,planeroicount,:) = mean(roinorm_single_tracesCS,2);

%                 roi_dop_allsuc_perireward_doubleCS(alldays,planeroicount,:) = mean(roinorm_double_tracesCS,2);

                roi_roe_allsuc_perirewardCS(alldays,planeroicount,:) = mean(roisingle_traces_roesmthCS,2);
                
%                 roi_roe_allsuc_perireward_doubleCS(alldays,planeroicount,:) = mean(roidouble_traces_roesmthCS,2);


                
                                
             end
            
            end
                
                save('params','roi_single_traces','roi_norm_single_traces','-append')
%                 save('params','roi_first3_traces','roi_last3_traces','roi_probe_trace','roe_first3_traces','roe_last3_traces','roe_probe_traces','-append')
 save('params','roi_first3_traces','roi_last3_traces','roe_first3_traces','roe_last3_traces','-append')

                save('params','roi_dop_success_peristop_no_reward','roi_dop_success_peristop_reward','roi_dop_success_perimov','roi_dop_success_peristop','-append')
               
                    
                    
            
        end
        
    end

end

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


