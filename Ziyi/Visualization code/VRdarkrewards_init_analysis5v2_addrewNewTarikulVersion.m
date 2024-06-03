clear all
close all
mouse_id=179;
addon = 'checkearly_dark_reward2';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
pr_dir0 = uipickfiles;

%mouseid 1(156) 2(157) 3(167) 4(168) 5(169) 6(171) 7(170) 8(179) 9(181)

mouseid=input('enter mouseid =');

oldbatch=input('if oldbatch press 1 else 0=');
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
            %             if allplanes == 1 %%% CHANGE 1 FOR DEFAULT
            %                 forwardvel1= forwardvel;
            %                 rewards1 = rewards;
            %             else
            %                 forwardvel = forwardvel1;
            %                 rewards = rewards1;
            %             end
            %%%%%%%%%%%%%%%
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
            if mouseid<=2
                CSUStimelag = 0; %seconds between
            else
                CSUStimelag=0.5;
            end

            
        
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
            max_nrew_stop_licktol=2*frame_rate;
            
            
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
            
            
            %double rewards
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            R = bwlabel(reward_binned>rew_thresh);%label rewards, ascending
            rew_idx=find(R);%get indexes of all rewards
            temp = consecutive_stretch(rew_idx);
            rew_idx = cellfun(@(x) x(1), temp,'UniformOutput',1); %If the threshold over counts the same reward
            
            
            rew_idx_diff=diff(rew_idx);%difference in reward index from last

            if oldbatch~=1
                short=rew_idx_diff<num_rew_win_frames;%logical for rewards that happen less than x frames from last reward. 0 = single rew.
            elseif oldbatch==1
                short=rewardsALL(rew_idx) == 2;
            end
            ysize=[];
            %if there are any multi rewards
            if any(short)
                multi_reward_num=bwlabel(short);%label all multiple rewards ascending. doubles have single number, triples two consecutive, etc.
                % double_rew=find(ysize==0);
                for i=1:max(multi_reward_num)  %find number of consecutive rewards < window.
                    ysize(i)=find(multi_reward_num==i,1,'last')-find(multi_reward_num==i,1,'first');
                    %ysize length is number of multirewards,
                    %             ysize(1)=1 corresponds to 1st entry in multi_reward_num and is triple rew (double=0), etc
                    
                end
                
            end
            %             if exist('supraLick','var')
            %                 supraLick=supraLick;
            %                 licks=supraLick;
            %             else
            supraLick=licksALL;
            %             end
            
            %double rewards. must be doubles is any(short), assuming not just tri.
            double_rew=find(ysize==0);%double events have ysize=0
            double_lick_idx = [];
            double_lick_gap = [];
            double_idx=[];
            for i=1:length(double_rew)
                double_idx(i)=rew_idx(find(multi_reward_num==double_rew(i)));
                if double_idx(i)+rew_lick_win_frames < length(supraLick)%if window to search for lick after rew is past length of supraLick, doesn't update single_lick_idx, but single_idx is
                    lick_exist=(find(supraLick(double_idx(i):double_idx(i)+rew_lick_win_frames),1,'first'))+double_idx(i)-1;
                    if isempty(lick_exist)~=1
                        double_lick_idx(i)= (find(supraLick(double_idx(i):double_idx(i)+rew_lick_win_frames),1,'first'))+double_idx(i)-1;%finding closest lick after rew
                        double_lick_gap(i) = double_lick_idx(i)-double_idx(i);
                    else
                        warning('no lick after reward was delivered!!!');
                        double_lick_idx(i) = NaN;
                        double_lick_gap(i) = NaN;
                    end
                end
            end
            didntlickdouble = find(isnan(double_lick_idx));
            double_lick_idx(didntlickdouble) = [];
            
            %       %erase double rew from subsequent anaysis
            %       rew_ind_no_trip_doub=rew_ind_no_trip;
            %
            %       for i=1:
            
            %remove trials too close to beginning or end
            if ~isempty(double_lick_idx)
                if double_lick_idx(1) - pre_win_framesALL <0%remove events too early
                    double_lick_idx(1)=[];
                end
                if double_lick_idx(end) + post_win_framesALL > length(utimedFF)%remove events too late
                    double_lick_idx(end)=[];
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            %save variables in params
            save('params','didntlickdouble','double_lick_idx','double_lick_gap','-append')
            
            R = bwlabel(reward_binned>rew_thresh);%label rewards, ascending
            rew_idx=find(R);%get indexes of all rewards
            rew_idx_diff=diff(rew_idx);%difference in reward index from last
            temp = consecutive_stretch(rew_idx);
            rew_idx = cellfun(@(x) x(1), temp,'UniformOutput',1); %If the threshold over counts the same reward
            
            short= (reward_binned == 1);%logical for rewards that happen less than x frames from last reward. 0 = single rew.
            short(rew_idx(find(rew_idx_diff<num_rew_win_frames))) = 0;
            short(rew_idx(find(rew_idx_diff<num_rew_win_frames)+1)) = 0;
            
            
            
            % single rewards
            
            
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
                if single_lick_idx(1) - pre_win_framesALL <0%remove events too early
                    single_lick_idx(1)=[];
                end
                if single_lick_idx(end) + post_win_framesALL > length(utimedFF)%remove events too late
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
            
            %%%%%%%%%%%%%%%%%%%%%%
            %Peri Reward Lick Triggered Average - Single
            single_traces_roesmth=zeros(pre_win_framesALL+post_win_framesALL+1,length(single_lick_idx));
            coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
            for i=1:length(single_lick_idx)
                currentrewidxperplane = find(timedFF>=utimedFF(single_lick_idx(i)),1);
                single_traces(:,i)=base_mean(currentrewidxperplane-pre_win_frames:currentrewidxperplane+post_win_frames)';%lick at pre_win_frames+1
                single_traces_roesmth(:,i)=speed_smth_1(single_lick_idx(i)-pre_win_framesALL:single_lick_idx(i)+post_win_framesALL)';
                %                 [rho, pval]=corrcoef(single_traces(:,i),single_traces_roesmth(:,i));
                %                 coeff_rewarded_licks(i,:)=[rho(1,2) pval(1,2)];
                
                %%% compute cross-corr and find the lag
                %                 s2=single_traces(:,i);
                %                 s1=single_traces_roesmth(:,i);
                %                 [C21,lag21] = xcorr(s2,s1);
                %                 C21 = C21/max(C21);
                %                 [M21,I21] = max(C21);
                %                 t21 = lag21(I21);
                %                 lags_single_traces(1,i)=t21;
            end
            norm_single_traces=single_traces./mean(single_traces(1:pre_win_frames,:));
            norm_single_traces_roesmth=single_traces_roesmth./mean(single_traces_roesmth(1:pre_win_framesALL,:));
            %save variables in params
            save('params','single_traces','single_traces_roesmth','norm_single_traces','norm_single_traces_roesmth','-append')
            
            %plot for single reward
            
            figure;
            hold on;
            title('Single rewards');
            xlabel('seconds from reward lick')
            ylabel('dF/F')
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,norm_single_traces);%auto color
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,mean(norm_single_traces,2),'k','LineWidth',2);
            legend(['n = ',num2str(size(norm_single_traces,2))])%n=
            
            
            currfile=strcat('PeriReward_Fl_single_rew.fig');
            savefig(currfile)
            
            %%%%%%%%%%%%%%
            %non-rewarded licks
            all_rew_lick=single_lick_idx;
            if exist('double_lick_idx','var')
                all_rew_lick=[all_rew_lick double_lick_idx];%combine arrays, could also use union but should not have replicates
            end
            if exist('triple_lick_idx','var')
                all_rew_lick=[all_rew_lick triple_lick_idx];
            end
            
            nr_lick=bwlabel(supraLick');
            for i=1:length(all_rew_lick)
                start_exc_wind=all_rew_lick(1,i)-exclusion_win_frames;
                stop_exc_wind=all_rew_lick(1,i)+exclusion_win_frames;
                if start_exc_wind<0
                    start_exc_wind=1;
                    nr_lick(start_exc_wind:stop_exc_wind,1)=0;
                else
                    nr_lick(start_exc_wind:stop_exc_wind,1)=0;
                end
                %     nr_lick((all_rew_lick(1,i)-exclusion_win_frames):(all_rew_lick(1,i)+exclusion_win_frames),1)=0;
            end
            nr_lick(1:exclusion_win_frames,1)=0;%get rid of non-rewarded licks at start, otherwise will crash when grab traces
            nr_lick(end-exclusion_win_frames:end,1)=0;%same at end
            nr_lick=bwlabel(nr_lick);
            nr_lick_idx=[];
            for i=1:max(nr_lick)
                nr_lick_idx(i)= (find(nr_lick==i,1,'first'));%
            end
            nr_traces=zeros(pre_win_frames+post_win_frames+1,length(nr_lick_idx));
            for i=1:length(nr_lick_idx)
                currentnrlickidxperplane = find(timedFF>=utimedFF(nr_lick_idx(i)),1);
                nr_traces(:,i)=base_mean(currentnrlickidxperplane-pre_win_frames:currentnrlickidxperplane+post_win_frames)';%lick at pre_win_frames+1
            end
            norm_nr_traces=nr_traces./mean(nr_traces(1:pre_win_frames,:));
            
            figure;
            hold on;
            title('Non-rewarded licks');
            xlabel('seconds from non-rewarded lick')
            ylabel('dF/F')
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,norm_nr_traces,'Color',[.8 .8 .8]);
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,nanmean(norm_nr_traces,2),'k','LineWidth',2);
            legend(['n = ',num2str(size(norm_nr_traces,2))])
            
            currfile=strcat('non_rew_licks.fig');
            savefig(currfile)
            
            save('params','norm_nr_traces','nr_traces','nr_lick_idx','all_rew_lick','nr_lick','-append');
            
            
            %%%%%%%%%%%%
            %for doubles
            double_traces=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
            
            
            double_traces_roesmth=zeros(pre_win_framesALL+post_win_framesALL+1,length(double_lick_idx));
            coeff_doublerewarded_licks=[]; coeff_norm_doublerewarded_licks=[];  lags_double_traces=[];
            if ~isempty(double_lick_idx)
                for i=1:length(double_lick_idx)
                    currentdoublelickidxperplane = find(timedFF>=utimedFF(double_lick_idx(i)),1);
                    
                    double_traces(:,i)=base_mean(currentdoublelickidxperplane-pre_win_frames:currentdoublelickidxperplane+post_win_frames)';%lick at pre_win_frames+1
                    double_traces_roesmth(:,i)=speed_smth_1(double_lick_idx(i)-pre_win_framesALL:double_lick_idx(i)+post_win_framesALL)';
                    %                     [rho, pval]=corrcoef(double_traces(:,i),double_traces_roesmth(:,i));
                    %                     coeff_doublerewarded_licks(i,:)=[rho(1,2) pval(1,2)];
                    
                    %%% compute cross-corr and find the lag
                    %                     s2=double_traces(:,i);
                    %                     s1=double_traces_roesmth(:,i);
                    %                     [C21,lag21] = xcorr(s2,s1);
                    %                     C21 = C21/max(C21);
                    %                     [M21,I21] = max(C21);
                    %                     t21 = lag21(I21);
                    %                     lags_double_traces(1,i)=t21;
                end
            else
                double_traces = NaN(length(-pre_win_frames:post_win_frames));
                double_traces_roesmth = NaN(length(-pre_win_framesALL:post_win_framesALL));
            end
            norm_double_traces=double_traces./mean(double_traces(1:pre_win_frames,:));
            norm_double_traces_roesmth=double_traces_roesmth./mean(double_traces_roesmth(1:pre_win_framesALL,:));
            %save variables in params
            save('params','double_traces','double_traces_roesmth','norm_double_traces','norm_double_traces_roesmth','-append')
            
            %plot doubles and save figure
            
            figure;
            hold on;
            title('Double rewards');
            xlabel('seconds from reward lick')
            ylabel('dF/F')
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,norm_double_traces);%auto color
            plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,mean(norm_double_traces,2),'k','LineWidth',2);
            legend(['n = ',num2str(size(norm_double_traces,2))])%n=
            %legend()
            
            
            currfile=strcat('PeriReward_Fl_double_rew.fig');
            savefig(currfile)
            
            
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if exist('single_traces','var')
                %                 find_figure(strcat(mouse_id,'_perireward','_plane',num2str(allplanes)))
                %                 subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),
                %                 xlabel('seconds from first reward lick')
                %                 ylabel('dF/F')
                %                 striptitleindx = strfind(pr_dir0{Day},'\');
                %                 title(pr_dir0{Day}(striptitleindx(end)+1:end))
                %                 day_labels{Day} = pr_dir0{Day}(striptitleindx(end)+1:end);
                %                 hold on
                %                 subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces);
                %                 hold on
                %                 subplot(ceil(sqrt(length(pr_dir0))),ceil(sqrt(length(pr_dir0))),cnt),plot(mean(norm_single_traces,2),'k','LineWidth',2);
                %                 legend(['n = ',num2str(size(norm_single_traces,2))])
                %
                %                 plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(norm_single_traces_roesmth,0.99,1))
                %
                
                %%%%moving and stop activity
                
                
                %                 find_figure('dop_forwardvel_control_binned_activity');clf;
                %                 subplot(4,1,allplanes),plot(norm_base_mean)
                %                 hold on
                %                 forwardvel=smoothdata(speed_binned,'gaussian',gauss_win)';
                %                 plot(rescale(forwardvel,0.999,1.05),'k')
                
                
                
                
                if size(forwardvelALL,2) == 1
                    [moving_middle stop]=get_moving_time_V3(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)),speed_thresh,Stopped_frame,speedftol);
                else
                    [moving_middle stop]=get_moving_time_V3(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5))',speed_thresh,Stopped_frame,speedftol);
                end
                mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                
                
                
                
                
                
                
                
                
                idx_rm=(mov_success_tmpts- pre_win_framesALL)<=0;
                rm_idx=find(idx_rm==1)
                mov_success_tmpts(rm_idx)=[];
                
                mov_success_tmpts(end)=[];
                idx_rm=(mov_success_tmpts+post_win_framesALL)>=length(utimedFF)-10;
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
                
                
                stop_success_tmpts(end)=[];
                idx_rm=(stop_success_tmpts+post_win_framesALL)>length(utimedFF)-10;
                rm_idx=find(idx_rm==1)
                stop_success_tmpts(rm_idx)=[];
                
                rew_idx = find(reward_binned);
                rew_stop_success_tmpts = [];
                
                
                
                for r = 1:length(rew_idx)
                    if ~isempty(find(stop == rew_idx(r)))
                        if ~isempty(stop_success_tmpts(find(stop_success_tmpts<rew_idx(r),1,'last')))
                            rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts<rew_idx(r),1,'last'));
                        else
                            rew_stop_success_tmpts(r) = NaN;
                        end
                    elseif ~isempty(find(stop_success_tmpts-rew_idx(r)>=0 & stop_success_tmpts-rew_idx(r) <max_reward_stop,1))
                        rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts-rew_idx(r)>=0 & stop_success_tmpts-rew_idx(r) <max_reward_stop,1));
                    else
                        rew_stop_success_tmpts(r) = NaN;
                    end
                end
                didntstoprew = find(isnan(rew_stop_success_tmpts));
                rew_stop_success_tmpts(isnan(rew_stop_success_tmpts)) = [];
                 rew_stop_success_tmpts=unique(rew_stop_success_tmpts);
                nonrew_stop_success_tmpts = setxor(rew_stop_success_tmpts,stop_success_tmpts);
                
                
%                 find_figure('velocity');clf
%                 vr_speed=smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5));
%                 plot(vr_speed)
%                 hold on
%                 vr_speed2=zeros(1,length(vr_speed)); vr_speed2(moving_middle)=1; vr_speed2(find(vr_speed2==0))=NaN;
%                 plot(vr_speed.*vr_speed2,'r.')
%                 vr_speed2=zeros(1,length(vr_speed));vr_speed2(stop)=1; vr_speed2(find(vr_speed2==0))=NaN;
%                 plot(vr_speed.*vr_speed2,'k.')
%                 rew_idx2=zeros(1,length(forwardvelALL)); rew_idx2(rew_idx)=1; rew_idx2(find(rew_idx2==0))=NaN;
%                 plot(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)).*rew_idx2,'bo')
%                 hold on
%                 tempspeed = smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5));
%                 stop_tmpts2=zeros(1,length(forwardvelALL)); stop_tmpts2(stop_success_tmpts)=1; stop_tmpts2(find(stop_tmpts2==0))=NaN;
%                 plot(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)).*stop_tmpts2,'go')
%                 scatter(nonrew_stop_success_tmpts,tempspeed(nonrew_stop_success_tmpts),'ys','filled')
%                 scatter(rew_stop_success_tmpts,tempspeed(rew_stop_success_tmpts),'bs','filled')
% %                 %
%                 plot(rescale(supraLick,-10,-5),'Color',[0.7 0.7 0.7])
                
                save('params','didntstoprew','rew_stop_success_tmpts','nonrew_stop_success_tmpts','stop_success_tmpts','mov_success_tmpts','-append')
                
                
                
                dop_success_peristop=[]; roe_success_peristop=[];
                allstop_success=NaN(1,length(base_mean));
                allstop_success(setxor(1:length(base_mean),moving_middle)) = 1;
                for stamps=1:length(stop_success_tmpts)
                    currentnrrewstopidxperplane = find(timedFF>=utimedFF(stop_success_tmpts(stamps)),1);
                    dop_success_peristop(stamps,:)= base_mean(currentnrrewstopidxperplane-pre_win_frames:currentnrrewstopidxperplane+post_win_frames);
                    roe_success_peristop(stamps,:)= forwardvelALL(stop_success_tmpts(stamps)-pre_win_framesALL:stop_success_tmpts(stamps)+post_win_framesALL);
                end
                
                find_figure('velocity')
                
                rew_idx2=zeros(1,length(forwardvelALL)); rew_idx2(rew_idx)=1; rew_idx2(find(rew_idx2==0))=NaN;
                plot(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)).*rew_idx2,'bo')
                
                %  non rewarded stops
                dop_success_peristop_no_reward=[]; roe_success_peristop_no_reward=[];
                if ~isempty(nonrew_stop_success_tmpts)
                    for stamps=1:length(nonrew_stop_success_tmpts)
                        
                        currentnrrewstopidxperplane = find(timedFF>=utimedFF(nonrew_stop_success_tmpts(stamps)),1);
                        dop_success_peristop_no_reward(stamps,:)= base_mean(currentnrrewstopidxperplane-pre_win_frames:currentnrrewstopidxperplane+post_win_frames);
                        roe_success_peristop_no_reward(stamps,:)= forwardvelALL(nonrew_stop_success_tmpts(stamps)-pre_win_framesALL:nonrew_stop_success_tmpts(stamps)+post_win_framesALL);
                        
                    end
                else
                    dop_success_peristop_no_reward=NaN(length(-pre_win_frames:post_win_frames));
                    roe_success_peristop_no_reward= NaN(length(-pre_win_framesALL:post_win_framesALL));
                end
                
                
                % rewarded stops
                dop_success_peristop_reward=[]; roe_success_peristop_reward=[];
                if ~isempty(rew_stop_success_tmpts)
                    for stamps=1:length(rew_stop_success_tmpts)
                        
                        currentnrrewstopidxperplane = find(timedFF>=utimedFF(rew_stop_success_tmpts(stamps)),1);
                        dop_success_peristop_reward(stamps,:)= base_mean(currentnrrewstopidxperplane-pre_win_frames:currentnrrewstopidxperplane+post_win_frames);
                        roe_success_peristop_reward(stamps,:)= forwardvelALL(rew_stop_success_tmpts(stamps)-pre_win_framesALL:rew_stop_success_tmpts(stamps)+post_win_framesALL);
                    end
                else
                    dop_success_peristop_reward=NaN(length(-pre_win_frames:post_win_frames));
                    roe_success_peristop_reward=NaN(length(-pre_win_framesALL:post_win_framesALL));
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
                
                if size(dop_success_peristop_no_reward,1)>1
                    dim_c=1;
                else
                    dim_c=3;
                end
                
                
                
                dop_allsuc_stop_no_reward(alldays,allplanes,:)=mean(dop_success_peristop_no_reward,dim_c);
                
                if size(dop_success_peristop_reward,1)>1
                    dim_c=1;
                else
                    dim_c=3;
                end
                
                dop_allsuc_stop_reward(alldays,allplanes,:)=mean(dop_success_peristop_reward,dim_c);
                
                
                
                roe_allsuc_mov(alldays,allplanes,:)=mean(roe_success_perimov,1);
                roe_allsuc_stop(alldays,allplanes,:)=mean(roe_success_peristop,1);
                
              
                %%%%% perireward
                dop_alldays_planes_perireward_0{alldays,allplanes}=single_traces;
                roe_alldays_planes_perireward_0{alldays,allplanes}=single_traces_roesmth;
                
                dop_alldays_planes_perireward_double_0{alldays,allplanes} = double_traces;
                roe_alldays_planes_perireward_double_0{alldays,allplanes} = double_traces_roesmth;
                  if ~isempty(roe_success_peristop_no_reward)
                    roe_allsuc_stop_no_reward(alldays,allplanes,:)=mean(roe_success_peristop_no_reward,1);
                else
                    roe_allsuc_stop_no_reward(alldays,allplanes,:)=NaN(1,1,313);
                end
                if ~isempty(roe_success_peristop_reward)
                    roe_allsuc_stop_reward(alldays,allplanes,:)=mean(roe_success_peristop_reward,1);
                else
                    roe_allsuc_stop_reward(alldays,allplanes,:) = NaN(1,1,313);
                end
                
                dop_alldays_planes_perireward{alldays,allplanes}=norm_single_traces;
                roe_alldays_planes_perireward{alldays,allplanes}=norm_single_traces_roesmth;
                
                dop_alldays_planes_perireward_double{alldays,allplanes} = norm_double_traces;
                roe_alldays_planes_perireward_double{alldays,allplanes} = norm_double_traces_roesmth;
                
                dop_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces,2);
                roe_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces_roesmth,2);
                
                dop_allsuc_perireward_double(alldays,allplanes,:) = mean(norm_double_traces,2);
                roe_allsuc_perireward_double(alldays,allplanes,:) = mean(norm_double_traces_roesmth,2);
                
                dop_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces,[],2)./sqrt(size(norm_single_traces,2));
                roe_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces_roesmth,[],2)./sqrt(size(norm_single_traces_roesmth,2));
                
                dop_allsuc_perireward_double_se(alldays,allplanes,:)=std(norm_double_traces,[],2)./sqrt(size(norm_double_traces,2));
                roe_allsuc_perireward_double_se(alldays,allplanes,:)=std(norm_double_traces_roesmth,[],2)./sqrt(size(norm_double_traces_roesmth,2));
                
                
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
                
                
                
                
                


            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%% for regions of interest

            if str2num(pr_dir0{alldays}(strfind(pr_dir0{alldays},'157'):19))==157
                id_loc= strfind(pr_dir0{alldays},'_');
                dayc=str2num(pr_dir0{alldays}(id_loc(2)+1:end))


                if oldbatch==1 & dayc<7  %%%% for E157 DAY3: DAY6 ROIBASEMEAN3 AND REST DAYS ROIBASEMEAN2

                    df_f=params.roibasemean3;
                elseif oldbatch==1 & dayc>=7
                    df_f=params.roibasemean2;
                end

            else
                if oldbatch==1  %%%% fore156

                    df_f=params.roibasemean2;
                else
                    df_f=params.roibasemean3;
                end
            end


            for roii = 1:size(df_f,1)
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                %                 figure;
                %                 plot(rescale(reward_binned,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
                %                 hold on
                %                 plot(rescale(speed_smth_1,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
                %                 plot(roidop_smth,'LineWidth',1.5)
                %                 legend({'Reward','Speed','Dopamine'})
                %                 ylabel('dF/F')
                
                
                figure;
                plot(utimedFF,rescale(speed_smth_1,min(roidop_smth)-range(roidop_smth),min(roidop_smth)),'LineWidth',1.5)
                hold on
                plot(timedFF,roidop_smth,'LineWidth',1.5)
                plot(utimedFF,rescale(reward_binned,min(roidop_smth),max(roidop_smth)),'LineWidth',1.5)
                legend({'Reward','Speed','Dopamine'})
                ylabel('dF/F')
                savefig(['ROI_' num2str(roii) 'Fl_Speed_Reward.fig'])
                
                %%
                roisingle_traces=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
                
                
                single_traces_roesmth=zeros(pre_win_framesALL+post_win_framesALL+1,length(single_lick_idx));
                coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
                for i=1:length(single_lick_idx)
                    currentrewidxperplane = find(timedFF>=utimedFF(single_lick_idx(i)),1);
                    roisingle_traces(:,i)=roibase_mean(currentrewidxperplane-pre_win_frames:currentrewidxperplane+post_win_frames)';%lick at pre_win_frames+1
                    single_traces_roesmth(:,i)=speed_smth_1(single_lick_idx(i)-pre_win_framesALL:single_lick_idx(i)+post_win_framesALL)';
                    
                end
                roinorm_single_traces=roisingle_traces./mean(roisingle_traces(1:pre_win_frames,:));
                norm_single_traces_roesmth=single_traces_roesmth./mean(single_traces_roesmth(1:pre_win_framesALL,:));
                
                %plot for single reward
                
                figure;
                hold on;
                title(['ROI ' num2str(roii) ' Single rewards']);
                xlabel('seconds from reward lick')
                ylabel('dF/F')
                plot(frame_time*numplanes*(-pre_win_frames):frame_time*numplanes:frame_time*numplanes*post_win_frames,roinorm_single_traces);%auto color
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_single_traces,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_single_traces,2))])%n=
                
                currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_rew.fig']);
                savefig(currfile)
                %%
                
                %%%% for single rew CS
                
                singlerew = single_rew(find(single_rew>pre_win_frames&single_rew<length(licksALL)-post_win_frames))-CSUSframelag_win_frames;
                singlerew(end)=[];
                
                
                %%%
                
                %%%%%%%%%%
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
                roinorm_single_traces_roesmthCS=roisingle_traces_roesmthCS./mean(roisingle_traces_roesmthCS(1:pre_win_framesALL,:));
                
                
                save('params','roinorm_single_tracesCS','roinorm_single_traces_roesmthCS','-append')
                
                
                %plot for single reward CS
                
                figure;
                hold on;
                title(['ROI ' num2str(roii) ' Single rewards CS']);
                xlabel('seconds from CS')
                ylabel('dF/F')
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames,roinorm_single_tracesCS);%auto color
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames,mean(roinorm_single_tracesCS,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_single_tracesCS,2))])%n=
                
                currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_rewCS.fig']);
                savefig(currfile)
                
                
                
                %%
                %%%%%%% for single rew US
                singlerew = single_rew(find(single_rew>pre_win_framesALL&single_rew<length(licksALL)-post_win_framesALL));
                roisingle_tracesUS=zeros(pre_win_frames+post_win_frames+1,length(singlerew));
                
                
                roisingle_traces_roesmthUS=zeros(pre_win_framesALL+post_win_framesALL+1,length(singlerew));
                for i=1:length(singlerew)
                    currentnrrewUSidxperplane = find(timedFF>=utimedFF(singlerew(i)),1);
                    roisingle_tracesUS(:,i)=roibase_mean(currentnrrewUSidxperplane-pre_win_frames:currentnrrewUSidxperplane+post_win_frames)';%lick at pre_win_frames+1
                    roisingle_traces_roesmthUS(:,i)=speed_smth_1(singlerew(i)-pre_win_framesALL:singlerew(i)+post_win_framesALL)';
                    
                end
                roinorm_single_tracesUS=roisingle_tracesUS./mean(roisingle_tracesUS(1:pre_win_frames,:));
                roinorm_single_traces_roesmthUS=roisingle_traces_roesmthUS./mean(roisingle_traces_roesmthUS(1:pre_win_framesALL,:));
                
                
                save('params','roinorm_single_tracesUS','roinorm_single_traces_roesmthUS','-append')
                
                %plot for single reward US
                
                figure;
                hold on;
                title(['ROI ' num2str(roii) ' Single rewards US']);
                xlabel('seconds from US')
                ylabel('dF/F')
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames,roinorm_single_tracesUS);%auto color
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*numplanes*post_win_frames,mean(roinorm_single_tracesUS,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_single_tracesUS,2))])%n=
                
                currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_rewUS.fig']);
                savefig(currfile)
                
                %%%%%%%%%%%%%%
                %%
                %non-rewarded licks
                all_rew_lick=single_lick_idx;
                if exist('double_lick_idx','var')
                    all_rew_lick=[all_rew_lick double_lick_idx];%combine arrays, could also use union but should not have replicates
                end
                if exist('triple_lick_idx','var')
                    all_rew_lick=[all_rew_lick triple_lick_idx];
                end
                
                nr_lick=bwlabel(supraLick);
                for i=1:length(all_rew_lick)
                    start_exc_wind=all_rew_lick(1,i)-exclusion_win_frames;
                    stop_exc_wind=all_rew_lick(1,i)+exclusion_win_frames;
                    if start_exc_wind<0
                        start_exc_wind=1;
                        nr_lick(1,start_exc_wind:stop_exc_wind)=0;
                    else
                        nr_lick(1,start_exc_wind:stop_exc_wind)=0;
                    end
                    %     nr_lick((all_rew_lick(1,i)-exclusion_win_frames):(all_rew_lick(1,i)+exclusion_win_frames),1)=0;
                end
                nr_lick(1:exclusion_win_frames)=0;%get rid of non-rewarded licks at start, otherwise will crash when grab traces
                nr_lick(end-exclusion_win_frames:end)=0;%same at end
                nr_lick=bwlabel(nr_lick);
                for i=1:max(nr_lick)
                    nr_lick_idx(i)= (find(nr_lick==i,1,'first'));%
                end
                roinr_traces=zeros(pre_win_frames+post_win_frames+1,length(nr_lick_idx));
                for i=1:length(nr_lick_idx)
                    currentidx = find(timedFF>=utimedFF(nr_lick_idx(i)),1);
                    roinr_traces(:,i)=roibase_mean(currentidx-pre_win_frames:currentidx+post_win_frames)';%lick at pre_win_frames+1
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
                
                
                
                save('params','roinorm_nr_traces','roinr_traces','nr_lick_idx','all_rew_lick','nr_lick','-append');
                %%%%%%%%%%%%%%%%%%
                
                
                %%
                %%
                %%%%%%% for UNREWARD SOLENOID



                if exist('urew_solenoidALL','var')
                    unrew_single=find(urew_solenoidALL);

                  
                    unrew_single(find(diff(unrew_single) == 1)) = 0;
                    unrew_single=unrew_single(find(unrew_single));
                    

                    
                    %placed an error here for you to check this line. changed
                    %urew_solenoid to urew_solenoidall. Do you need to change
                    %the 4 here? or not? not sure what this line does
                    singleunrew = unrew_single(find(unrew_single>pre_win_framesALL& unrew_single<length(licksALL)-post_win_framesALL));
                    roiunrew_single_traces_CS=zeros(pre_win_frames+post_win_frames+1,length(singleunrew));
                    
                    
                    roiunrew_single_traces_roesmthCS=zeros(pre_win_framesALL+post_win_framesALL+1,length(singleunrew));
                    if ~isempty(unrew_single)
                        for i=1:length(unrew_single)
                            currentunrewidxperplane = find(timedFF>=utimedFF(singleunrew(i)),1);
                            roiunrew_single_traces_CS(:,i)=roibase_mean(currentunrewidxperplane-pre_win_frames:currentunrewidxperplane+post_win_frames)';%lick at pre_win_frames+1
                            roiunrew_single_traces_roesmthCS(:,i)=speed_smth_1(singleunrew(i)-pre_win_framesALL:singleunrew(i)+post_win_framesALL)';
                            
                            
                        end
                    else
                        roiunrew_single_traces_CS = NaN(length(-pre_win_frames:post_win_frames));
                        roiunrew_single_traces_roesmthCS = NaN(length(-pre_win_framesALL:post_win_framesALL));
                    end
                    roinorm_unrew_single_tracesCS=roiunrew_single_traces_CS./mean(roiunrew_single_traces_CS(1:pre_win_frames,:));
                    roinorm_unrew_single_traces_roesmthCS=roiunrew_single_traces_roesmthCS./mean(roiunrew_single_traces_roesmthCS(1:pre_win_framesALL,:));
                    
                    save('params','roinorm_unrew_single_tracesCS','roinorm_unrew_single_traces_roesmthCS','-append')
                    
                    
                    %plot for single reward US
                    
                    figure;
                    hold on;
                    title(['ROI ' num2str(roii) ' unrewarded Single rewards CS']);
                    xlabel('seconds from UnrewS')
                    ylabel('dF/F')
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_unrew_single_tracesCS);%auto color
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_unrew_single_tracesCS,2),'k','LineWidth',2);
                    legend(['n = ',num2str(size(roinorm_unrew_single_tracesCS,2))])%n=
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes,rescale(mean(roinorm_unrew_single_traces_roesmthCS,2),0.99,0.995),'k','LineWidth',2);
                    
                    
                    currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_unrew _CS.fig']);
                    savefig(currfile)
                end
                %%%%%%%%%%%%%%
                
                %%
                %%%%
                %%%%%%% for REWARD WITHOUT CS SOLENOID
                
                if exist('rew_us_solenoidALL','var')
                    if length(find(rew_us_solenoidALL))>0
                        %placed an error here for you to check this line. changed
                        %rew_us_solenoid to rew_us_solenoidall. Do you need to change
                        %the 4 here? or not? not sure what this line does
                        rew_us_single=find(rew_us_solenoidALL);
                        rew_us_single(find(diff(rew_us_single)==1))=[];
                        singlerew_us = rew_us_single(find(rew_us_single>pre_win_framesALL& rew_us_single<length(licksALL)-post_win_framesALL));
                        singlerew_us(end)=[];
                        roirew_us_single_traces_US=zeros(pre_win_frames+post_win_frames+1,length(singlerew_us));
                        roirew_us_single_traces_roesmthUS=zeros(pre_win_framesALL+post_win_framesALL+1,length(singlerew_us));
                        if ~isempty(rew_us_single)
                            for i=1:length(rew_us_single)-1
                                currentunrewidxperplane = find(timedFF>=utimedFF(rew_us_single(i)),1);
                                roirew_us_single_traces_US(:,i)=roibase_mean(currentunrewidxperplane-pre_win_frames:currentunrewidxperplane+post_win_frames)';%lick at pre_win_frames+1
                                roirew_us_single_traces_roesmthUS(:,i)=speed_smth_1(singlerew_us(i)-pre_win_framesALL:singlerew_us(i)+post_win_framesALL)';
                                
                                
                            end
                        else
                            roirew_us_single_traces_US = NaN(length(-pre_win_frames:post_win_frames));
                            roirew_us_single_traces_roesmthUS = NaN(length(-pre_win_frames:post_win_frames));
                        end
                        roinorm_rew_us_single_tracesUS=roirew_us_single_traces_US./mean(roirew_us_single_traces_US(1:pre_win_frames,:));
                        roinorm_rew_us_single_traces_roesmthUS=roirew_us_single_traces_roesmthUS./mean(roirew_us_single_traces_roesmthUS(1:pre_win_framesALL,:));
                        
                        save('params','roinorm_rew_us_single_tracesUS','roinorm_rew_us_single_traces_roesmthUS','-append')
                        
                        
                        
                        
                        %plot for single reward US
                        
                        figure;
                        hold on;
                        title(['ROI ' num2str(roii) ' rewarded Single rewards US only']);
                        xlabel('seconds from rewarded US')
                        ylabel('dF/F')
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_rew_us_single_tracesUS);%auto color
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_rew_us_single_tracesUS,2),'k','LineWidth',2);
                        legend(['n = ',num2str(size(roinorm_rew_us_single_tracesUS,2))])%n=
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes,rescale(mean(roinorm_rew_us_single_traces_roesmthUS,2),0.99,0.995),'k','LineWidth',2);
                        
                        
                        currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_single_rew_usonly_US.fig']);
                        savefig(currfile)
                    end
                end
                
                
                %%
                if exist('singlerew_us','var')
                    if ~isempty(singlerew_us)
                        single_us_idx=[];single_lick_us_idx=[]; single_lick_us_gap = [];
                        
                        allrew=zeros(1,length(rewards));
                        allrew(single_rew)=1;%%%ALL CS
                        allrew(singlerew_us)=2;%%%REWARD ADDIT
                        
                        
                        for i=1:length(singlerew_us)-1
                            
                            single_us_idx=singlerew_us(i);
                            if single_us_idx+rew_lick_win_frames < length(supraLick)%if window to search for lick after rew is past length of supraLick, doesn't update single_lick_idx, but single_idx is
                                %                     single_lick_idx(i) = (find(supraLick(single_idx(i):single_idx(i)+rew_lick_win_frames),1,'first'))+single_idx(i)-1;
                                %                     %looks for first lick after rew with window =exclusion_win_frames
                                %                     %however first lick can be much further in naive animals
                                [val idz]=min(abs((single_rew-singlerew_us(i))));
                                
                                %                 end
                                prepostid= [single_rew(idz) single_rew(idz+1)];
                                lick_exist=find(supraLick(prepostid(1):prepostid(2)-5),1,'first')+single_us_idx-1
                                %                         lick_exist=(find(supraLick(single_us_idx:single_us_idx+rew_lick_win_frames),1,'first'))+single_us_idx-1;
                                if isempty(lick_exist)~=1
                                    
                                    %                             single_lick_us_idx(i) = (find(supraLick(single_us_idx:single_us_idx+rew_lick_win_frames),1,'first'))+single_us_idx-1;
                                    
                                    prepostid= [single_rew(idz) single_rew(idz+1)];
                                    single_lick_us_idx(i) =find(supraLick(prepostid(1):prepostid(2)-5),1,'first')+single_us_idx-1;
                                    
                                    %looks for first lick after rew with window =exclusion_win_frames
                                    %however first lick can be much further in naive animals
                                    single_lick_us_gap(i) = single_lick_us_idx(i)-single_us_idx;
                                else
                                    warning('no lick after reward was delivered!!!');
                                    single_lick_us_idx(i)= NaN;
                                    single_lick_us_gap(i) = NaN;
                                end
                            end
                        end
                        didntlicksingle_us = find(isnan(single_lick_us_idx));
                        single_lick_us_idx(didntlicksingle_us) = [];
                        if ~isempty(single_lick_us_idx)
                            if single_lick_us_idx(1) - pre_win_framesALL <0%remove events too early
                                single_lick_us_idx(1)=[];
                            end
                            if single_lick_us_idx(end) + post_win_framesALL > length(supraLick)%remove events too late
                                single_lick_us_idx(end)=[];
                            end
                        end
                        
                        %save variables in params
                        save('params','didntlicksingle_us','single_lick_us_idx','single_lick_us_gap','-append')
                        
                        
                        %%%for
                        %%%% if no lick after reward strange though!!
                        
                        [r c]=find(single_lick_us_idx==0);
                        single_lick_us_idx(c)=[];
                        roisingle_traces_us_only=zeros(pre_win_frames+post_win_frames+1,length(single_lick_us_idx));
                        
                        
                        roisingle_traces_us_only_roesmth=zeros(pre_win_framesALL+post_win_framesALL+1,length(single_lick_us_idx));
                        coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
                        for i=1:length(single_lick_us_idx)
                            currentUSlickidxperplane = find(timedFF>=utimedFF(single_lick_us_idx(i)),1);
                            roisingle_traces_us_only(:,i)=roibase_mean(currentUSlickidxperplane-pre_win_frames:currentUSlickidxperplane+post_win_frames)';%lick at pre_win_frames+1
                            roisingle_traces_us_only_roesmth(:,i)=speed_smth_1(single_lick_us_idx(i)-pre_win_framesALL:single_lick_us_idx(i)+post_win_framesALL)';
                            
                        end
                        roinorm_single_traces_us_only= roisingle_traces_us_only./mean( roisingle_traces_us_only(1:pre_win_frames,:));
                        roinorm_single_traces_us_only_roesmth=  roisingle_traces_us_only_roesmth./mean(  roisingle_traces_us_only_roesmth(1:pre_win_framesALL,:));
                        %save variables in params
                        save('params','roisingle_traces_us_only','roisingle_traces_us_only_roesmth','roinorm_single_traces_us_only','roinorm_single_traces_us_only_roesmth','-append')
                        
                        
                        %plot for single reward
                        
                        figure;
                        hold on;
                        title('Single rewards US only');
                        xlabel('seconds from reward lick US only')
                        ylabel('dF/F')
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes, roinorm_single_traces_us_only);%auto color
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean( roinorm_single_traces_us_only,2),'k','LineWidth',2);
                        legend(['n = ',num2str(size(roinorm_single_traces_us_only,2))])%n=
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes,rescale(mean(roinorm_single_traces_us_only_roesmth,2),0.99,0.995),'--k','LineWidth',2);
                        
                        
                        currfile=strcat('PeriReward_Fl_afterlick_single_rew_US_only.fig');
                        savefig(currfile)
                        
                        
                        %% assuming double reward if US solenoid is missed
                        
                        if length(single_lick_us_idx)~=length(singlerew_us)
                            idz_c=[];
                            for jj=1:length(single_lick_us_idx);
                                [val ids]=min(abs(singlerew_us-single_lick_us_idx(jj)));
                                idz_c=[idz_c ids];
                            end
                            single_lick_us_idx_nr=singlerew_us;
                            single_lick_us_idx_nr(idz_c)=[];
                        else
                            single_lick_us_idx_nr=[];
                        end
                        if ~isempty(single_lick_us_idx_nr)
                            drew_mUS=[];
                            for kk=1:length(single_lick_us_idx_nr)
                                
                                [val ids]=min(abs(singlerew-single_lick_us_idx_nr(kk)));
                                doubleCS_missedUS= single_rew(ids+1);
                                drew_mUS=[drew_mUS doubleCS_missedUS];
                                
                            end
                        else
                            drew_mUS=[];
                            
                        end
                        
                        
                        if ~isempty( drew_mUS)
                            roidoublerew_mus_single_traces_CS=zeros(pre_win_frames+post_win_frames+1,length(drew_mUS));
                            roidoublerew_mus_single_traces_roesmthCS=zeros(pre_win_framesALL+post_win_framesALL+1,length(drew_mUS));
                            for i=1:length( drew_mUS)
                                currentUSlickidxperplane = find(timedFF>=utimedFF( drew_mUS(i)),1);
                                roidoublerew_mus_single_traces_CS(:,i)=roibase_mean(currentUSlickidxperplane-pre_win_frames:currentUSlickidxperplane+post_win_frames)';%lick at pre_win_frames+1
                                roidoublerew_mus_single_traces_roesmthCS(:,i)=speed_smth_1(drew_mUS(i)-pre_win_framesALL:drew_mUS(i)+post_win_framesALL)';
                                
                                
                            end
                        else
                            roidoublerew_mus_single_traces_CS= NaN(length(-pre_win_frames:post_win_frames));
                            roidoublerew_mus_single_traces_roesmthCS = NaN(length(-pre_win_framesALL:post_win_framesALL));
                        end
                        roinorm_doublerew_mus_single_traces_CS= roidoublerew_mus_single_traces_CS./mean( roidoublerew_mus_single_traces_CS(1:pre_win_frames,:));
                        roinorm_doublerew_mus_single_traces_roesmthCS= roidoublerew_mus_single_traces_roesmthCS./mean( roidoublerew_mus_single_traces_roesmthCS(1:pre_win_framesALL,:));
                        
                        save('params','roinorm_doublerew_mus_single_traces_CS','roinorm_doublerew_mus_single_traces_roesmthCS','-append')
                        
                        
                        
                        
                        %plot for single reward US
                        
                        figure;
                        hold on;
                        title(['ROI ' num2str(roii) ' rewarded double rewards CS+US only']);
                        xlabel('seconds from rewarded US')
                        ylabel('dF/F')
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_doublerew_mus_single_traces_CS);%auto color
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_doublerew_mus_single_traces_CS,2),'k','LineWidth',2);
                        legend(['n = ',num2str(size(roinorm_doublerew_mus_single_traces_CS,2))])%n=
                        plot(frame_time*(-pre_win_frames)*numplanes:frame_time:frame_time*post_win_frames*numplanes,rescale(mean(roinorm_doublerew_mus_single_traces_roesmthCS,2),0.99,0.995),'k','LineWidth',2);
                        
                        
                        currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_drew_missedUS_double_rew_CS.fig']);
                        savefig(currfile)
                        
                        
                        
                        
                        
                    end
                    
                end
                
                
                %%
                %for doubles
                roidouble_traces=zeros(pre_win_frames+post_win_frames+1,length(double_lick_idx));
                
                
                double_traces_roesmth=zeros(pre_win_framesALL+post_win_framesALL+1,length(double_lick_idx));
                if ~isempty(double_lick_idx)
                    for i=1:length(double_lick_idx)
                        
                        currentdoublerewidx = find(timedFF>=utimedFF(double_lick_idx(i)),1);
                        roidouble_traces(:,i)=roibase_mean(currentdoublerewidx-pre_win_frames:currentdoublerewidx+post_win_frames)';%lick at pre_win_frames+1
                        double_traces_roesmth(:,i)=speed_smth_1(double_lick_idx(i)-pre_win_framesALL:double_lick_idx(i)+post_win_framesALL)';
                        
                    end
                else
                    roidouble_traces = NaN(length(-pre_win_frames:post_win_frames));
                    roidouble_traces_roesmth = NaN(length(-pre_win_framesALL:post_win_framesALL));
                end
                roinorm_double_traces=roidouble_traces./mean(roidouble_traces(1:pre_win_frames,:));
                norm_double_traces_roesmth=double_traces_roesmth./mean(double_traces_roesmth(1:pre_win_framesALL,:));
                
                
                %plot doubles and save figure
                
                figure;
                hold on;
                title(['ROI ' num2str(roii) ' Double rewards']);
                xlabel('seconds from reward lick')
                ylabel('dF/F')
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_double_traces);%auto color
                plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_double_traces,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(roinorm_double_traces,2))])%n=
                
                
                currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_double_rew.fig']);
                savefig(currfile)
                
                %%
                % for double rew CS
                if exist('double_idx','var')&~isempty(double_idx)
                    doublerew = double_idx-CSUSframelag_win_frames;
                    roidouble_tracesCS=zeros(pre_win_frames+post_win_frames+1,length(doublerew));
                    roidouble_traces_roesmthCS=zeros(pre_win_framesALL+post_win_framesALL+1,length(doublerew));
                    for i=1:length(doublerew)
                        currentdoubleCSidxperplane = find(timedFF>=utimedFF(doublerew(i)),1);
                        if currentdoubleCSidxperplane+post_win_frames<length(roibase_mean)
                        roidouble_tracesCS(:,i)=roibase_mean(currentdoubleCSidxperplane-pre_win_frames:currentdoubleCSidxperplane+post_win_frames)';%lick at pre_win_frames+1
                        roidouble_traces_roesmthCS(:,i)=speed_smth_1(doublerew(i)-pre_win_framesALL:doublerew(i)+post_win_framesALL)';
                        end 
                    end
                    roinorm_double_tracesCS=roidouble_tracesCS./mean(roidouble_tracesCS(1:pre_win_frames,:));
                    roinorm_double_traces_roesmthCS=roidouble_traces_roesmthCS./mean(roidouble_traces_roesmthCS(1:pre_win_framesALL,:));
                    save('params','roinorm_double_tracesCS','roinorm_double_traces_roesmthCS','-append')
                    
                    %plot doubles CS and save figure
                    
                    figure;
                    hold on;
                    title(['ROI ' num2str(roii) ' Double rewards CS']);
                    xlabel('seconds from double CS')
                    ylabel('dF/F')
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_double_tracesCS);%auto color
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_double_tracesCS,2),'k','LineWidth',2);
                    legend(['n = ',num2str(size(roinorm_double_tracesCS,2))])%n=
                    
                    
                    currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_double_rew_CS.fig']);
                    savefig(currfile)
                    
                    
                    
                    %%
                    
                    % for double rew US
                    doublerew = double_idx;
                    roidouble_tracesUS=zeros(pre_win_frames+post_win_frames+1,length(doublerew));
                    roidouble_traces_roesmthUS=zeros(pre_win_framesALL+post_win_framesALL+1,length(doublerew));
                    for i=1:length(doublerew)
                        currentdoubleUSidxperplane = find(timedFF>=utimedFF(doublerew(i)),1);
                        if currentdoubleUSidxperplane+post_win_frames<length(roibase_mean)
                        roidouble_tracesUS(:,i)=roibase_mean(currentdoubleUSidxperplane-pre_win_frames:currentdoubleUSidxperplane+post_win_frames)';%lick at pre_win_frames+1
                        roidouble_traces_roesmthUS(:,i)=speed_smth_1(doublerew(i)-pre_win_framesALL:doublerew(i)+post_win_framesALL)';
                        end
                    end
                    roinorm_double_tracesUS=roidouble_tracesUS./mean(roidouble_tracesUS(1:pre_win_frames,:));
                    roinorm_double_traces_roesmthUS=roidouble_traces_roesmthUS./mean(roidouble_traces_roesmthUS(1:pre_win_framesALL,:));
                    save('params','roinorm_double_tracesUS','roinorm_double_traces_roesmthUS','-append')
                    
                    %plot doubles US and save figure
                    
                    figure;
                    hold on;
                    title(['ROI ' num2str(roii) ' Double rewards US']);
                    xlabel('seconds from double US')
                    ylabel('dF/F')
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,roinorm_double_tracesUS);%auto color
                    plot(frame_time*(-pre_win_frames)*numplanes:frame_time*numplanes:frame_time*post_win_frames*numplanes,mean(roinorm_double_tracesUS,2),'k','LineWidth',2);
                    legend(['n = ',num2str(size(roinorm_double_tracesCS,2))])%n=
                    
                    
                    currfile=strcat(['ROI_' num2str(roii) '_PeriReward_Fl_double_rew_US.fig']);
                    savefig(currfile)
                    
                    
                end
                
                
                %%
                if exist('roisingle_traces','var')
                    
                    if size(forwardvelALL,2) == 1
                    [moving_middle stop]=get_moving_time_V3(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)),speed_thresh,Stopped_frame,speedftol);
                else
                    [moving_middle stop]=get_moving_time_V3(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5))',speed_thresh,Stopped_frame,speedftol);
                end
                    mov_success_tmpts = moving_middle(find(diff(moving_middle)>1)+1);
                    idx_rm=(mov_success_tmpts- pre_win_framesALL)<=0;
                    rm_idx=find(idx_rm==1)
                    mov_success_tmpts(rm_idx)=[];
                    
                    mov_success_tmpts(end)=[];
                    idx_rm=(mov_success_tmpts+post_win_framesALL)>length(forwardvelALL)-10;
                    rm_idx=find(idx_rm==1);
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
                    
                    
                    idx_rm= (stop_success_tmpts - pre_win_framesALL) <0;
                    rm_idx=find(idx_rm==1);
                    stop_success_tmpts(rm_idx)=[];
                    
                    stop_success_tmpts(end)=[];
                    idx_rm=(stop_success_tmpts+post_win_framesALL)>length(forwardvelALL)-10;
                    rm_idx=find(idx_rm==1)
                    stop_success_tmpts(rm_idx)=[];
                    
                    rew_idx = find(reward_binned);
                    rew_stop_success_tmpts = [];
                    for r = 1:length(rew_idx)
                        if ~isempty(find(stop == rew_idx(r)))
                            if ~isempty(stop_success_tmpts(find(stop_success_tmpts<rew_idx(r),1,'last')))
                                rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts<rew_idx(r),1,'last'));
                            else
                                rew_stop_success_tmpts(r) = NaN;
                            end
                        elseif ~isempty(find(stop_success_tmpts-rew_idx(r)>=0 & stop_success_tmpts-rew_idx(r) <max_reward_stop,1))
                            rew_stop_success_tmpts(r) = stop_success_tmpts(find(stop_success_tmpts-rew_idx(r)>=0 & stop_success_tmpts-rew_idx(r) <max_reward_stop,1));
                        else
                            rew_stop_success_tmpts(r) = NaN;
                        end
                    end
                    didntstoprew = find(isnan(rew_stop_success_tmpts));
                    rew_stop_success_tmpts(isnan(rew_stop_success_tmpts)) = [];
                    rew_stop_success_tmpts=unique(rew_stop_success_tmpts);
                    nonrew_stop_success_tmpts = setxor(rew_stop_success_tmpts,stop_success_tmpts);
                    
                    
           
                    idx_rm=(stop_success_tmpts - pre_win_framesALL)<=0;
                    rm_idx=find(idx_rm==1)
                    stop_success_tmpts(rm_idx)=[];
                    
                    stop_success_tmpts(end)=[];
                    idx_rm=(stop_success_tmpts+post_win_framesALL)>length(forwardvelALL)-10;
                    rm_idx=find(idx_rm==1);
                    stop_success_tmpts(rm_idx)=[];
                    


                    roidop_success_peristop=[];
                    roiallstop_success=NaN(1,length(forwardvelALL));
                    roiallstop_success(setxor(1:length(forwardvelALL),moving_middle)) = 1;
                    for stamps=1:length(stop_success_tmpts)
                        currentidx =  find(timedFF>=utimedFF(stop_success_tmpts(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_peristop(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                        end
                    end
                    
                    %%%% unrewstop with and without licks
                    
                    lick_idx = find(supraLick);
                    lick_norew_stop_success_tmpts = []; nolick_norew_stop_success_tmpts = [];
                    lickstamps=0; nolickstamps=0;
                    for ii=1:length(nonrew_stop_success_tmpts)
                        
                        if ~isempty(find(supraLick(nonrew_stop_success_tmpts(ii)-max_nrew_stop_licktol:nonrew_stop_success_tmpts(ii)+  max_nrew_stop_licktol)));
                            lickstamps=lickstamps+1;
                            lick_norew_stop_success_tmpts(lickstamps)=nonrew_stop_success_tmpts(ii);
                        else
                            nolickstamps=nolickstamps+1;
                            nolick_norew_stop_success_tmpts(nolickstamps)=nonrew_stop_success_tmpts(ii);
                        end
                    end

       
                    idx_rm=(nolick_norew_stop_success_tmpts- pre_win_framesALL)<=0;
                    rm_idx=find(idx_rm==1)
                    nolick_norew_stop_success_tmpts(rm_idx)=[];
                    
%                     nolick_norew_stop_success_tmpts(end)=[];
                    idx_rm=(nolick_norew_stop_success_tmpts+post_win_framesALL)>length(forwardvelALL)-10;
                    rm_idx=find(idx_rm==1);
                    nolick_norew_stop_success_tmpts(rm_idx)=[];
                    
%                    find_figure('velocity') 
%                    scatter(lick_norew_stop_success_tmpts-1,tempspeed(lick_norew_stop_success_tmpts),'mv','filled')
%                    scatter(  nolick_norew_stop_success_tmpts-1,tempspeed(  nolick_norew_stop_success_tmpts),'gv','filled')
%                    
                
                    
                    
                    %%
                    %  non rewarded stops
                    
                    idx_rm=(nonrew_stop_success_tmpts - pre_win_framesALL)<=0;
                    rm_idx=find(idx_rm==1)
                    nonrew_stop_success_tmpts(rm_idx)=[];
                    

                    idx_rm=(nonrew_stop_success_tmpts+post_win_framesALL)>length(forwardvelALL)-10;
                    rm_idx=find(idx_rm==1);
                    nonrew_stop_success_tmpts(rm_idx)=[];


                    roidop_success_peristop_no_reward=[];
                    for stamps=1:length(nonrew_stop_success_tmpts)
                        currentidx =  find(timedFF>=utimedFF(nonrew_stop_success_tmpts(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_peristop_no_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                        end
                    end

                    



                    % rewarded stops
                    roidop_success_peristop_reward=[];
                    
                    for stamps=1:length(rew_stop_success_tmpts)
                        if sum(~isnan(rew_stop_success_tmpts))>=1
                            currentidx =  find(timedFF>=utimedFF(rew_stop_success_tmpts(stamps)),1);
                            roidop_success_peristop_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                        else
                            roidop_success_peristop_reward=NaN(length(-pre_win_frames:post_win_frames));
                            
                        end
                    end

                    
                    
                    %%% NOn REWARDED STOPS WITH and without LICKS 
                    
                    roidop_lick_peristop_no_reward=[]; roe_success_lick_peristop_no_reward=[];
                    
                    for stamps=1:length(lick_norew_stop_success_tmpts)
                        if sum(~isnan(lick_norew_stop_success_tmpts))>=1
                            currentidx =  find(timedFF>=utimedFF(lick_norew_stop_success_tmpts(stamps)),1);
                            roidop_lick_peristop_no_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_lick_peristop_no_reward(stamps,:)= forwardvelALL(lick_norew_stop_success_tmpts(stamps)-pre_win_framesALL:lick_norew_stop_success_tmpts(stamps)+post_win_framesALL);
                            
                            
                        else
                            roidop_lick_peristop_no_reward=NaN(length(-pre_win_frames:post_win_frames));
                            roe_success_lick_peristop_no_reward=NaN(length(-pre_win_frames:post_win_frames));
                        end
                    end
                    
                    
                    roidop_nolick_peristop_no_reward=[]; roe_success_nolick_peristop_no_reward=[];
                    
                    for stamps=1:length(nolick_norew_stop_success_tmpts)
                        if sum(~isnan(nolick_norew_stop_success_tmpts))>=1
                            currentidx =  find(timedFF>=utimedFF(nolick_norew_stop_success_tmpts(stamps)),1);
                            roidop_nolick_peristop_no_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_nolick_peristop_no_reward(stamps,:)= forwardvelALL(nolick_norew_stop_success_tmpts(stamps)-pre_win_framesALL:nolick_norew_stop_success_tmpts(stamps)+post_win_framesALL);
                            
                        else
                            roidop_nolick_peristop_no_reward=NaN(length(-pre_win_frames:post_win_frames));
                            roe_success_nolick_peristop_no_reward=NaN(length(-pre_win_frames:post_win_frames));
                        end
                    end
                    
                    if ~isempty(roe_success_nolick_peristop_no_reward)
                        roe_allsuc_nolick_stop_no_reward(alldays,allplanes,:)=mean(roe_success_nolick_peristop_no_reward,1);
                    else
                        roe_allsuc_nolick_stop_no_reward(alldays,allplanes,:)=NaN(1,1,313);
                    end
                    if ~isempty(roe_success_lick_peristop_no_reward)
                        roe_allsuc_lick_stop_no_reward(alldays,allplanes,:)=mean(roe_success_lick_peristop_no_reward,1);
                    else
                        roe_allsuc_lick_stop_no_reward(alldays,allplanes,:) = NaN(1,1,313);
                    end
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    %%
                    
                    
                    
                    find_figure('velocity');clf
                    vr_speed=smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5));
                    plot(vr_speed)
                    hold on
                    vr_speed2=zeros(1,length(vr_speed)); vr_speed2(moving_middle)=1; vr_speed2(find(vr_speed2==0))=NaN;
                    plot(vr_speed.*vr_speed2,'r.')
                    vr_speed2=zeros(1,length(vr_speed));vr_speed2(stop)=1; vr_speed2(find(vr_speed2==0))=NaN;
                    plot(vr_speed.*vr_speed2,'k.')
                    rew_idx2=zeros(1,length(forwardvelALL)); rew_idx2(rew_idx)=1; rew_idx2(find(rew_idx2==0))=NaN;
                    plot(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)).*rew_idx2,'bo')
                    hold on
                    tempspeed = smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5));
                    stop_tmpts2=zeros(1,length(forwardvelALL)); stop_tmpts2(stop_success_tmpts)=1; stop_tmpts2(find(stop_tmpts2==0))=NaN;
                    plot(smoothdata(forwardvelALL,'gaussian',round(gauss_win*1.5)).*stop_tmpts2,'go')
                    scatter(nonrew_stop_success_tmpts,tempspeed(nonrew_stop_success_tmpts),'ys','filled')
                    scatter(rew_stop_success_tmpts,tempspeed(rew_stop_success_tmpts),'bs','filled')
                   scatter(lick_norew_stop_success_tmpts-1,tempspeed(lick_norew_stop_success_tmpts),'mv','filled')
                   scatter(  nolick_norew_stop_success_tmpts-1,tempspeed(  nolick_norew_stop_success_tmpts),'gv','filled')
                   
                plot(rescale(supraLick,-10,-5),'Color',[.7 .7 .7])
                    
                    
                    %%
                    %%moving rewarded
                    rew_mov_success_tmpts=[];
                    for jj=1:length(rew_stop_success_tmpts)
                        if ~isempty(find(rew_stop_success_tmpts(jj)-mov_success_tmpts<0,1,'first'))
                        rew_mov_success_tmpts(jj) =mov_success_tmpts(find(rew_stop_success_tmpts(jj)-mov_success_tmpts<0,1,'first'));
                        end
                    end
                    
                    %%%moving unrewarded
                    
                    nonrew_mov_success_tmpts = setxor(rew_mov_success_tmpts,mov_success_tmpts);
%                     scatter(nonrew_mov_success_tmpts,tempspeed(nonrew_mov_success_tmpts),'gv','filled')
%                     scatter(rew_mov_success_tmpts,tempspeed(rew_mov_success_tmpts),'cv','filled')
%                     
                    
                    
                    %  non rewarded motion
                    roidop_success_perimov_no_reward=[];    roe_success_perimov_no_reward=[];
                    
                    for stamps=1:length(nonrew_mov_success_tmpts)
                        currentidx =  find(timedFF>=utimedFF(nonrew_mov_success_tmpts(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_perimov_no_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perimov_no_reward(stamps,:)= forwardvelALL(nonrew_mov_success_tmpts(stamps)-pre_win_framesALL:nonrew_mov_success_tmpts(stamps)+post_win_framesALL);
                            
                        end
                    end
                    
                    % rewarded motions
                    roidop_success_perimov_reward=[]; roe_success_perimov_reward=[];
                    
                    for stamps=1:length(rew_mov_success_tmpts)
                        if sum(~isnan(rew_mov_success_tmpts))>=1
                            currentidx =  find(timedFF>=utimedFF(rew_mov_success_tmpts(stamps)),1);
                            roidop_success_perimov_reward(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perimov_reward(stamps,:)= forwardvelALL(rew_mov_success_tmpts(stamps)-pre_win_framesALL:rew_mov_success_tmpts(stamps)+post_win_framesALL);
                            
                            
                        else
                            roidop_success_perimov_reward=NaN(length(-pre_win_frames:post_win_frames));
                            roe_success_perimov_reward=NaN(length(-pre_win_frames:post_win_frames));
                        end
                    end
                    
                    
                    if ~isempty(roe_success_perimov_no_reward)
                        roe_allsuc_mov_no_reward(alldays,allplanes,:)=mean(roe_success_perimov_no_reward,1);
                    else
                        roe_allsuc_mov_no_reward(alldays,allplanes,:)=NaN(1,1,313);
                    end
                    if ~isempty(roe_success_perimov_reward)
                        roe_allsuc_mov_reward(alldays,allplanes,:)=mean(roe_success_perimov_reward,1);
                    else
                        roe_allsuc_mov_reward(alldays,allplanes,:) = NaN(1,1,313);
                    end
                    
                    
                    
                    
                    
                    
                    
                    %% SAVING SECTION
                    
                    
                    
                    
                    
                    
                    %save per day roi reward
                    roi_single_traces{roii} = roisingle_traces;
                    
                    roi_norm_single_traces{roii} = roinorm_single_traces;
                    roi_single_tracesCS{roii}= roisingle_tracesCS;
                    roi_norm_single_tracesCS{roii} = roinorm_single_tracesCS;
                    
                    %%%%
                    roi_single_tracesUS{roii}= roisingle_tracesUS;
                    roi_norm_single_tracesUS{roii}= roinorm_single_tracesUS;
                    
                    
                    
                    
                    
                    if exist('double_idx','var')&~isempty(double_idx)
                        roi_double_traces{roii} = roidouble_traces;
                        roi_norm_double_tracesCS{roii} = roinorm_double_tracesCS;
                        roi_double_tracesCS{roii} = roidouble_tracesCS;
                        roi_norm_double_tracesUS{roii} = roinorm_double_tracesCS;
                        roi_double_tracesUS{roii} = roidouble_tracesUS;
                        roi_norm_double_traces{roii} = roinorm_double_traces;
                    end
                    
                    if exist('urew_solenoid','var')
                        
                        roi_unrew_single_tracesCS{roii}= roiunrew_single_traces_CS';
                        roi_norm_unrew_single_tracesCS{roii}=  roinorm_unrew_single_tracesCS';
                    end
                    
                    if exist('rew_us_solenoid','var')%%%wihout CS
                        if length(find(rew_us_solenoid))>0
                            %%%%
                            roi_single_tracesUS_only_lick{roii}= roisingle_traces_us_only;
                            roi_norm_single_tracesUS_only_lick{roii}= roinorm_single_traces_us_only;
                            roi_rew_us_single_tracesUS_only{roii}= roirew_us_single_traces_US;
                            roi_norm_rew_us_single_tracesUS_only{roii}= roinorm_rew_us_single_tracesUS;
                            roi_doublerew_mus_single_tracesCS_only{roii}=  roidoublerew_mus_single_traces_CS;
                            roi_norm_doublerew_mus_single_tracesCS_only{roii}= roinorm_doublerew_mus_single_traces_CS;
                        end
                        
                    end
                    
                    
                    roi_nr_traces{roii} = roinr_traces;
                    roi_norm_nr_traces{roii} = roinorm_nr_traces;
                    roi_dop_success_peristop_no_reward{roii} = roidop_success_peristop_no_reward;
                    roi_dop_success_peristop_reward{roii} = roidop_success_peristop_reward;
                    roi_dop_success_perimov{roii} = roidop_success_perimov;
                    roi_dop_success_peristop{roii} = roidop_success_peristop;
                    roi_dop_success_perimov_no_reward{roii} = roidop_success_perimov_no_reward;
                    roi_dop_success_perimov_reward{roii} = roidop_success_perimov_reward;
                    roi_dop_success_peristop_lick_noreward{roii} = roidop_success_perimov_no_reward;
                    roi_dop_success_peristop_nolick_noreward{roii} = roidop_success_perimov_reward;
                    
                    
                    
                    
                    
                    %save allDAYS peri locomotion dopamine variables
                    planeroicount = planeroicount + 1;
                    planeroicount
                    alldays=alldays;
                    roi_dop_alldays_planes_success_mov{alldays,planeroicount}=roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2);
                    roi_dop_alldays_planes_success_stop{alldays,planeroicount}=roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2);
                    
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
                    
                    
                    roi_dop_allsuc_mov(alldays,planeroicount,:)=mean(roidop_success_perimov./mean(roidop_success_perimov(:,1:pre_win_frames),2),1);
                    roi_dop_allsuc_stop(alldays,planeroicount,:)=mean(roidop_success_peristop./mean(roidop_success_peristop(:,1:pre_win_frames),2),1);
                    
                    if ~isempty(roidop_success_peristop_no_reward)
                        roi_dop_allsuc_stop_no_reward(alldays,planeroicount,:)=mean(roidop_success_peristop_no_reward./mean(roidop_success_peristop_no_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_stop_no_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_perimov,2));
                    end
                    if ~isempty(roidop_success_peristop_reward)
                        roi_dop_allsuc_stop_reward(alldays,planeroicount,:)=mean(roidop_success_peristop_reward./mean(roidop_success_peristop_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_stop_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_perimov,2));
                    end
                    %%
                    
                     if ~isempty(roidop_success_perimov_no_reward)
                        roi_dop_alldays_planes_success_mov_no_reward{alldays,planeroicount}= roidop_success_perimov_no_reward./mean(roidop_success_perimov_no_reward(:,1:pre_win_frames),2);
                    else
                        roi_dop_alldays_planes_success_mov_no_reward{alldays,planeroicount}= NaN(1,size(roidop_success_perimov,2));
                    end
                    if ~isempty(roidop_success_perimov_reward)
                        roi_dop_alldays_planes_success_mov_reward{alldays,planeroicount}= roidop_success_perimov_reward./mean(roidop_success_perimov_reward(:,1:pre_win_frames),2);
                    else
                        roi_dop_alldays_planes_success_mov_reward{alldays,planeroicount} = NaN(1,size(roidop_success_perimov,2));
                    end
                    
                    
                     if ~isempty(roidop_success_perimov_no_reward)
                        roi_dop_allsuc_mov_no_reward(alldays,planeroicount,:)=mean(roidop_success_perimov_no_reward./mean(roidop_success_perimov_no_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_mov_no_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_perimov,2));
                    end
                    if ~isempty(roidop_success_perimov_reward)
                        roi_dop_allsuc_mov_reward(alldays,planeroicount,:)=mean(roidop_success_perimov_reward./mean(roidop_success_perimov_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_mov_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_perimov,2));
                    end
                    
                    
                    
                    %%%
                    if ~isempty(  roidop_lick_peristop_no_reward)
                        roi_dop_alldays_planes_success_lick_stop_no_reward{alldays,planeroicount}=   roidop_lick_peristop_no_reward./mean(  roidop_lick_peristop_no_reward(:,1:pre_win_frames),2);
                    else
                        roi_dop_alldays_planes_success_lick_stop_no_reward{alldays,planeroicount}= NaN(1,size(roidop_success_peristop_no_reward,2));
                    end
                    if ~isempty(roidop_nolick_peristop_no_reward)
                        roi_dop_alldays_planes_success_nolick_stop_no_reward{alldays,planeroicount}= roidop_nolick_peristop_no_reward./mean(roidop_nolick_peristop_no_reward(:,1:pre_win_frames),2);
                    else
                        roi_dop_alldays_planes_success_nolick_stop_no_reward{alldays,planeroicount} = NaN(1,size(roidop_success_peristop_no_reward,2));
                    end
                    
                    
                     if ~isempty( roidop_lick_peristop_no_reward)
                        roi_dop_allsuc_lick_stop_no_reward(alldays,planeroicount,:)=mean( roidop_lick_peristop_no_reward./mean( roidop_lick_peristop_no_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_lick_stop_no_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_peristop_no_reward,2));
                    end
                    if ~isempty(roidop_nolick_peristop_no_reward)
                        roi_dop_allsuc_nolick_stop_no_reward(alldays,planeroicount,:)=mean(roidop_nolick_peristop_no_reward./mean(roidop_nolick_peristop_no_reward(:,1:pre_win_frames),2),1);
                    else
                        roi_dop_allsuc_nolick_stop_no_reward(alldays,planeroicount,:) = NaN(1,size(roidop_success_peristop_no_reward,2));
                    end
                    
                    
                    
                    
                    
                    
                    
                    %%
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    %%%%% perireward ALLDAYS
                    roi_dop_alldays_planes_perireward_0{alldays,planeroicount}=roisingle_traces;
                    
                    roi_dop_alldays_planes_perireward{alldays,planeroicount}=roinorm_single_traces;
                    
                    roi_dop_allsuc_perireward(alldays,planeroicount,:)=mean(roinorm_single_traces,2);
                    
                    roi_dop_allsuc_perireward_se(alldays,planeroicount,:)=std(roinorm_single_traces,[],2)./sqrt(size(roinorm_single_traces,2));
                    
                    roi_dop_alldays_planes_periCS{alldays,planeroicount} = roinorm_single_tracesCS;
                    
                    roi_dop_allsuc_perirewardCS(alldays,planeroicount,:) = mean(roinorm_single_tracesCS,2);
                    
                    roi_roe_alldays_planes_periCS{alldays,planeroicount} = roisingle_traces_roesmthCS;
                    
                    roi_roe_allsuc_perirewardCS(alldays,planeroicount,:) = mean(roisingle_traces_roesmthCS,2);
                    
                    roi_dop_alldays_planes_periUS{alldays,planeroicount} = roinorm_single_tracesUS;
                    
                    roi_dop_allsuc_perirewardUS(alldays,planeroicount,:) = mean(roinorm_single_tracesUS,2);
                    
                    roi_roe_alldays_planes_periUS{alldays,planeroicount} = roisingle_traces_roesmthUS;
                    
                    roi_roe_allsuc_perirewardUS(alldays,planeroicount,:) = mean(roisingle_traces_roesmthUS,2);
                    
                    
                    
                    
                    
                    
                    
                    roi_dop_alldays_planes_perireward_double_0{alldays,planeroicount} = roidouble_traces;
                    
                    roi_dop_alldays_planes_perireward_double{alldays,planeroicount} = roinorm_double_traces;
                    
                    roi_dop_allsuc_perireward_double(alldays,planeroicount,:) = mean(roinorm_double_traces,2);
                    
                    roi_dop_allsuc_perireward_double_se(alldays,planeroicount,:)=std(roinorm_double_traces,[],2)./sqrt(size(roinorm_double_traces,2));
                    
                    if exist('double_idx','var')&~isempty(double_idx)
                        
                        roi_dop_alldays_planes_peridoubleCS{alldays,planeroicount} = roinorm_double_tracesCS;
                        
                        roi_dop_allsuc_perireward_doubleCS(alldays,planeroicount,:) = mean(roinorm_double_tracesCS,2);
                        
                        roi_roe_alldays_planes_peridoubleCS{alldays,planeroicount} = roidouble_traces_roesmthCS;
                        
                        roi_roe_allsuc_perireward_doubleCS(alldays,planeroicount,:) = mean(roidouble_traces_roesmthCS,2);
                        
                        roi_dop_alldays_planes_peridoubleUS{alldays,planeroicount} = roinorm_double_tracesUS;
                        
                        roi_dop_allsuc_perireward_doubleUS(alldays,planeroicount,:) = mean(roinorm_double_tracesUS,2);
                        
                        roi_roe_alldays_planes_peridoubleUS{alldays,planeroicount} = roidouble_traces_roesmthUS;
                        
                        roi_roe_allsuc_perireward_doubleUS(alldays,planeroicount,:) = mean(roidouble_traces_roesmthUS,2);
                    end
                    
                    %%%%
                    if exist('urew_solenoid','var')
                        roi_dop_alldays_planes_unreward_single_0{alldays,planeroicount} = roi_unrew_single_tracesCS;
                        
                        roi_dop_alldays_planes_unreward_single{alldays,planeroicount} = roinorm_unrew_single_tracesCS';
                        
                        roi_dop_allsuc_unreward_single(alldays,planeroicount,:) = mean( roinorm_unrew_single_tracesCS,2);
                        
                        roi_dop_allsuc_unreward_single_se(alldays,planeroicount,:)=std( roinorm_unrew_single_tracesCS,[],2)./sqrt(size( roinorm_unrew_single_tracesCS,2));
                        
                        roi_roe_alldays_planes_unrewardCS{alldays,planeroicount} =  roiunrew_single_traces_roesmthCS';
                        
                        roi_roe_allsuc_perireward_unrewardCS(alldays,planeroicount,:) = mean(  roiunrew_single_traces_roesmthCS,2)';
                    end
                    
                    if exist('rew_us_solenoid','var')
                        if length(find(rew_us_solenoid))>0
                            roi_dop_alldays_planes_rewusonly_single_0{alldays,planeroicount} = roi_rew_us_single_tracesUS_only;
                            
                            roi_dop_alldays_planes_rewusonly_single{alldays,planeroicount} = roinorm_rew_us_single_tracesUS';
                            
                            roi_dop_allsuc_rewusonly_single(alldays,planeroicount,:) = mean(  roinorm_rew_us_single_tracesUS,2);
                            
                            roi_dop_allsuc_rewusonly_single_se(alldays,planeroicount,:)=std(   roinorm_rew_us_single_tracesUS,[],2)./sqrt(size(  roi_norm_rew_us_single_tracesUS_only,2));
                            
                            roi_roe_alldays_planes_rewusonly{alldays,planeroicount} =   roirew_us_single_traces_roesmthUS';
                            
                            roi_roe_allsuc_rewusonly(alldays,planeroicount,:) = mean(  roirew_us_single_traces_roesmthUS,2)';
                            
                            
                            
                            
                            roi_dop_alldays_planes_perireward_usonly_0{alldays,planeroicount}=roisingle_traces_us_only;
                            
                            roi_dop_alldays_planes_perireward_usonly{alldays,planeroicount}= roinorm_single_traces_us_only;
                            
                            roi_dop_allsuc_perireward_usonly(alldays,planeroicount,:)=mean( roinorm_single_traces_us_only,2);
                            
                            roi_dop_allsuc_perireward_usonly_se(alldays,planeroicount,:)=std( roinorm_single_traces_us_only,[],2)./sqrt(size( roinorm_single_traces_us_only,2));
                            
                            roi_roe_alldays_planes_rewusonly_lick{alldays,planeroicount} =   roisingle_traces_us_only_roesmth';
                            
                            roi_roe_allsuc_rewusonly_lick(alldays,planeroicount,:) = mean(  roisingle_traces_us_only_roesmth,2)';
                            
                            
                            
                            roi_dop_alldays_planes_perirewarddoublemUS_CS_0{alldays,planeroicount}=roidoublerew_mus_single_traces_CS;
                            
                            roi_dop_alldays_planes_perirewarddoublemUS_CS{alldays,planeroicount}= roinorm_doublerew_mus_single_traces_CS;
                            
                            roi_dop_allsuc_perirewarddoublemUS_CS(alldays,planeroicount,:)=mean( roinorm_doublerew_mus_single_traces_CS,2);
                            
                            roi_dop_allsuc_perirewarddoublemUS_CS_se(alldays,planeroicount,:)=std( roinorm_doublerew_mus_single_traces_CS,[],2)./sqrt(size( roinorm_single_traces_us_only,2));
                            
                            roi_roe_alldays_planes_perirewarddoublemUS_CS{alldays,planeroicount} = roidoublerew_mus_single_traces_roesmthCS';
                            
                            roi_roe_allsuc_perirewarddoublemUS_CS(alldays,planeroicount,:) = mean(roidoublerew_mus_single_traces_roesmthCS,2)';
                            
                            
                        end
                        
                        
                    end
                    
                    
                    
                    
                    
                    
                    
                    
                    %%%
                    
                    
                    
                    
                    
                    roi_dop_alldays_planes_perinrlicks_0{alldays,planeroicount}=roinr_traces;
                    
                    roi_dop_alldays_planes_perinrlicks{alldays,planeroicount}=roinorm_nr_traces;
                    
                    
                    
                    
                end
                
            end
            
            % SAVING ALL ROI VARIABLES FOR ONE DAY
            save('params','roi_single_traces','roi_norm_single_traces','roi_nr_traces','roi_norm_nr_traces','-append')
            save('params','roi_dop_success_peristop_no_reward','roi_dop_success_peristop_reward','roi_dop_success_perimov','roi_dop_success_peristop','-append')
            if exist('double_idx','var')&~isempty(double_idx)
                save('params','roi_double_traces','roi_norm_double_traces','roi_norm_double_tracesCS','roi_double_tracesCS','roi_norm_double_tracesUS','roi_double_tracesUS','-append')
            end
            save('params','roi_single_tracesCS','roi_norm_single_tracesCS','roi_single_tracesUS','roi_single_tracesUS','-append')
            if exist('urew_solenoid','var')
                save('params','roi_unrew_single_tracesCS','roi_norm_unrew_single_tracesCS','-append')
            end
            if exist('rew_us_solenoid','var')
                if length(find(rew_us_solenoid))>0
                    save('params','roi_rew_us_single_tracesUS_only','roi_norm_rew_us_single_tracesUS_only','-append')
                end
            end
            
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


