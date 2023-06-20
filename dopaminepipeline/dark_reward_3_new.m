clear all
close all
mouse_id=149%158;
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
%e148 alldays=[1:12] e149=[1:13]; e156=[3:7 9:16 18] e157 =[3 5:6 8:19]
%e158=[3:12 14:19]

for alldays=[1:13]%[3:12 14:19]%[3:12 13:19]%[3 5:1]%[5:12 14]%[5:12 14]%1:5%26:30%1:33%31:32%30%27:30%21%[1:21]%[1:2 4:5 12:22]%[8:21]%[2:4 6:11]%[1:2 4:5 12:20]%%[1:21]%%
    clearvars -except alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr ...
        mov_corr_success stop_corr_success mov_corr_prob stop_corr_prob mov_corr_fail stop_corr_fail...
        c  alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr dop_suc_movint dop_suc_stopint roe_suc_movint roe_suc_stopint ...
        dop_allsuc_mov dop_allsuc_stop roe_allsuc_mov roe_allsuc_stop ...
        dop_allfail_mov dop_allfail_stop roe_allfail_mov roe_allfail_stop...
        dop_alldays_planes_success_mov dop_alldays_planes_fail_mov dop_alldays_planes_success_stop dop_alldays_planes_fail_stop...
        roe_alldays_planes_success_mov roe_alldays_planes_fail_mov roe_alldays_planes_success_stop roe_alldays_planes_fail_stop...
        subp days_check cnt dop_alldays_planes_perireward roe_alldays_planes_perireward dop_allsuc_perireward roe_allsuc_perireward...
          dop_alldays_planes_perireward_0   roe_alldays_planes_perireward_0
    
    cnt=cnt+1;
    %     close all
    Day=alldays;
    for allplanes=1:4
        plane=allplanes;
        %%%% test
        
        
        %         %%% dop e156 e157
        
        %         pr_dir=strcat('E:\E',num2str(mouse_id),'\HRZ\Day_',num2str(Day));
        %         if exist( pr_dir, 'dir')
        %         else
        %             pr_dir=strcat('D:\MM\E',num2str(mouse_id),'\HRZ\Day_',num2str(Day));
        %             if exist( pr_dir, 'dir')
        %             else
        %                 pr_dir=strcat('C:\MM\E',num2str(mouse_id),'\HRZ\Day_',num2str(Day));
        %             end
        %
        %         end
        
        %         %%%%% for HRZ
        %         pr_dir=strcat('E:\E',num2str(mouse_id),'\HRZ\Day_',num2str(Day));
        %         if exist( pr_dir, 'dir')
        %         else
        %             pr_dir=strcat('F:\MM\HRZ\E',num2str(mouse_id),'\Day_',num2str(Day));
        %             if exist( pr_dir, 'dir')
        %             else
        %                 pr_dir=strcat('C:\MM\E',num2str(mouse_id),'\HRZ\Day_',num2str(Day));
        %             end
        %
        %         end
        
        %%%%% for GFP E158 HRZ
        %         pr_dir=strcat('H:\E',num2str(mouse_id),'forMunne\Day_',num2str(Day));
        
        
        
        %%%% for dark rewards
        %          pr_dir=strcat('D:\E',num2str(mouse_id),'\D',num2str(Day));
%         pr_dir=strcat('D:\E',num2str(mouse_id),'\Dark_reward\Day_',num2str(Day));
        %%% for dark rewards
        pr_dir1=strcat('G:\dark_reward\E',num2str(mouse_id),'\Day_',num2str(alldays),'\suite2p');
        pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','')
        
        
        %
        %%% egfp e158
        %         pr_dir=strcat('F:\E',num2str(mouse_id),'\D',num2str(alldays));
        
        if exist( pr_dir, 'dir')
            %%%egfp e158
            %             cd(strcat('F:\E',num2str(mouse_id),'\D',num2str(alldays)))%%%e158
            
            %             %%% dop e156 e157
            cd (pr_dir)
            list=dir('*.mat');
           
            if isempty(strfind(list(1).name,'mean'))
                load(strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'.mat'))
                strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'.mat')
            else
                load(strcat(list(1).name(1:end-16),'_mean_plane',num2str(plane),'.mat'))
                strcat(list(1).name(1:end-16),'_mean_plane',num2str(plane),'.mat')
            end
            
            %             strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'.mat')
            %load VR file
%             for jj=1:length(list)
%                 if  isempty(strfind(list(jj).name,'('))~=1
%                     load(list(jj).name)
%                 end
%             end
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
            
            
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time);
            pre_win_frames=round(pre_win/frame_time);
            exclusion_win_frames=round(exclusion_win/frame_time);
            
%             [B,~,bin_indx] = histcounts(1:numframes,length(base_mean));
%             rew_binned = accumarray(bin_indx(:),rewards,[],@mean);
            mean_base_mean=mean(base_mean);
%              mean_base = prctile(base_mean,8);
            
            norm_base_mean=base_mean/mean_base_mean;
            
%             speed_binned=forwardvel;
            roe_binned = speed_binned;
            reward_binned=rewards;
%             lick_binned=licks;
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
            dop_smth=smoothdata(norm_base_mean,'gaussian',gauss_win);
            
            %%%% Just checking
%             figure;  plot(1/1000:1/1000:length(urewards)/1000,urewards)
%             hold on
%             plot(4/31.25:4/31.25:12500*4/31.25,speed_binned/100); hold on; plot(1/1000:1/1000:length(uROE)/1000,uROE)
%             
            
            %%%%%%%%%%%%%%%%%%%calculate single traces
            %%%%%%rewarded licks
            
            R = bwlabel(rew_binned>rew_thresh);%label rewards, ascending
            rew_idx=find(R);%get indexes of all rewards
            rew_idx_diff=diff(rew_idx);%difference in reward index from last
            short=rew_idx_diff<num_rew_win_frames;%logical for rewards that happen less than x frames from last reward. 0 = single rew.
            
            %single rewards
            multi_rew_expand=[];
            multi_rew_expand=bwlabel(short);%single rewards are 0
            for i=1:length(multi_rew_expand)
                multi_rew_expand(find(multi_rew_expand==i,1,'last')+1)=i;%need to expand index of multi reward by 1 to properly match rew_ind
            end
            
            if length(multi_rew_expand) < length(rew_idx)
                multi_rew_expand(end+1)=0;%need to add extra on end to match index. Above for loop does this if last rew is multi reward. this does for single last.
            end
            
            single_rew=find(multi_rew_expand==0);
            single_idx=[];single_lick_idx=[];
            for i=1:length(single_rew)
                %single_idx(i)=rew_idx(i); %orig but doesn't eliminate doubles
                single_idx(i)=rew_idx(single_rew(i));
                if single_idx(i)+rew_lick_win_frames < length(supraLick)%if window to search for lick after rew is past length of supraLick, doesn't update single_lick_idx, but single_idx is
                    %                     single_lick_idx(i) = (find(supraLick(single_idx(i):single_idx(i)+rew_lick_win_frames),1,'first'))+single_idx(i)-1;
                    %                     %looks for first lick after rew with window =exclusion_win_frames
                    %                     %however first lick can be much further in naive animals
                    %                 end
                    
                    lick_exist=(find(supraLick(single_idx(i):single_idx(i)+rew_lick_win_frames),1,'first'))+single_idx(i)-1;
                    if isempty(lick_exist)~=1
                        
                        single_lick_idx(i) = (find(supraLick(single_idx(i):single_idx(i)+rew_lick_win_frames),1,'first'))+single_idx(i)-1;
                        %looks for first lick after rew with window =exclusion_win_frames
                        %however first lick can be much further in naive animals
                    else
                        warning('no lick after reward was delivered!!!');
                    end
                end
            end
            if single_lick_idx(1) - pre_win_frames <0%remove events too early
                single_lick_idx(1)=[];
            end
            if single_lick_idx(end) + post_win_frames > length(base_mean)%remove events too late
                single_lick_idx(end)=[];
            end
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
            
            
            
            
            
%             figure; plot(4/31.25:4/31.25:12500*4/31.25,speed_binned/100); hold on; plot(1/1000:1/1000:length(uROE)/1000,uROE)
%             hold on
%             plot(4/31.25:4/31.25:12500*4/31.25,rewards); hold on; plot(1/1000:1/1000:length(urewards)/1000,urewards)
%             pause
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%5
            
            
            
            %norm_single_traces_roesmth=single_traces_roesmth./mean(single_traces_roesmth(1:pre_win_frames,:));
            if exist('single_traces','var')
                find_figure(strcat(mouse_id,'_perireward','_plane',num2str(allplanes)))
                %                 norm_single_traces=single_traces./mean(single_traces(1:pre_win_frames,:));
                subplot(5,4,cnt),
                xlabel('seconds from first reward lick')
                ylabel('dF/F')
                hold on
                %     plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces,'Color',[.8 .8 .8]);
                subplot(5,4,cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces);
                hold on
                subplot(5,4,cnt),plot(mean(norm_single_traces,2),'k','LineWidth',2);
                legend(['n = ',num2str(size(norm_single_traces,2))])
                
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(norm_single_traces_roesmth,0.99,1))
                %                 set(gca,'ylim',[0.98 1.05])
                
%                 pause
                
                
                %%%%moving and stop activity
                
                
                find_figure('dop_forwardvel_control_binned_activity');clf; 
                subplot(4,1,allplanes),plot(norm_base_mean)
                hold on
                forwardvel=smoothdata(speed_binned,'gaussian',gauss_win)';
                plot(rescale(forwardvel,0.999,1.05),'k')
                
                
                
                [moving_middle,stopping_middle]=mov_stop_tmstmp_2(forwardvel,forwardvel);
                mov_success_tmpts=[];
                ids_loc=[1 find(diff(moving_middle)>1) length(moving_middle)]; c_tmstmp=0;
                for movwin=1:length(ids_loc)-1
                    tmstmps=moving_middle(ids_loc(movwin)+1:ids_loc(movwin+1));
                    if size(tmstmps,2)>round(2/frame_time)
                        c_tmstmp=c_tmstmp+1;
                        mov_success_tmpts(c_tmstmp,1)=tmstmps(1);
                        mov_success_tmpts(c_tmstmp,2)=tmstmps(end);
                    end
                end
                
                
                %                     if mov_success_tmpts(1) - pre_win_frames <0%remove events too early
                %                         mov_success_tmpts(1,:)=[];
                %                     end
                
                idx_rm=(mov_success_tmpts- pre_win_frames)<0;
                rm_idx=find(idx_rm(:,1)==1)
                mov_success_tmpts(rm_idx,:)=[];
                
                %         if mov_success_tmpts(end) + post_win_frames > length(dop_success)%remove events too late
                %             mov_success_tmpts(end,:)=[];
                %         end
                idx_rm=(mov_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm(:,1)==1)
                mov_success_tmpts(rm_idx,:)=[];
                allmov_success=NaN(1,size(norm_base_mean,2));
                dop_success_perimov=[]; roe_success_perimov=[];
                for stamps=1:size(mov_success_tmpts,1)
                    dop_success_perimov(stamps,:)= norm_base_mean(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
                    roe_success_perimov(stamps,:)= forwardvel(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
                    
                    allmov_success(mov_success_tmpts(stamps,1):mov_success_tmpts(stamps,2))=1;
                end
                
                
                %%%%stopping success trials
                %
                stop_success_tmpts=[];
                ids_loc=[1 find(diff(stopping_middle)>1) length(stopping_middle)]; c_tmstmp=0;
                for movwin=1:length(ids_loc)-1
                    tmstmps=stopping_middle(ids_loc(movwin)+1:ids_loc(movwin+1));
                    if size(tmstmps,2)>round(2/frame_time)
                        c_tmstmp=c_tmstmp+1;
                        stop_success_tmpts(c_tmstmp,1)=tmstmps(1);
                        stop_success_tmpts(c_tmstmp,2)=tmstmps(end);
                    end
                end
                
                
                %                     if stop_success_tmpts(1) - pre_win_frames <0%remove events too early
                %                         stop_success_tmpts(1,:)=[];
                %                     end
                
                idx_rm= stop_success_tmpts(:,1)- pre_win_frames <0;
                rm_idx=find(idx_rm(:,1)==1)
                stop_success_tmpts(rm_idx,:)=[];
                
                
                %         if stop_success_tmpts(end) + post_win_frames > length(dop_success)%remove events too late
                %             stop_success_tmpts(end,:)=[];
                %         end
                idx_rm=(stop_success_tmpts+post_win_frames)>length(norm_base_mean);
                rm_idx=find(idx_rm(:,1)==1)
                stop_success_tmpts(rm_idx,:)=[];
                
                dop_success_peristop=[]; roe_success_peristop=[];
                allstop_success=NaN(1,size(norm_base_mean,2));
                for stamps=1:size(stop_success_tmpts,1)
                    dop_success_peristop(stamps,:)= norm_base_mean(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
                    roe_success_peristop(stamps,:)= forwardvel(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
                    %%% for plotting
                    allstop_success(stop_success_tmpts(stamps,1):stop_success_tmpts(stamps,2))=1;
                end
                
                
                find_figure('per_mov_stop'); clf
                subplot(4,3,1),   plot(forwardvel); hold on;  plot(allstop_success,'k','Linewidth',2)
                hold on;  plot(allmov_success,'g','Linewidth',2)
                %                 plot(forwardvel.*allmov_success,'g.');
                title('successful trials')
                %         plot(ybinned_success.*allmov_success','g.')size
                %         chk=zeros(1,length(roe_success));
                %         chk(1,stop_success_tmpts(:,1))=1;
                %         chk(1,stop_success_tmpts(:,2))=2;
                %         plot(rescale(chk,0,50))
                subplot(4,6,7), plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov); hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(dop_success_perimov,1),'k','LineWidth',2);
                legend(['n = ',num2str(size(dop_success_perimov,1))]); plot([0 0],[0.995 1.005],'r','LineWidth',2)
                xlabel('seconds from moving initiation'); ylabel('dF/F')
                subplot(4,6,13), plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,roe_success_perimov); hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(roe_success_perimov,1),'k','LineWidth',2);
                legend(['n = ',num2str(size(roe_success_perimov,1))]); plot([0 0],[0.998 1.003],'r','LineWidth',2)
                xlabel('seconds from moving initiation');  ylabel('velocity')
                subplot(4,6,8), plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop); hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(dop_success_peristop,1),'k','LineWidth',2);
                legend(['n = ',num2str(size(dop_success_peristop,1))]); plot([0 0],[0.995 1.005],'r','LineWidth',2)
                xlabel('seconds from moving initiation'); ylabel('dF/F')
                subplot(4,6,14), plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,roe_success_peristop); hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(roe_success_peristop,1),'k','LineWidth',2);
                legend(['n = ',num2str(size(roe_success_peristop,1))]); plot([0 0],[0.998 1.003],'r','LineWidth',2)
                xlabel('seconds from moving initiation');  ylabel('velocity')
                
                pre_success_mov=dop_success_perimov(:,24:39);pst_success_mov=dop_success_perimov(:,40:55);%%% 2sec before and after compare
                [h_sm(alldays),p_sm(alldays)]=ttest(mean(pre_success_mov,2),mean(pst_success_mov,2));
                vals=[mean(pre_success_mov,2); mean(pst_success_mov,2)];
                ids=[ones(1,size(mean(pre_success_mov,2),1)) 2*ones(1,size(mean(pst_success_mov,2),1))]';
                subplot(4,6,19), plot(ids,vals,'ko'); set(gca,'xlim',([0.5 2.5])); hold on; plot([mean(mean(pre_success_mov)),mean(mean(pst_success_mov))],'k--')
                legend(['h = ',num2str(h_sm(alldays))]);
                set(gca,'xTick',([1 2]))
                set(gca,'XTickLabel',{'pre','post'})
                
                pre_success_stop=dop_success_peristop(:,24:39);pst_success_stop=dop_success_peristop(:,40:55);%%% 2sec before and after compare
                [h_ss(alldays),p_ss(alldays)]=ttest(mean(pre_success_stop,2),mean(pst_success_stop,2));
                vals=[mean(pre_success_stop,2); mean(pst_success_stop,2)];
                ids=[ones(1,size(mean(pre_success_stop,2),1)) 2*ones(1,size(mean(pst_success_stop,2),1))]';
                subplot(4,6,20), plot(ids,vals,'ko'); set(gca,'xlim',([0.5 2.5])); hold on; plot([mean(mean(pre_success_stop)),mean(mean(pst_success_stop))],'k--')
                legend(['h = ',num2str(h_ss(alldays))]); set(gca,'xTick',([1 2]))
                set(gca,'XTickLabel',{'pre','post'})
                
                alldays=alldays;
                dop_alldays_planes_success_mov{alldays,allplanes}=dop_success_perimov;
                dop_alldays_planes_success_stop{alldays,allplanes}=dop_success_peristop;
                
                
                
                
                roe_alldays_planes_success_mov{alldays,allplanes}=roe_success_perimov;
                roe_alldays_planes_success_stop{alldays,allplanes}=roe_success_peristop;
                
                dop_allsuc_mov(alldays,allplanes,:)=mean(dop_success_perimov);
                dop_allsuc_stop(alldays,allplanes,:)=mean(dop_success_peristop);
                
                
                roe_allsuc_mov(alldays,allplanes,:)=mean(roe_success_perimov);
                roe_allsuc_stop(alldays,allplanes,:)=mean(roe_success_peristop);
                
                %%%%% perireward
                dop_alldays_planes_perireward_0{alldays,allplanes}=single_traces;
                roe_alldays_planes_perireward_0{alldays,allplanes}=single_traces_roesmth;
                
                dop_alldays_planes_perireward{alldays,allplanes}=norm_single_traces;
                roe_alldays_planes_perireward{alldays,allplanes}=norm_single_traces_roesmth;
                
                dop_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces,2);
                roe_allsuc_perireward(alldays,allplanes,:)=mean(norm_single_traces_roesmth,2);
                
                dop_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces,[],2)./sqrt(size(norm_single_traces,2));
                roe_allsuc_perireward_se(alldays,allplanes,:)=std(norm_single_traces_roesmth,[],2)./sqrt(size(norm_single_traces_roesmth,2));
                
                
                
                
                %%%%% moving
                find_figure(strcat(mouse_id,'_perilocomotion moving','_plane',num2str(allplanes)));
                %                     norm_single_traces=single_traces./mean(single_traces(1:pre_win_frames,:));
                subplot(5,4,cnt),%title(strcat(paradigm{c},'_',list_fold{alldays}));
                %                     xlabel('seconds from first reward lick')
                ylabel('dF/F')
                
                
                
                hold on
                %     plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces,'Color',[.8 .8 .8]);
                subplot(5,4,cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov);
                hold on
                subplot(5,4,cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(dop_success_perimov,1),'k','LineWidth',2);
                
                legend(['n = ',num2str(size(dop_success_perimov,1))])
                hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(mean(roe_success_perimov,1),0.985,0.99),'k--','LineWidth',2)
                
                text(-5, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
                text(5, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
                
                
                
                
                
                
                find_figure(strcat(mouse_id,'_perilocomotion stop','_plane',num2str(allplanes)));
                %                     norm_single_traces=single_traces./mean(single_traces(1:pre_win_frames,:));
                subplot(5,4,cnt)
                %                     xlabel('seconds from first reward lick')
                ylabel('dF/F')
                
                hold on
                %     plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces,'Color',[.8 .8 .8]);
                subplot(5,4,cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop);
                hold on
                subplot(5,4,cnt),plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,mean(dop_success_peristop,1),'k','LineWidth',2);
                legend(['n = ',num2str(size(dop_success_peristop,1))])
                hold on
                plot(frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,rescale(mean(roe_success_peristop,1),0.985,0.99),'k--','LineWidth',2)
                text(-5, min(ylim), 'Moving', 'Horiz','left', 'Vert','bottom')
                text(5, min(ylim), 'Stopped', 'Horiz','right', 'Vert','bottom')
                
                
                
            else
                %                     subplot(3,5,c),legend(['n = ',num2str(0)]);
                %
                %                     title(strcat(paradigm{c},'_',list_fold{alldays}));
                
            end
            
        end
        
    end
%         pause
%         cats
end

%%
% files=[3:6 8  10:12 14:19]%[5:12 14]%[5:12 14]%[5:14]%[8:13 15:18]
files=[1:12]
find_figure('dop_days_loc_planes');clf
find_figure('fail vs success moving-triggered');clf
find_figure('fail vs success stop-triggered'); clf
find_figure('success moving-triggered activity over days');clf
for jj=1:4
    find_figure('dop_days_loc_planes')
    ax(jj)=subplot(6,4,1),imagesc(squeeze(roe_allsuc_mov(files,jj,:))); hold on
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
    
    
    text(10, min(ylim), 'Stopped', 'Horiz','left', 'Vert','bottom')
    text(62, min(ylim), 'Moving', 'Horiz','right', 'Vert','bottom')
    ylabel('Days')
    
    ax(jj)=subplot(6,4,(jj+1-1)*4+1),imagesc(squeeze(dop_allsuc_mov(files,jj,:))); hold on
%     caxis([0.995 1.005]) 
colorbar
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet);
    title(strcat('success_Plane',num2str(jj),'__Start Triggered'));set(gca,'xtick',[])
    
    ylabel('Days')
    
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
    yax=mean(squeeze(dop_allsuc_mov(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_mov(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_mov(files,jj,:)),1))
    subplot(6,4,(6-1)*4+1),hold on, shadedErrorBar(xax,yax,se_yax,color(jj));
    plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    clear xlabel; xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.01])
    plot(xax,rescale(mean(squeeze(roe_allsuc_mov(files,jj,:)),1),0.995,1),'k')
    
    find_figure('fail vs success moving-triggered');
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_perimov;
    yax=mean(squeeze(dop_allsuc_mov(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_mov(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_mov(files,jj,:)),1))
    subplot(6,4,jj),hold on, shadedErrorBar(xax,yax,se_yax,color(jj));
    plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.005])
    plot(xax,rescale(mean(squeeze(roe_allsuc_mov(files,jj,:)),1),0.995,1),'k')
end


%%%%%%%%% stopping triggered successful

find_figure('dop_days_loc_planes');

for jj=1:4
    find_figure('dop_days_loc_planes')
    ax(jj)=subplot(6,4,3),imagesc(squeeze(roe_allsuc_stop(files,jj,:))); hold on
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray); set(gca,'xtick',[])
    
    text(10, min(ylim), 'Moving', 'Horiz','left', 'Vert','bottom')
    text(62, min(ylim), 'Stopped', 'Horiz','right', 'Vert','bottom');
    
    ylabel('Days')
    
    ax(jj)=subplot(6,4,(jj+1-1)*4+3),imagesc(squeeze(dop_allsuc_stop(files,jj,:))); hold on
%     caxis([0.995 1.005])
colorbar
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet); set(gca,'xtick',[])
    title(strcat('success_Plane',num2str(jj),'__Stop Triggered'));
    
    ylabel('Days')
    
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    yax=mean(squeeze(dop_allsuc_stop(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_stop(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_stop(files,jj,:)),1))
    subplot(6,4,(6-1)*4+3),hold on, shadedErrorBar(xax,yax,se_yax,color(jj)); plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    %     hp=shadedErrorBar(xax,yax,se_yax,color(jj))
    %     hp=plot(yax,color(jj))
    %     legend(hp,strcat('plane',num2str(jj)));hold on
    xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.01])
    plot(xax,rescale(mean(squeeze(roe_allsuc_stop(files,jj,:)),1),0.99,1),'k')
    
    find_figure('fail vs success stop-triggered');
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,dop_success_peristop;
    yax=mean(squeeze(dop_allsuc_stop(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_stop(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_stop(files,jj,:)),1))
    subplot(6,4,jj),hold on, shadedErrorBar(xax,yax,se_yax,color(jj)); plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    %     hp=shadedErrorBar(xax,yax,se_yax,color(jj))
    %     hp=plot(yax,color(jj))
    %     legend(hp,strcat('plane',num2str(jj)));hold on
    xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.01])
    plot(xax,rescale(mean(squeeze(roe_allsuc_stop(files,jj,:)),1),0.99,1),'k')
    title(strcat('success__allPlane__Stop Triggered'));
end

%%%%%
%%%%%%%%%%
clear dop_suc_movt_pre dop_suc_movt_pre  dop_fail_movt_pre  dop_fail_movt_pst dop_suc_stopt_pre dop_suc_stopt_pst...
    dop_fail_stopt_pre dop_fail_stopt_pst dop_suc_movt_pst
for planes=1:4
    vals=dop_alldays_planes_success_mov(files,planes);
    
    dop_suc_movt_pre(:,planes)=cellfun(@(x) mean(x(:,1:39),2),vals,'UniformOutput',false);
    dop_suc_movt_pst(:,planes)=cellfun(@(x) mean(x(:,40:79),2),vals,'UniformOutput',false);
    
    vals=dop_alldays_planes_success_stop(files,planes);
    
    dop_suc_stopt_pre(:,planes)=cellfun(@(x) mean(x(:,1:39),2),vals,'UniformOutput',false);
    dop_suc_stopt_pst(:,planes)=cellfun(@(x) mean(x(:,40:79),2),vals,'UniformOutput',false);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55555
cats={ 'dop_suc_movt_pst'   'dop_suc_stopt_pst' };
cats4={ 'dop_suc_movt_pre' 'dop_suc_stopt_pre' };
cats2={ 'dop_allsuc_mov'  'dop_allsuc_stop' };
cats3={ 'roe_allsuc_mov'   'roe_allsuc_stop' };
for kk=1:2
    find_figure(strcat('activity-over-days',cats{kk}));clf
    find_figure(strcat('meanactivity-over-days',cats{kk}));clf
    for planes=1:4
        data=[];
        eval(sprintf('data=%s(:,%i)',cats{kk},planes))%%%% pst dop
        eval(sprintf('data3=%s(:,%i)',cats4{kk},planes))%%%pre dop
        xvals=1:length(data);
        yvals=data;
        xcord=[];ycord=[];
        for xidx = 1:length(xvals)
            % spread data over region from x-0.2 to x+0.2
            num_yvals = length(yvals{xidx});
            jitter_delta = 0.2/num_yvals;
            x = repmat(xvals(xidx),[num_yvals, 1]);
            jitter = ((0:(num_yvals-1))*jitter_delta - 0.2);
            x = x + jitter';
            y = yvals{xidx};
            %         err = errvals{xidx};
            %             find_figure(strcat('plane_success_fail_prob_epoch',add_dum_fig)); subplot(4,1,allplanes)
            %             hold on
            %             plot(x,y,'r.');
            %             hold on
            xcord=[xcord ;x];
            ycord=[ycord ;y];
        end
        find_figure(strcat('activity-over-days',cats{kk}));  subplot(2,4,planes+4); hold on
        plot(xcord,ycord,'r.');
        set(gca,'ylim',[0.995 1.005])
        hold on
        plot(cellfun(@(x) mean(x),data),'k','Linewidth',2)
        lsline
        %          subplot(3,4,(planes-1)*1+9), plot(mean(yax2(:,40:79),2))
        
        eval(sprintf('data1=%s',cats2{kk}))%%%20 days dop
        eval(sprintf('data2=%s',cats3{kk}))%%% 20 days roe
        
        
        
        
        yax2=squeeze(data1(files,planes,:));
        subplot(2,4,planes),imagesc(yax2),%stackedplot(xax,yax2)
        %         set(gca,'XTick',[1:118],'XTickLabel',xax);
        clear xlabel
        hold on
        plot([40 40],[0 length(files)],'Linewidth',2);  set(gca,'xtick',[])
        xlabel('Time(s)')
        
        %%%%%%% mean activity over days relationship
        find_figure(strcat('meanactivity-over-days',cats{kk}));
        color={'b','g','y','r'}
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,data1;
        yax=mean(squeeze(data1(files,planes,:)),1);
        se_yax=std(squeeze(data1(files,planes,:)),1)./sqrt(size(squeeze(data1(files,planes,:)),1))
        subplot(2,4,planes),hold on, shadedErrorBar(xax,yax,se_yax,color(planes));
        plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
        clear xlabel
        xlabel('Time(s)');
        plot(xax,rescale(mean(squeeze(data2(files,planes,:)),1),0.995,1),'k');
        %%% mean activity duration calculated for plot grey dotted
        %%% lines
        [rg_x]=xlim; [rg_y]=ylim
        plot([-2.5 -2.5],rg_y,'c'); plot([-0.5 -0.5],rg_y,'c--');
        plot([0.5 0.5],rg_y,'c'); ,plot([2.5 2.5],rg_y,'c--')
        text(5,0.997,strcat('Plane',num2str(planes)),'Color',color{planes})
        text(5,0.996,strcat('ROE'),'Color','k')
        ylabel('norm dF/F')
        
        pre=mean(squeeze(data1(files,planes,20:35)),2); %% -2.56-- -0.64
        pst=mean(squeeze(data1(files,planes,45:60)),2);%%% 0.64-2.56
        mean_pre_pst(1,1)=mean(pre);
        mean_pre_pst(1,2)=mean(pst);
        se_pre_pst(1,1)=std(pre)./sqrt(size(pst,1));
        se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
        
        
        
        %         ids_pre=ones(1,1:size(pre,1))
        %         ids_pre=2*ones(1,1:size(pst,1))
        
        subplot(2,4,planes+4),errorbar([1:2],mean_pre_pst,se_pre_pst); hold on
        
        [h,p]=ttest(pre,pst);
        if h==1
            plot(1.5,mean_pre_pst(1)+se_pre_pst(1),'*k')
        end
        set(gca,'xlim',[0.5 2.5])
        set(gca,'xtick',[1 2],'xticklabel',{'Pre','Post'})
        ylabel('norm dF/F')
        
        
    end
end

%%%%%% perireward


find_figure('dop_days_perireward_planes');clf

for jj=1:4
    find_figure('dop_days_perireward_planes')
    ax(jj)=subplot(6,4,1),imagesc(squeeze(roe_allsuc_perireward(files,jj,:))); hold on
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),gray);  set(gca,'xtick',[])
    
    
    text(10, min(ylim), 'Moving', 'Horiz','left', 'Vert','bottom')
    text(62, min(ylim), 'Stopped', 'Horiz','right', 'Vert','bottom')
    ylabel('Days')
    
    ax(jj)=subplot(6,4,(jj+1-1)*4+1),imagesc(squeeze(dop_allsuc_perireward(files,jj,:))); hold on
%     caxis([0.995 1.005])
    plot([40 40],[0 length(files)],'Linewidth',2); colormap(ax(jj),jet);
    title(strcat('success_Plane',num2str(jj),'__Stop Triggered'));set(gca,'xtick',[])
    
    ylabel('Days')
    
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces;
    yax=mean(squeeze(dop_allsuc_perireward(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_perireward(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_perireward(files,jj,:)),1))
    subplot(6,4,(6-1)*4+1),hold on, shadedErrorBar(xax,yax,se_yax,color(jj));
    plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    clear xlabel; xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.01])
    plot(xax,rescale(mean(squeeze(roe_allsuc_perireward(files,jj,:)),1),0.995,1),'k')
    
    find_figure('fail vs success stop-triggered');
    color=['b','g','y','r']
    xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,norm_single_traces;
    yax=mean(squeeze(dop_allsuc_perireward(files,jj,:)),1);
    se_yax=std(squeeze(dop_allsuc_perireward(files,jj,:)),1)./sqrt(size(squeeze(roe_allsuc_perireward(files,jj,:)),1))
    subplot(6,4,jj),hold on, shadedErrorBar(xax,yax,se_yax,color(jj));
    plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
    xlabel('Time(s)');
    %     set(gca,'ylim',[0.99 1.005])
    plot(xax,rescale(mean(squeeze(roe_allsuc_perireward(files,jj,:)),1),0.995,1),'k')
end


clear dop_suc_movt_pre dop_suc_movt_pre  dop_fail_movt_pre  dop_fail_movt_pst dop_suc_stopt_pre dop_suc_stopt_pst...
    dop_fail_stopt_pre dop_fail_stopt_pst dop_suc_movt_pst dop_suc_perirew_pre  dop_suc_perirew_pst
for planes=1:4
    
    vals=dop_alldays_planes_perireward(files,planes);
    dop_suc_perirew_pre(:,planes)=cellfun(@(x) mean(x(1:39,:),2),vals,'UniformOutput',false);
    dop_suc_perirew_pst(:,planes)=cellfun(@(x) mean(x(40:79,:),2),vals,'UniformOutput',false);
    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cats={ 'dop_suc_perirew_pst'  };
cats4={ 'dop_suc_perirew_pre'  };
cats2={ 'dop_allsuc_perireward'  };
cats3={ 'roe_allsuc_perireward' };
   
for kk=1
    find_figure(strcat('perirew_activity-over-days',cats{kk}));clf
    find_figure(strcat('perirew_meanactivity-over-days',cats{kk}));clf
    for planes=1:4
        data=[];
        eval(sprintf('data=%s(:,%i)',cats{kk},planes))%%%% pst dop
        eval(sprintf('data3=%s(:,%i)',cats4{kk},planes))%%%pre dop
        xvals=1:length(data);
        yvals=data;
        xcord=[];ycord=[];
        for xidx = 1:length(xvals)
            % spread data over region from x-0.2 to x+0.2
            num_yvals = length(yvals{xidx});
            jitter_delta = 0.2/num_yvals;
            x = repmat(xvals(xidx),[num_yvals, 1]);
            jitter = ((0:(num_yvals-1))*jitter_delta - 0.2);
            x = x + jitter';
            y = yvals{xidx};
            %         err = errvals{xidx};
            %             find_figure(strcat('plane_success_fail_prob_epoch',add_dum_fig)); subplot(4,1,allplanes)
            %             hold on
            %             plot(x,y,'r.');
            %             hold on
            xcord=[xcord ;x];
            ycord=[ycord ;y];
        end
        find_figure(strcat('perirew_activity-over-days',cats{kk}));  subplot(2,4,planes+4); hold on
        plot(xcord,ycord,'r.');
        set(gca,'ylim',[0.995 1.005])
        hold on
        plot(cellfun(@(x) mean(x),data),'k','Linewidth',2)
        lsline
        %          subplot(3,4,(planes-1)*1+9), plot(mean(yax2(:,40:79),2))
        
        eval(sprintf('data1=%s',cats2{kk}))%%%20 days dop
        eval(sprintf('data2=%s',cats3{kk}))%%% 20 days roe
        
        
        
        
        yax2=squeeze(data1(files,planes,:));
        subplot(2,4,planes),imagesc(yax2),%stackedplot(xax,yax2)
        %         set(gca,'XTick',[1:118],'XTickLabel',xax);
        clear xlabel
        hold on
        plot([40 40],[0 length(files)],'Linewidth',2);  set(gca,'xtick',[])
        xlabel('Time(s)')
        
        %%%%%%% mean activity over days relationship
        find_figure(strcat('perirew_meanactivity-over-days',cats{kk}));
        color={'b','g','y','r'}
        xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames,data1;
        yax=mean(squeeze(data1(files,planes,:)),1);
        se_yax=std(squeeze(data1(files,planes,:)),1)./sqrt(size(squeeze(data1(files,planes,:)),1))
        subplot(2,4,planes),hold on, shadedErrorBar(xax,yax,se_yax,color(planes));
        plot([0 0],[min(yax) max(yax)],'k','Linewidth',2)
        clear xlabel
        xlabel('Time(s)');
        plot(xax,rescale(mean(squeeze(data2(files,planes,:)),1),0.995,1),'k');
        %%% mean activity duration calculated for plot grey dotted
        %%% lines
        [rg_x]=xlim; [rg_y]=ylim
        plot([-2.5 -2.5],rg_y,'c'); plot([-0.5 -0.5],rg_y,'c--');
        plot([0.5 0.5],rg_y,'c'); ,plot([2.5 2.5],rg_y,'c--')
        text(5,0.997,strcat('Plane',num2str(planes)),'Color',color{planes})
        text(5,0.996,strcat('ROE'),'Color','k')
        ylabel('norm dF/F')
        
        pre=mean(squeeze(data1(files,planes,20:35)),2); %% -2.56-- -0.64
        pst=mean(squeeze(data1(files,planes,45:60)),2);%%% 0.64-2.56
        mean_pre_pst(1,1)=mean(pre);
        mean_pre_pst(1,2)=mean(pst);
        se_pre_pst(1,1)=std(pre)./sqrt(size(pst,1));
        se_pre_pst(1,2)=std(pst)./sqrt(size(pst,1));
        
        
        
        %         ids_pre=ones(1,1:size(pre,1))
        %         ids_pre=2*ones(1,1:size(pst,1))
        
        subplot(2,4,planes+4),errorbar([1:2],mean_pre_pst,se_pre_pst); hold on
        
        [h,p]=ttest(pre,pst);
        if h==1
            plot(1.5,mean_pre_pst(1)+se_pre_pst(1),'*k')
        end
        set(gca,'xlim',[0.5 2.5])
        set(gca,'xtick',[1 2],'xticklabel',{'Pre','Post'})
        ylabel('norm dF/F')
        
        
    end
end




