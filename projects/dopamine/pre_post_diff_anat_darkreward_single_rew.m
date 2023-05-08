clear all
close all
% mouse_id=156;
mov_corr=[]; stop_corr=[]; mov_stop=[];
dop_suc_movint=[]; dop_suc_movint=[];
dop_suc_stopint=[]; dop_suc_stopint=[];
roe_suc_movint=[]; roe_suc_movint=[];
roe_suc_stopint=[]; roe_suc_stopint=[];
% dop_fail_movint=[]; dop_fail_stopint=[];
% days_check=[2:3 5:8 10 12:17 21 24 30:33];
% days_check=[15:31]%[12:31];
% days_check=[40:44];
% days_check=[26:30]%[40:44]%[26:30]%40:44%[12:31];
subp=0;
GFP=0;
darkreward=1;
pr_dir0 = uipickfiles;


for alldays=1:length(pr_dir0)%31%41%12%30%ays_check%[1:2 4:5 12:27 29]%[2:3 5:8 10 12:17 21  24 26]%22:29%[1:2 4:5 12:22]%1:21%[1:2 4:5 12:22]%21%[1:2 4:5 12:22]%[1:21]%%[1:2 4:5 12:20]%%[1:21]%
    
    subp=subp+1;
    clearvars -except alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr dop_suc_movint dop_suc_stopint roe_suc_movint roe_suc_stopint ...
        dop_allsuc_mov dop_allsuc_stop roe_allsuc_mov roe_allsuc_stop ...
        dop_allfail_mov dop_allfail_stop roe_allfail_mov roe_allfail_stop...
        dop_alldays_planes_success_mov dop_alldays_planes_fail_mov dop_alldays_planes_success_stop dop_alldays_planes_fail_stop...
        roe_alldays_planes_success_mov roe_alldays_planes_fail_mov roe_alldays_planes_success_stop roe_alldays_planes_fail_stop...
        subp days_check GFP darkreward pr_dir0
    
    %     close all
    Day=alldays;
    for allplanes=2:4
        plane=allplanes;
        pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
        pr_dir2=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','')
        
        
        if exist( pr_dir2, 'dir')
            cd (pr_dir2)
            pr_dir2
            list=dir('*.mat');
            load('params.mat')
            
            tiffpath = pr_dir2;
            cd(tiffpath)
            
            
            list_tif=dir('*.tif');
            filenames={list_tif(:).name};
            checks=strfind(filenames,('plane'));
            plane_tif=cellfun(@(x) ~isempty(x), checks);
            ntif=find(plane_tif~=1);
            myfilename = (list_tif(ntif(1)).name);
            info=imfinfo(myfilename);
            M=info(1).Width;
            N=info(1).Height;
            
            chone=[];
            for jj=1:length(ntif)
                myfilename = (list_tif(ntif(jj)).name);
                info=imfinfo(myfilename);
                numframes=length(info);
                M=info(1).Width;
                N=info(1).Height;
                
                chone_temp=zeros(N,M,numframes);
                parfor i=1:numframes
                    chone_temp(:,:,i)=imread(myfilename,i,'Info',info);
                end
                chone=cat(3,chone,chone_temp);
            end
            clear chone_temp
            
            %%%%%%%%%%%
            %             norm_chone=chone-mean(mean(mean(chone(:,:,:))));
            %             find_figure(''); imagesc(mean(norm_chone,3))
            
            %             single_lick_idx=rew_idx;
            
            %%%%%%%
            %%% cropping it
            sp_img=1;
            a=size(chone,1); n=sp_img;
            b=size(chone,2); n=sp_img;
            nr_rsz=((sp_img - rem(a,sp_img))+a)-sp_img;
            nc_rsz=((sp_img - rem(b,sp_img))+b)-sp_img;
            %             new_img=chone(1:nr_rsz,1:nc_rsz,:);
            %%%bin im r X c
            
            
            %Parameters of MOVIE
            numplanes=4;
            gauss_win=5;
            frame_rate=31.25/numplanes;
            lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
            rew_thresh=0.001;
            sol2_thresh=1.5;
            num_rew_win_sec=5;%window in seconds for looking for multiple rewards
            rew_lick_win=20;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
            nr=size(chone,1); nc=size(chone,2)
            nr_steps=1:sp_img:nr; nc_steps=1:sp_img:nc;
            resz_post_pre=zeros(size(chone,1),size(chone,2));
            perc_post_pre=zeros(size(chone,1),size(chone,2));
            pre_img_binz=[]; post_img_binz=[];
            pre_win=5;%pre window in s for rewarded lick average
            post_win=5;%post window in s for rewarded lick average
            exclusion_win=10;%exclusion window pre and post rew lick to look for non-rewarded licks
            speed_thresh = 0.8; %cm/s cut off for stopped
            Stopped_frame = 40;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 4; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            CSUStimelag = 0.5; %seconds between
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time);
            pre_win_frames=round(pre_win/frame_time);
            exclusion_win_frames=round(exclusion_win/frame_time);
            CSUSframelag_win_frames=round(CSUStimelag/frame_time);
            
            %%%% time stamps for CS
            %              reward_binned=rewardsALL;
            %             % temporary artefact check and remove
            %             temp= find(reward_binned);
            %             reward_binned(temp(find(diff(temp) == 1))) = 0;
            
            reward_binned=rewards;
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
            
            singlerew = single_rew(find(single_rew>pre_win_frames&single_rew<length(licksALL)-post_win_frames))-CSUSframelag_win_frames;
            
            
            tm_fr=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames;
            clear chone_lick_fr chone_lick_fr_norm
            chone_lick_fr(:,:,size(tm_fr,2),length(singlerew))=zeros(size(chone,1),size(chone,2));
            chone_lick_fr_norm(:,:,size(tm_fr,2),length(singlerew))=zeros(size(chone,1),size(chone,2));
            
            
            for i=1:length(singlerew)
                chone_lick_fr(:,:,:,i)=squeeze(chone(:,:,singlerew(i)-pre_win_frames:singlerew(i)+post_win_frames));
                chone_lick_fr_norm(:,:,:,i) = squeeze(chone_lick_fr(:,:,:,i))-nanmean(squeeze(chone_lick_fr(:,:,1:pre_win_frames,i)),3);
            end
            
            size(chone_lick_fr) %%% x y frames rw_idx
            
            
            %%%% normalized
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %left off here!
            
            clear pre_chone_lick post_chone_lick
            pre_chone_lick(:,:,length(find(tm_fr<0)),length(singlerew))=zeros(size(chone,1),size(chone,2));
            post_chone_lick(:,:,length(find(tm_fr<0)),length(singlerew))=zeros(size(chone,1),size(chone,2));
            pre_chone_lick(:,:,:,:)=chone_lick_fr(:,:,find(tm_fr<0),:);
            post_chone_lick(:,:,:,:)=chone_lick_fr(:,:,find(tm_fr>=0,length(find(tm_fr<0))),:);
            chone_lick_movie = nanmean(chone_lick_fr_norm,4);
            limss = max([abs(min(min(min(chone_lick_movie)))) abs(max(max(max(chone_lick_movie))))]);
            
            
            %%%% remove strippy lines
            
            
            x1=41; % ZD changed this dim to remove stripy lines
            x2=619;
            y1=169;
            y2=500;
            
            chone_lick_movie=chone_lick_movie(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);%
            
            
            
            
            clear smallimage
 
            I=chone_lick_movie;
            for col = 1:4:size(I,2)-4
                for row = 1:4:size(I,1)-4
                    smallimage((row+3)/4,(col+3)/4,:) = (mean(mean((I(row:row+4,col:col+4,:)))));
                end
            end
            
            limss = max([abs(min(min(min(smallimage)))) abs(max(max(max(smallimage))))]);
            chone_crop=chone(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);%
            
            find_figure('reward_alig_movie'); h1=subplot(1,2,1),imagesc((imresize(squeeze(mean(chone_crop,3)),1/4))); colormap(h1,gray); axis image
            
            
            find_figure('reward_alig_movie')
            clearvars F imagen
            for fra = 1:size(smallimage,3)
                imagen=squeeze(smallimage(:,:,fra));
                h2=subplot(1,2,2),imagesc(imagen);
                colormap(h2,redblue); axis image
                caxis([-1*limss limss]);
                
                
                if fra>40
                    position =  [1 50; 100 50];
                    value = [555 pi];
                    text(60,60, 'reward');
                end
                
                F(fra) = getframe(gcf);
                drawnow
                
            end
            
            cd(pr_dir0{1})
            writerObj = VideoWriter(strcat('plane_',num2str(allplanes),'_PeriRewardDiff3.avi'));
            writerObj.FrameRate = 8;
            
            % open the video writer
            open(writerObj);
            % write the frames to the video
            for i=1:length(F)
                % convert the image to a frame
                frame = F(i) ;
                writeVideo(writerObj, frame);
            end
            % close the writer object
            close(writerObj);
            %  figure;plot([(squeeze(mean(mean(mean(pre_chone_lick,1),2),4)))' (squeeze(mean(mean(mean(post_chone_lick,1),2),4)))'])
            %             find_figure('post-pre');clf;hold on;
            %             %             subplot(3,2,1), imagesc(squeeze(mean(chone(:,:,:),3))); axis image; title('mean_image'); colormap(gray)
            %             %             subplot(3,2,4),imagesc(squeeze((mean(mean(post_chone_lick,4),3))));colormap(parula);axis image; title('post_img')
            %             %             subplot(3,2,3),imagesc(squeeze((mean(mean(pre_chone_lick,4),3))));colormap(parula);axis image; title('pre_img')
            %             %
            %             sig_img_neg=zeros(size(chone,1),size(chone,2));
            %             sig_img_pos=zeros(size(chone,1),size(chone,2));
            %             dpr_img=zeros(size(chone,1),size(chone,2));
            %
            %             %             for jj=1:length(nr_steps)-1
            %             %                 for kk=1:length(nc_steps)-1
            %             %                     %%%calc mean of binned pixel
            %             %                     bin_pix=chone(nr_steps(jj):nr_steps(jj+1)-1, nc_steps(kk):nc_steps(kk+1)-1,:);
            %             %                     if sp_img~=1
            %             %                         perbin_bmean=squeeze(mean(mean(bin_pix)));
            %             %                     else
            %             %                         perbin_bmean=squeeze(bin_pix)';
            %             %                     end
            %             %                     binz_r=[min(nr_steps(jj):nr_steps(jj+1)-1) max(nr_steps(jj):nr_steps(jj+1)-1)];
            %             %                     binz_c=[min(nc_steps(kk):nc_steps(kk+1)-1) max(nc_steps(kk):nc_steps(kk+1)-1)];
            %             %                     x = [binz_c(1), binz_c(2),binz_c(2) ,binz_c(1), binz_c(1)];
            %             %                     y = [binz_r(1), binz_r(1),binz_r(2) ,binz_r(2), binz_r(1)];
            %             %                     perbin_single_traces=[];
            %             %                     for i=1:length(singlerew)
            %             %                         perbin_single_traces(:,i)=perbin_bmean(singlerew(i)-pre_win_frames:singlerew(i)+post_win_frames)';%lick at pre_win_frames+1
            %             %                     end
            %             %                     normperbin_single_traces=perbin_single_traces./mean(perbin_single_traces(1:pre_win_frames,:));
            %             %                     tm_fr=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames;
            %             %                     pre_win=normperbin_single_traces(find(tm_fr<0),:);
            %             %                     post_win=normperbin_single_traces(find(tm_fr>=0,size(pre_win,1)),:);
            %             %                     pre_img_binz(jj,kk)=mean(mean(pre_win));
            %             %                     post_img_binz(jj,kk)=mean(mean(post_win));
            %             %                     resz_post_pre(min(y):max(y),min(x):max(x))= post_img_binz(jj,kk)- pre_img_binz(jj,kk);
            %             %                     perc_post_pre(min(y):max(y),min(x):max(x))= (post_img_binz(jj,kk)-pre_img_binz(jj,kk)/pre_img_binz(jj,kk))*100;
            %             %                     %                     %%%%% perform significance
            %             %                     %                     %%%validate by dpr
            %             %                     %                     basedff=mean(pre_win,1); respdff=mean(post_win(1:size(pre_win,1),:),1);
            %             %                     %                     dpr=(mean(respdff)-mean(basedff))/sqrt((std(respdff)^2+std(basedff)^2));
            %             %                     %                     dpr_img(min(y):max(y),min(x):max(x))=dpr;
            %             %                     %                     %          dpr
            %             %                     %                     %%% validate by ttest
            %             %                     %                     [h_neg,p_neg]=ttest(basedff, respdff,0.05,'right');
            %             %                     %                     [h_pos,p_pos]=ttest(basedff, respdff,0.05,'left');
            %             %                     %                     %          %         if abs(dpr)>0.5
            %             %                     %                     %          find_figure('check pix');
            %             %                     %                     %          title(num2str([p dpr]))
            %             %                     %
            %             %                     %                     if h_neg==1
            %             %                     %
            %             %                     %                         sig_img_neg(min(y):max(y),min(x):max(x))=1;
            %             %                     %                     end
            %             %                     %                     if h_pos==1
            %             %                     %                         sig_img_pos(min(y):max(y),min(x):max(x))=1;
            %             %                     %                     end
            %             %                 end
            %             %             end
            %             %             subplot(3,2,5),imagesc(resz_post_pre); axis image; title('post-pre'); colormap(parula)
            %             diff_post_prelick=post_chone_lick-pre_chone_lick;
            %             subplot(3,2,5),imagesc(squeeze(mean(mean(diff_post_prelick,4),3)))
            %             %               subplot(3,2,6),imagesc(perc_post_pre); axis image; title('%post-pre/pre')
            %             colorbar
            %             subplot(3,2,6),pre_mean_sintrace=squeeze(mean(mean(pre_chone_lick(:,:,:,:),1)));
            %             post_mean_sintrace=squeeze(mean(mean(post_chone_lick(:,:,:,:),1)));
            %             allsin_traces=[pre_mean_sintrace' post_mean_sintrace'];
            %             plot(allsin_traces'); hold on ; plot(mean(allsin_traces),'k','Linewidth',2)
            %             %%%%%%
            %             % %            save(strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'_','SP_SO_SR','.mat'))
            %             %              save(strcat('pre_post_df','anat_plane',num2str(plane)), '-v7.3','chone','pre_chone_lick','post_chone_lick')
            %
            %             %             %%%%% seleceted area
            %             %             find_figure('for area selection'); clf
            %             %             imagesc(squeeze(mean(chone(:,:,:),3))); axis image; title('mean_image'); colormap(gray)
            %             %             [x1,y1,BW,xi,yi]=roipoly;
            %             %             find_figure('resp_profile_sel_area');clf
            %             %             lin_idx=find(BW==1);
            %             %             allzeros=ind2sub(size(BW),lin_idx);
            %             %             subplot(4,2,1); imagesc(squeeze(mean(chone,3)))
            %             %             subplot(4,2,2), imagesc(BW);
            %             %             sel_img=BW.*(chone); subplot(4,2,3),imagesc(squeeze(mean(sel_img,3)))
            %             %             %%%%%
            %             %             sel_pre_chone_lick=pre_chone_lick.*BW;
            %             %             sel_post_chone_lick=post_chone_lick.*BW;
            %             %             subplot(4,2,5),imagesc(squeeze((mean(mean(sel_post_chone_lick,4),3))));colormap(parula);axis image; title('post_img')
            %             %             subplot(4,2,6),imagesc(squeeze((mean(mean(sel_pre_chone_lick,4),3))));colormap(parula);axis image; title('pre_img')
            %             %             sel_pre_chone_lick(sel_pre_chone_lick==0)=NaN;
            %             %             sel_post_chone_lick(sel_post_chone_lick==0)=NaN;
            %             %             sel_pre_mean_sintrace=squeeze(nanmean(nanmean(sel_pre_chone_lick(:,:,:,:),1)));
            %             %             sel_post_mean_sintrace=squeeze(nanmean(nanmean(sel_post_chone_lick(:,:,:,:),1)));
            %             %             sel_allsin_traces=[sel_pre_mean_sintrace' sel_post_mean_sintrace'];
            %             %             subplot(4,2,8),plot(sel_allsin_traces'); hold on ; plot(mean(sel_allsin_traces),'Linewidth',2)
            %             mean_image=squeeze(mean(chone,3));
            %             save(strcat('perpix_pre_pos_plane',num2str(allplanes)),'-v7.3','mean_image','pre_chone_lick','post_chone_lick')
        end
        %         pause
    end
end