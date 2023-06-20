%%%%% batch processing
%%%lets plot mean image of first 500 frames across days to analyze
%%%parent directory
clear all
close all
mouse_id=158;
pr_dir=uipickfiles;
days_check=1:length(pr_dir);
ref_exist=0;%%% if reference image hase been a;ready choose
if ref_exist
pr_dirref=uipickfiles;%%% chose reference day here day1
end

for allplanes=1:4
    for days=days_check
        pr_dir1=strcat(pr_dir{days},'\')
        %%%s2p directory
        pr_dir2=strcat(pr_dir1,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
        %%% grab 1st file/initial 500 frames
        cd(pr_dir2)
        list_tif=dir('*.tif');
        filenames={list_tif(:).name};
        checks=strfind(filenames,('plane'));
        plane_tif=cellfun(@(x) ~isempty(x), checks);
        ntif=find(plane_tif~=1);
        %             ntif=1:length(filenames);
        myfilename = (list_tif(ntif(1)).name);
        info=imfinfo(myfilename);
        M=info(1).Width;
        N=info(1).Height;
        numframes=length(info);
        chone_temp=[];
        for i=1:numframes
            chone_temp(:,:,i)=imread(myfilename,i,'Info',info);
        end
        
        crop_points1=[41 169 619 512]; %%% Direct .sbx crop
        
        eval(['x1=crop_points' num2str(1) '(1)']);  %x for area for correction
        eval(['x2=crop_points' num2str(1) '(3)']);
        eval(['y1=crop_points' num2str(1) '(2)']);
        eval(['y2=crop_points' num2str(1) '(4)'])
        
        M=size(chone_temp,2);
        N=size(chone_temp,1);
        chone_temp=chone_temp(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);
        
        mimg(:,:,allplanes,days)=squeeze(mean(chone_temp(:,:,:),3));
        dir_data{days,1}=pr_dir2;
    end
    
    %%% plot number of days mean image
    find_figure(strcat('m3000f_days_','plane',num2str(allplanes))); clf
    for kk=days_check
        find_figure(strcat('m3000f_days_','plane',num2str(allplanes))), subplot(ceil(sqrt(length(days_check))),ceil(sqrt(length(days_check))),kk),
        imagesc(squeeze(mean(chone_temp(:,:,:),3))); colormap(gray), axis image
    end
    
    
    
    
    %%%%% reference Roi last drawn on day
    
    %%%% plot reference image and poylgon
    if ref_exist==1
%         pr_dir3=strcat(pr_dir,num2str(mouse_id),'\Day_',num2str(ref_day),'\');
        %%%s2p directory
        pr_dir4ref=strcat(pr_dirref{1},'\suite2p\plane',num2str(allplanes-1),'\reg_tif\');
        %%% grab 1st file/initial 3000 frames
        cd(pr_dir4ref)
        load('params')
        
        for jj=1:size(params.BW_mask,1)
            if jj==1
                find_figure(strcat('REFERENCE','_plane',num2str(allplanes)))
                imagesc(params.mimg); colormap(gray), axis image
            end
            
            if ~isempty(params.BW_mask{jj,1})
                color_cod={'y','--y','-oy'};
                find_figure(strcat('REFERENCE','_plane',num2str(allplanes)))
                hold on;  plot(params.roi_coords{jj,1}(:,1),params.roi_coords{jj,1}(:,2),color_cod{jj},'Linewidth',2)
                
                
            end
        end
    else
        %%%% draw a polygon on first day based on 4 days of data
        
        
        for kk=days_check
            find_figure(strcat('REFERENCE','_plane',num2str(allplanes))), subplot(ceil(sqrt(length(days_check))),ceil(sqrt(length(days_check))),kk),
            imagesc(squeeze(mean(mimg(:,:,allplanes,days_check(kk)),4))); colormap(gray), axis image
            
        end
%         pr_dir3=strcat(pr_dir,num2str(mouse_id),'\Day_',num2str(1),'\');
        pr_dir3 = strcat(pr_dir{1},'\')
        pr_dir4=strcat(pr_dir3,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
        pr_dir4ref = pr_dir4;
        find_figure(strcat('REFERENCE','_plane',num2str(allplanes)))
        %         for roi_sel=1:4
        color_cod={'k','--k','-ok'};
        h=subplot(ceil(sqrt(length(days_check))),ceil(sqrt(length(days_check))),1);
        w=1;lopc=0;
        clear BW_cell coords
        while w==1
            lopc=lopc+1;
            mpimg=mimg(:,:,allplanes,days_check(1));
            
            [x1,y1,BW,xi,yi]=roipoly;hold on
            plot(xi,yi,color_cod{lopc},'Linewidth',2)
            
            w=input('input vals 1 to keep drawing or 0 to move to next subplot');
            BW_cell{lopc,1}=BW;
            coords{lopc,1}=[xi,yi];
            
        end
        
        datapath=pr_dir4;
        params.mimg=mpimg;
        params.dir=datapath;
        params.BW_mask=BW_cell;
        params.roi_coords=coords;
        extention='.mat';
        
        matname = fullfile(datapath, ['params' extention]);
        save(matname, 'params')
    end
    
    
    
    
    %%% extract ROI
    %%%%% gcf subplot
    
    %%% plot number of days mean image
    cd(pr_dir2)
    %         close
    find_figure(strcat('m500f_days_','plane',num2str(allplanes))); clf
    
    for kk=days_check
        find_figure(strcat('m500f_days_','plane',num2str(allplanes))), subplot(ceil(sqrt(length(days_check))),ceil(sqrt(length(days_check))),kk),
        imagesc(squeeze(mean(mimg(:,:,allplanes,days_check(kk)),4))); colormap(gray), axis image
    end
    
    %%%% drag the polygon over days  now
    
    find_figure(strcat('m500f_days_','plane',num2str(allplanes)));
    ndays=0;
    for roi_sel=days_check%%% days
        clear newpos BWnewpos
        ndays=ndays+1;
        for fk=1:size(params.roi_coords,1)
           
            cd(pr_dir4ref)
            load('params')
            
            
            
            color_cod={'k','--k','-ok'};
            h=subplot(ceil(sqrt(length(days_check))),ceil(sqrt(length(days_check))),roi_sel),
            w=1;lopc=0;
            cd(pr_dir4ref)
            
            h = impoly(gca,  params.roi_coords{fk,1});
            setColor(h,'yellow');
            id=addNewPositionCallback(h,@(p) title(mat2str(p,3))) ;
            
            fcn = makeConstrainToRectFcn('impoly',get(gca,'XLim'),...
                get(gca,'YLim'));
            setPositionConstraintFcn(h,fcn);
            
            w2=input('Roi position final press1 else 0=')
            
            if w2==1
                
                h1=get(gca,'title');
                mpimg=mimg(:,:,allplanes,days_check(ndays));
                newpos{fk,1}=str2num(h1.String);
                BWnewpos{fk,1}=poly2mask(newpos{fk,1}(:,1),newpos{fk,1}(:,2),size(params.mimg,1),size(params.mimg,2));
            end
            
        end
        pr_dir_ns=strcat(pr_dir{roi_sel},'\');
        pr_dir_s=strcat(pr_dir_ns,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
        cd(pr_dir_s)
        params.mimg=mpimg;
        params.newroicoords=newpos;
        params.newBWmask=BWnewpos;
        save('params.mat','params')
        w=input('input vals 1 to keep drawing or 0 to move to next subplot');
        
    end
end

%%
%extract base mean from all rois selected before
clear all
close all
mouse_id=158;
pr_dir=uipickfiles;
days_check=1:length(pr_dir);
tic
for allplanes=1:4
    for days=days_check
        days
        pr_dir1=strcat(pr_dir{days},'\')
        %%%s2p directory
        pr_dir2=strcat(pr_dir1,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
        %%% grab 1st file/initial 3000 frames
        cd(pr_dir2)
        
        load('params')
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
        tic
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
        toc
        %%%%%
        
        crop_points1=[41 169 619 512]; %%% Direct .sbx crop
        
        eval(['x1=crop_points' num2str(1) '(1)']);  %x for area for correction
        eval(['x2=crop_points' num2str(1) '(3)']);
        eval(['y1=crop_points' num2str(1) '(2)']);
        eval(['y2=crop_points' num2str(1) '(4)'])
        
        M=size(chone,2);
        N=size(chone,1);
        chone=chone(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);
        
        
        %         toc
        
        clear chone_temp  roim_sel roibase_mean
        %%%% extract df/f for an roi
        load('params.mat')
        for jj=1:size(params.newBWmask,1)
            if ~isempty(params.newBWmask{jj,1})
                masksel=params.newBWmask{jj,1};
                roim_sel=chone.*masksel;
                roim_sel(roim_sel==0)=NaN;
                mean_roi=squeeze(nanmean(nanmean(roim_sel(:,:,:,:),1)));
                
                %%%%% Baseline correction each roi
                
                numframes=size(roim_sel,3);
                base_window=200;
                meanlastframes=nanmedian(nanmean(nanmean(roim_sel(:,:,(end-base_window):end))));
                meanfirstframes=nanmedian(nanmean(nanmean(roim_sel(:,:,1:base_window))));
                roim_sel=roim_sel*(meanlastframes/meanfirstframes);
                %baseline subtract whole movie
                junk=squeeze(nanmean(nanmean(roim_sel)));%mean of each image, frame x 1 vector
                mean_all=nanmean(nanmean(nanmean(roim_sel)));
                junk2=zeros(size(junk));
                parfor kk=1:length(junk)
                    cut=junk(max(1,kk-base_window):min(numframes,kk+base_window));
                    cutsort=sort(cut);
                    a=round(length(cut)*.08);
                    junk2(kk)=cutsort(a);
                end
                
                parfor i=1:numframes
                    roim_sel(:,:,i)=(roim_sel(:,:,i)/junk2(i))*mean_all;
                end
                
                
                roibase_mean{jj,1}=squeeze(nanmean(nanmean(roim_sel(:,:,:),1)));
                
                find_figure('base_mean_rois');
                if jj==1
                    subplot(4,1,1),imagesc(squeeze(mean(chone,3))); colormap(gray), axis image
                end
                color_cod={'k','--k','-ok'};
                find_figure('base_mean_rois'); subplot(4,1,1)
                hold on;  plot(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),color_cod{jj},'Linewidth',2)
                
                subplot(4,1,jj+1),plot( roibase_mean{jj,1});
                
                
            end
            params.roibasemean2=roibase_mean;
            
            
        end
        junk=squeeze(nanmean(nanmean(chone)));%mean of each image, frame x 1 vector
        mean_all=nanmean(nanmean(nanmean(chone)));
        junk2=zeros(size(junk));
        parfor kk=1:length(junk)
            cut=junk(max(1,kk-base_window):min(numframes,kk+base_window));
            cutsort=sort(cut);
            a=round(length(cut)*.08);
            junk2(kk)=cutsort(a);
        end
        params.raw_mean = junk;
        parfor i=1:numframes
            chone(:,:,i)=(chone(:,:,i)/junk2(i))*mean_all;
        end
        params.base_mean = squeeze(nanmean(nanmean(chone)));
        save('params','params','-append')
    end
    
end
toc

% %%
% clear all
% % close all
% mouse_id=148;
% pr_dir='G:\dark_reward\E';
% days_check=[1:4];
% tic
% find_figure('alldays2n'); clf
% find_figure('perireward-perilocn');clf
% c=0;
% for allplanes=1:4
%    
%     find_figure(strcat('mouse',num2str(mouse_id)))
%     
%     for days=days_check
%         
%         c=c+1;
%         
%         pr_dir1=strcat(pr_dir,num2str(mouse_id),'\Day_',num2str(days),'\')
%         %%%s2p directory
%         pr_dir2=strcat(pr_dir1,'suite2p\plane',num2str(allplanes-1),'\reg_tif\')
%         %%% grab 1st file/initial 3000 frames
%         cd(pr_dir2)
%         
%         
%         load('params')
%        find_figure(strcat('mouse',num2str(mouse_id)))
%         subplot(4,4,c),  imagesc(params.mimg); colormap(gray), axis image;
%         title('Day',num2str(days))
%         for jj=1:size(params.newroicoords,1)
%             if jj==1
%                 find_figure(strcat('base_mean_rois','_plane',num2str(allplanes)))
%                 subplot(4,1,1),imagesc(params.mimg); colormap(gray), axis image
%             end
%             
%             if ~isempty(params.newroicoords{jj,1})
%                 color_cod={'k','--k','-ok'};
%                 find_figure(strcat('base_mean_rois','_plane',num2str(allplanes))); subplot(4,1,1)
%                 hold on;  plot(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),color_cod{jj},'Linewidth',2)
%                 BWnewpos{jj,1}=poly2mask(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),size(params.mimg,1),size(params.mimg,2));
%                 
%                 
%                 %                 subplot(4,1,jj+1),plot(params.roibasemean{jj,1});
%                 find_figure(strcat('mouse',num2str(mouse_id)));subplot(4,4,c),
%                 hold on;  plot(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),color_cod{jj},'Linewidth',2)
%                 %                     imagesc(params.newBWmask{jj,1})
%                 
%                 
%             end
%         end
%         params.BWnewpos=BWnewpos;
% %         save('params.mat','params','-append')
%     end
% end
% 
% %% perireward and periloc for all the rois extarct
% clear all
% close all
% for days=1:12
%     
%     find_figure('perireward-periloc'); clf
%     mouse_id=149;
%     pr_dir='G:\dark_reward\E';
%     
%     col=['b','g','y','r'];
%     mark={'','--','-o'};
%     
%     cnt=0;
%     for plane=1:4
%         
%         pr_dir1=strcat(pr_dir,num2str(mouse_id),'\Day_',num2str(days),'\')
%         %%%s2p directory
%         pr_dir2=strcat(pr_dir1,'suite2p\plane',num2str(plane-1),'\reg_tif\')
%         %%% grab 1st file/initial 3000 frames
%         cd(pr_dir2)
%         list=dir('*.mat');
%         %%%%%load worksapce with behav vars appended
%         if isempty(strfind(list(1).name,'mean'))
%             load(strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'.mat'))
%             strcat(list(1).name(1:end-4),'_1_mean_plane',num2str(plane),'.mat')
%         else
%             load(strcat(list(1).name(1:end-16),'_mean_plane',num2str(plane),'.mat'))
%             strcat(list(1).name(1:end-16),'_mean_plane',num2str(plane),'.mat')
%         end
%         
%         
%         
%         
%         %%%load params
%         load('params')
%         subplot(4,4,(plane-1)*4+1), imagesc(params.mimg); colormap(gray); axis image
%         
%         for nroi=1:length(params.newBWmask)
%             cnt=cnt+1;
%             roibase_mean=params.roibasemean2{nroi,1}%%%base mean
%             
%             numplanes=4;
%             gauss_win=5;
%             frame_rate=31.25/numplanes;
%             lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
%             rew_thresh=0.001;
%             sol2_thresh=1.5;
%             num_rew_win_sec=5;%window in seconds for looking for multiple rewards
%             rew_lick_win=20;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
%             pre_win=5;%pre window in s for rewarded lick average
%             post_win=5;%post window in s for rewarded lick average
%             exclusion_win=20;%exclusion window pre and post rew lick to look for non-rewarded licks
%             
%             
%             frame_time=1/frame_rate;
%             num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
%             rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
%             post_win_frames=round(post_win/frame_time);
%             pre_win_frames=round(pre_win/frame_time);
%             exclusion_win_frames=round(exclusion_win/frame_time);
%             
%             
%             mean_roibase_mean=mean(roibase_mean);
%             norm_roibase_mean=base_mean/mean_roibase_mean;
%             speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
%             doproi_smth=smoothdata(norm_roibase_mean,'gaussian',gauss_win);
%             
%             
%             R = bwlabel(rew_binned>rew_thresh);%label rewards, ascending
%             rew_idx=find(R);%get indexes of all rewards
%             rew_idx_diff=diff(rew_idx);%difference in reward index from last
%             short=rew_idx_diff<num_rew_win_frames;%logical for rewards that happen less than x frames from last reward. 0 = single rew.
%             
%             %single rewards
%             
%             
%             [r c]=find(single_lick_idx==0);
%             single_lick_idx(c)=[];
%             roisingle_traces=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
%             
%             
%             roisingle_traces_roesmth=zeros(pre_win_frames+post_win_frames+1,length(single_lick_idx));
%             coeff_rewarded_licks=[]; coeff_norm_rewarded_licks=[];  lags_single_traces=[];
%             for i=1:length(single_lick_idx)
%                 
%                 roisingle_traces(:,i)=roibase_mean(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';%lick at pre_win_frames+1
%                 roisingle_traces_roesmth(:,i)=speed_smth_1(single_lick_idx(i)-pre_win_frames:single_lick_idx(i)+post_win_frames)';
%                 
%             end
%             
%             norm_roisingle_traces=roisingle_traces./mean(roisingle_traces(1:pre_win_frames,:));
%             norm_roisingle_traces_roesmth=roisingle_traces_roesmth./mean(roisingle_traces_roesmth(1:pre_win_frames,:));
%             
%             if exist('single_traces','var')
%                 find_figure('perireward-periloc');
%                 
%                 xlabel('seconds from first reward lick')
%                 ylabel('dF/F')
%                 hold on
%                 color_cod=cell2mat(strcat(mark(nroi),col(plane)));
%                 
%                 subplot(4,4,(plane-1)*4+1),
%                 hold on;  plot(params.newroicoords{nroi,1}(:,1),params.newroicoords{nroi,1}(:,2),color_cod,'Linewidth',2)
%                 
%                 xax=frame_time*(-pre_win_frames):frame_time:frame_time*post_win_frames;
%                 %              subplot(4,4,(plane-1)*4+2),plot(xax,norm_roisingle_traces);
%                 %             hold on
%                 subplot(4,4,(plane-1)*4+2),hold on; plot(xax,mean(norm_roisingle_traces,2),color_cod,'LineWidth',2);
%                 legend(['n = ',num2str(size(norm_roisingle_traces,2))])
%                 title('perireward')
%                 plot(xax,rescale(mean(norm_roisingle_traces_roesmth,2),0.99,1),'k','Linewidth',2)
%                 plot([0 0],[min(ylim) max(ylim)],'k--')
%                 %             plot(norm
%                 
%             end
%             
%             %%%%perlocomoton success moving initiation
%             
%             
%             [moving_middle,stopping_middle]=mov_stop_tmstmp_2(speed_smth_1,speed_smth_1);
%             mov_success_tmpts=[];
%             ids_loc=[1 find(diff(moving_middle)>1) length(moving_middle)]; c_tmstmp=0;
%             for movwin=1:length(ids_loc)-1
%                 tmstmps=moving_middle(ids_loc(movwin)+1:ids_loc(movwin+1));
%                 if size(tmstmps,2)>round(2/frame_time)
%                     c_tmstmp=c_tmstmp+1;
%                     mov_success_tmpts(c_tmstmp,1)=tmstmps(1);
%                     mov_success_tmpts(c_tmstmp,2)=tmstmps(end);
%                 end
%             end
%             
%             idx_rm=(mov_success_tmpts- pre_win_frames)<0;
%             rm_idx=find(idx_rm(:,1)==1)
%             mov_success_tmpts(rm_idx,:)=[];
%             
%             
%             idx_rm=(mov_success_tmpts+post_win_frames)>length(roibase_mean);
%             rm_idx=find(idx_rm(:,1)==1)
%             mov_success_tmpts(rm_idx,:)=[];
%             allmov_success=NaN(1,size(roibase_mean,2));
%             dop_success_perimov=[]; roe_success_perimov=[];
%             for stamps=1:size(mov_success_tmpts,1)
%                 dop_success_perimov(stamps,:)= roibase_mean(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
%                 roe_success_perimov(stamps,:)= speed_smth_1(mov_success_tmpts(stamps)-pre_win_frames:mov_success_tmpts(stamps)+post_win_frames);
%                 
%                 allmov_success(mov_success_tmpts(stamps,1):mov_success_tmpts(stamps,2))=1;
%             end
%             
%             norm_dopsuccess_perimov= dop_success_perimov./mean( dop_success_perimov(:,1:pre_win_frames),2);
%             norm_roesuccess_perimov= roe_success_perimov./mean( roe_success_perimov(:,1:pre_win_frames),2);
%             
%             %%%%stopping success trials
%             %
%             stop_success_tmpts=[];
%             ids_loc=[1 find(diff(stopping_middle)>1) length(stopping_middle)]; c_tmstmp=0;
%             for movwin=1:length(ids_loc)-1
%                 tmstmps=stopping_middle(ids_loc(movwin)+1:ids_loc(movwin+1));
%                 if size(tmstmps,2)>round(2/frame_time)
%                     c_tmstmp=c_tmstmp+1;
%                     stop_success_tmpts(c_tmstmp,1)=tmstmps(1);
%                     stop_success_tmpts(c_tmstmp,2)=tmstmps(end);
%                 end
%             end
%             
%             
%             idx_rm= stop_success_tmpts(:,1)- pre_win_frames <0;
%             rm_idx=find(idx_rm(:,1)==1)
%             stop_success_tmpts(rm_idx,:)=[];
%             
%             
%             idx_rm=(stop_success_tmpts+post_win_frames)>length(norm_roibase_mean);
%             rm_idx=find(idx_rm(:,1)==1)
%             stop_success_tmpts(rm_idx,:)=[];
%             
%             dop_success_peristop=[]; roe_success_peristop=[];
%             allstop_success=NaN(1,size(norm_base_mean,2));
%             for stamps=1:size(stop_success_tmpts,1)
%                 dop_success_peristop(stamps,:)= roibase_mean(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
%                 roe_success_peristop(stamps,:)= speed_smth_1(stop_success_tmpts(stamps)-pre_win_frames:stop_success_tmpts(stamps)+post_win_frames);
%                 %%% for plotting
%                 allstop_success(stop_success_tmpts(stamps,1):stop_success_tmpts(stamps,2))=1;
%             end
%             
%             norm_dopsuccess_peristop= dop_success_peristop./mean( dop_success_peristop(:,1:pre_win_frames),2);
%             norm_roesuccess_peristop= roe_success_peristop./mean( roe_success_peristop(:,1:pre_win_frames),2);
%             %%%%plotting everything
%             subplot(4,4,(plane-1)*4+3),plot(xax,mean(norm_dopsuccess_peristop,1),color_cod,'LineWidth',2);hold on
%             legend(['n = ',num2str(size(norm_dopsuccess_peristop,1))])
%             %         plot([0 0],)
%             plot([0 0],[min(ylim) max(ylim)],'k--')
%             plot(xax,rescale(mean(norm_roesuccess_peristop,1),0.99,1),'k','Linewidth',2)
%             title('stop initiation')
%             
%             
%             subplot(4,4,(plane-1)*4+4),plot(xax,mean(norm_dopsuccess_perimov,1),color_cod,'LineWidth',2);hold on
%             legend(['n = ',num2str(size(norm_dopsuccess_perimov,1))])
%             plot([0 0],[min(ylim) max(ylim)],'k--')
%             plot(xax,rescale(mean(norm_roesuccess_perimov,1),0.99,1),'k','Linewidth',2)
%             title('moving initiation')
%             %%%1 reward 2 stop 3 motion
%             stack_peri(days,cnt,:,1)=mean(norm_roisingle_traces,2);
%             stack_peri(days,cnt,:,2)=mean(norm_dopsuccess_peristop,1);
%             stack_peri(days,cnt,:,3)=mean(norm_dopsuccess_perimov,1);
%             
%             speedstack_peri(days,cnt,:,1)=mean(roisingle_traces_roesmth,2);
%             speedstack_peri(days,cnt,:,2)=mean(roe_success_peristop,1);
%             speedstack_peri(days,cnt,:,3)=mean( roe_success_perimov,1);
%             
%             
%             
%         end
%         
%     end
%     %     pause
% end
% % for kk=1:12
% %         find_figure('merge'),subplot(3,4,kk), 
% %         S=stackedplot(xax,squeeze(stack_peri(kk,:,:,1))','Title','E148:RR')
% %         ax = findobj(S.NodeChildren, 'Type','Axes');
% % end
% e148_anat=[{'ROE'},{'SR/SP-P1'},{'SP2'},{'SO/SP-P2'},{'SO/SP-P3'},{'SP3'},{'SO/SP4'}];
% % e149_anat=[{'ROE'},{'SR/SP-P1'}, {'SR/SP-P1'},{'SP2'},{'SR/SP-P2'},{'SO/SP-P3'},{'SO-P4'}];
% 
% find_figure('roisplit'); clf
% title_label={'perireward','stop initiation','moving initiation'};
% mat=[0 10; 0.99 1.02; 0.995 1.005; 0.995 1.005]%% E148
% % mat=[-5 50; 0.99 1.01; 0.995 1.005; 0.995 1.005]%% E149
% 
% % mat2=[0 10; 0.995 1.005; 0.998 1.002; 0.999 1.003]
% 
% for ll=1:3
%     for kk=1:7
%         if kk==1
%             
%             find_figure('roisplit'); ax=subplot(7,3,ll),
%             imagesc(squeeze(speedstack_peri(:,1,:,ll))); colormap(ax,gray), axis image
%             title(title_label{ll}); hold on
%             plot([40 40],[1 12],'--k','Linewidth',2)
%             set(gca,'xtick',[1 40 79], 'xticklabel',{'-5','0','5'})
%             
%         else
%         find_figure('roisplit'); ax2=subplot(7,3,(kk-1)*3+ll), imagesc(squeeze(stack_peri(:,kk-1,:,ll))); hold on
%         title(e148_anat{kk})
%         plot([40 40],[1 12],'--k','Linewidth',2)
%         caxis(mat(ll+1,:))
%         colormap(ax2,'jet')
%         xlabel('first lick after reward')
%         ylabel('Days')
%         set(gca,'xtick',[])
%         c=get(ax2,'position')
%         subplot('Position', [c(1)+0.22 c(2)-0.006 c(3)-0.18 c(4)+0.009]);
%         plot(mean(squeeze(stack_peri(:,kk-1,40:56,ll)),2))
%         hold on
%         plot(rescale(mean(squeeze(speedstack_peri(:,kk-1,40:56,ll)),2),0.999,1.002),'k--')
% 
%         camroll(270)
% %         set(gca,'ylim',[mat2(ll+1,:)])
% %         set(gca,'Visible','off')
%         end
%         
%         
%     end
% end
% 
% 
% 
% % %%%e148
% % idx=[1 2 4 3 6 5];%%%e156
% % % idx=[1 2 3 4 5];%% e157
% % % idx=[1 2 3 4 5];
% % e156_anat=[{'SRSP-P1'},{'SP2'},{'SOSR-P2'},{'SOSP-P3'},{'SP3'},{'SOSP4'}];
% % newylabel={'SOSP-P4','SOSP-P3','SP3','SP2','SRSP-P3','SRSP-P2','SRSP-P1'};
% % 
% % squeeze()
% % 
% % 
% % 
% % for kk=1:12
% %     find_figure('merge'),subplot(3,4,kk), stackedplot()
% %     S=stackedplot(xax,squeeze(stack_peri(kk,:,:,1))','Title','E148:RR','DisplayLabels',newylabel)
% %     ax = findobj(S.NodeChildren, 'Type','Axes');
% %     
% %     set(ax, 'YLim', [min(min(perireward_new(:,1:end-1))),max(max(perireward_new(:,1:end-1)))])
% %     for col=1:7%% 6 for e157 and 7 fro e156
% %         S.LineProperties(col).Color = mycolorse156(col,:);
% %     end
% %     xlabel('First rewarded lick Time(s)')
% %     set(ax(1), 'YLim', [0.997,1.001])
% %     
% %     
% % end
