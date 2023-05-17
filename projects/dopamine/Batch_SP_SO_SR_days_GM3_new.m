%%%%% batch processing
%%%lets plot mean image of first 500 frames across days to analyze
%%%parent directory
%%% DOPAMINE MM
%%ROI selection and the extraction of raw flourescence


%%

clear all
close all
mouse_id=194;
pr_dir=uipickfiles;
days_check=1:length(pr_dir);
ref_exist=1;%%% if reference image hase been already choosen
if ref_exist
    pr_dirref=uipickfiles;%%% chose reference day here day1
end
%%
for allplanes=1:3
    for days=days_check
        pr_dir1=strcat(pr_dir{days},'\')
        %%%s2p directory
        pr_dir2=dir(fullfile(pr_dir1,"**",sprintf("plane%i",allplanes-1),'\reg_tif'));
        pr_dir2 = pr_dir2(1).folder;
        % ZD changed this in case you have a nested folder structure,
        % should work even if not
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
        %ZD CROP IS DIFFERENT
%         crop_points1=[41 169 619 512]; %%% Direct .sbx crop
%         
%         eval(['x1=crop_points' num2str(1) '(1)']);  %x for area for correction
%         eval(['x2=crop_points' num2str(1) '(3)']);
%         eval(['y1=crop_points' num2str(1) '(2)']);
%         eval(['y2=crop_points' num2str(1) '(4)'])
%         
%         M=size(chone_temp,2);
%         N=size(chone_temp,1);
%         chone_temp=chone_temp(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);
        
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
        pr_dir4ref=dir(fullfile(pr_dirref{1},"**",sprintf("plane%i",allplanes-1),'\reg_tif\'));
        pr_dir4ref= pr_dir4ref(1).folder;        
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
        pr_dir4=dir(fullfile(pr_dir3,"**",sprintf("plane%i",allplanes-1),'\reg_tif'));
        pr_dir4 = pr_dir4(1).folder;
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
            
            w=input('input vals 1 to keep drawing or 0 to move to next subplot: ');
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
        pr_dir_s=dir(fullfile(pr_dir_ns,"**",sprintf("plane%i",allplanes-1),'\reg_tif'));
        pr_dir_s = pr_dir_s(1).folder;
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
% clear all
% close all
% pr_dir=uipickfiles;
days_check=1:length(pr_dir);
tic
for allplanes=1:3 %1:4
    for days=days_check        
        pr_dir1=strcat(pr_dir{days},'\')
        %%%s2p directory
        pr_dir2=dir(fullfile(pr_dir1,"**",sprintf("plane%i",allplanes-1),'\reg_tif')); 
        % ZD changed this in case you have a nested folder structure,
        % should work even if not
        pr_dir2 = pr_dir2(1).folder;
        %%% grab 1st file/initial 3000 frames
        cd(pr_dir2)
        
        load('params')
        list_tif=dir('*.tif');
        %ZD ADDED TO FIX FILE SORTING ISSUE IN SUITE2P NEW VERSION
        for i=1:length(list_tif)
            if strcmp(list_tif(i).name,'file500_chan0.tif') % only messes with this file which seems to be the problem for me
                % may want to change this if it's different based on num of
                % frames for different people
                rename = 'file0500_chan0.tif';
                movefile(fullfile(list_tif(1).folder, list_tif(i).name), fullfile(list_tif(1).folder, rename))
            elseif strcmp(list_tif(i).name,'file000_chan0.tif')
                rename = 'file0000_chan0.tif';
                movefile(fullfile(list_tif(1).folder, list_tif(i).name), fullfile(list_tif(1).folder, rename))
            end
        end
        list_tif=dir('*.tif'); % re list
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
            jj
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
        
%         crop_points1=[41 169 619 512]; %%% Direct .sbx crop
%         
%         eval(['x1=crop_points' num2str(1) '(1)']);  %x for area for correction
%         eval(['x2=crop_points' num2str(1) '(3)']);
%         eval(['y1=crop_points' num2str(1) '(2)']);
%         eval(['y2=crop_points' num2str(1) '(4)'])
%         
%         M=size(chone,2);
%         N=size(chone,1);
%         chone=chone(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);
%         
        
        %         toc
        
        clear chone_temp  roim_sel roibase_mean roibase_mean2 roibase_mean3 roibase_mean4
        %%%% extract df/f for an roi
        load('params.mat')
        for jj=1:size(params.newBWmask,1)
            if ~isempty(params.newBWmask{jj,1})
                masksel=params.newBWmask{jj,1};
                roim_sel=chone.*masksel;
                roim_sel(roim_sel==0)=NaN;
                mean_roi=squeeze(nanmean(nanmean(roim_sel(:,:,:,:),1)));
                roiraw_mean{jj,1} = mean_roi;
                
                %%%%% Baseline correction each roi
                
                
                tic
                %%%
                for deg=1:4
                    roibase_mean{jj,deg}=detrend(mean_roi,deg);%%%detrend
                    
                    %%% munni method 3rd degree moving filtering
                    try
                        noise=sgolayfilt(mean_roi,deg,size(mean_roi,1)-1);
                    catch % ZD added for odd error?
                        noise=sgolayfilt(mean_roi,deg,size(mean_roi,1));
                    end
                    bz_sig=mean_roi-noise;
                    roibase_mean4{jj,deg}=bz_sig;
                end
                
                
                toc
                find_figure('base_mean_rois');
                if jj==1
                    subplot(4,1,1),imagesc(squeeze(mean(chone,3))); colormap(gray), axis image
                end
                color_cod={'k','--k','-ok'};
                find_figure('base_mean_rois'); subplot(4,1,1)
                hold on;  plot(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),color_cod{jj},'Linewidth',2)
                
                subplot(4,1,jj+1),plot( roibase_mean{jj,1});
                %%%%%% basleine zeroing with old version
                junk=squeeze(nanmean(nanmean(roim_sel)));%mean of each image, frame x 1 vector
                numframes = size(chone,3);
                mean_all=nanmean(nanmean(nanmean(roim_sel)));
                base_window=200;
                junk2=zeros(size(junk));
                parfor kk=1:length(junk)
                    kk
                    cut=junk(max(1,kk-base_window):min(numframes,kk+base_window));
                    cutsort=sort(cut);
                    a=round(length(cut)*.08);
                    junk2(kk)=cutsort(a);
                end
                
                parfor i=1:numframes
                    i
                    roim_sel(:,:,i)=(roim_sel(:,:,i)/junk2(i))*mean_all;
                end
                
                
                roibase_mean3{jj,1}=squeeze(nanmean(nanmean(roim_sel(:,:,:),1)));
   
            end
            params.roibasemean2=roibase_mean;
            params.roibasemean3=roibase_mean3;
            params.roibasemean4=roibase_mean4;
            params.roirawmean2=roiraw_mean;
            
            
        end
        base_window=200;
        numframes = size(chone,3); %fix  to set slidingwindow limits to full video
        % old version
        tic
        junk=squeeze(nanmean(nanmean(chone)));%mean of each image, frame x 1 vector
        mean_all=nanmean(nanmean(nanmean(chone)));
        junk2=zeros(size(junk));
        for kk=1:length(junk)
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
        toc
        save('params','params','-append')
        %%%%  for checking
        %                 figure; subplot(3,1,1),plot(params.roirawmean2{1})
        %                 hold on
        %                 subplot(3,1,2),plot(params.roibasemean2{1})
        %                 subplot(3,1,3),plot(params.roibasemean3{1})
        
        
    end
    
end
toc

