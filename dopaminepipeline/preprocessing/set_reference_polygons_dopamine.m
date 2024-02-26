function set_reference_polygons_dopamine(pr_dir, pr_dirref, days_check,ref_exists)

dir_s2p = struct2cell(dir([pr_dir{1} '\**\suite2p']));
planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));

for allplanes=1:size(planefolders,2)
    for days=days_check
        dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
        planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
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
        %         for i=1:numframes
        %             chone_temp(:,:,i)=imread(myfilename,i,'Info',info);
        %         end
        chone_temp = double(TIFFStack(myfilename));

        crop_points1=[41 169 619 512]; %%% Direct .sbx crop
        %         crop_points1=[1 169 619 512]; %%% Direct .sbx crop

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
    if isempty(ref_exists)
        %         pr_dir3=strcat(pr_dir,num2str(mouse_id),'\Day_',num2str(ref_day),'\');
        %%%s2p directory
        dir_s2p = struct2cell(dir([pr_dirref{1} '\**\suite2p']));
        planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir4ref=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\');

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
        dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
        planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));

        pr_dir4=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
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

            w=input('1 to keep drawing or ENTER to move to next subplot: ');
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
            fprintf('******** \n MOVE ROI TO FINAL POSITION (drag polygon slightly so it registers with MATLAB!) \n ********')
            w2=input('ROI position final? Hit ENTER; to change ROI type 1: ')

            if length(w2)==0
                h1=get(gca,'title');
                mpimg=mimg(:,:,allplanes,days_check(ndays));
                newpos{fk,1}=str2num(h1.String);
                BWnewpos{fk,1}=poly2mask(newpos{fk,1}(:,1),newpos{fk,1}(:,2),size(params.mimg,1),size(params.mimg,2));
            end

        end
        dir_s2p = struct2cell(dir([pr_dir{roi_sel} '\**\suite2p']));
        planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir_s=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        cd(pr_dir_s)
        params.mimg=mpimg;
        params.newroicoords=newpos;
        params.newBWmask=BWnewpos;
        save('params.mat','params')
        w=input('1 to keep drawing or ENTER to move to next subplot: ');
    end
end
end