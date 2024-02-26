function [params] = extract_dff_from_ROI_dopamine(pr_dir)

% extract base mean from all rois selected before
% also calculates dff
% saves to params file in reg tif folder
days_check=1:length(pr_dir);
tic

for days=days_check
    dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
    for allplanes=1:size(planefolders,2) %1:4
        clearvars -except mouse_id pr_dir days_check days dir_s2p planefolders allplanes

        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')

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
            chone_temp = double(TIFFStack(myfilename));
            chone=cat(3,chone,chone_temp);

        end
        %         toc
        %%%%%

        crop_points1=[41 169 619 512]; %%% Direct .sbx crop
        %          crop_points1=[1 169 619 512]; %%% Direct .sbx crop

        eval(['x1=crop_points' num2str(1) '(1);']);  %x for area for correction
        eval(['x2=crop_points' num2str(1) '(3);']);
        eval(['y1=crop_points' num2str(1) '(2);']);
        eval(['y2=crop_points' num2str(1) '(4);'])

        M=size(chone,2);
        N=size(chone,1);
        chone=chone(max(y1,1):min(y2,N),max(x1,1):min(x2,M),:);


        %         toc

        clear chone_temp  roim_sel roibase_mean roibase_mean2 roibase_mean3 roibase_mean4
        %%%% extract df/f for an roi
        load('params.mat')
        for jj=1:size(params.newBWmask,1)
            if ~isempty(params.newBWmask{jj,1})
                masksel=params.newBWmask{jj,1};
                %                 if size(masksel,1)< size(chone,1)
                %                     masksel = [masksel; zeros(size(chone,1)-size(masksel,1),size(masksel,2))];
                %                 end
                %                 if size(masksel,2)< size(chone,2)
                %                     masksel = [masksel zeros(size(masksel,1),size(chone,2)-size(masksel,2))];
                %                 end
                roim_sel=chone.*masksel;
                roim_sel(roim_sel==0)=NaN;
                mean_roi=squeeze(nanmean(nanmean(roim_sel(:,:,:,:),1)));
                roiraw_mean{jj,1} = mean_roi;

                %%%%% Baseline correction each roi


                %                 tic
                %%%
                %                 for deg=1:4
                %                     roibase_mean{jj,deg}=detrend(mean_roi,deg);%%%detrend
                %
                %                     %%% munni method 3rd degree moving filtering
                % %                     noise=sgolayfilt(mean_roi,deg,size(mean_roi,1)-1);
                %                     bz_sig=mean_roi;%-noise;
                %                     roibase_mean4{jj,deg}=bz_sig;
                %                 end


                %                 toc
                find_figure('base_mean_rois');
                if jj==1
                    subplot(4,1,1),imagesc(squeeze(mean(chone,3))); colormap(gray), axis image
                end
                color_cod={'k','--k','-ok'};
                find_figure('base_mean_rois'); subplot(4,1,1)
                hold on;  plot(params.newroicoords{jj,1}(:,1),params.newroicoords{jj,1}(:,2),color_cod{jj},'Linewidth',2)

                subplot(4,1,jj+1),
                %                 plot( roibase_mean{jj,1});

                %%%%%% basleine zeroing with old version
                junk=squeeze(mean(mean(roim_sel,'omitnan'),'omitnan'));%mean of each image, frame x 1 vector
                numframes = size(chone,3);
                mean_all=mean(mean(mean(roim_sel,'omitnan'),'omitnan'),'omitnan');
                base_window=200;
                %                 junk2=zeros(size(junk));
                %                 parfor kk=1:length(junk)
                %                     kk
                %                     cut=junk(max(1,kk-base_window):min(numframes,kk+base_window));
                %                     cutsort=sort(cut);
                %                     a=round(length(cut)*.08);
                %                     junk2(kk)=cutsort(a);
                %                 end
                junk2 = movquant(junk,0.08,base_window,[],'omitnan','truncate');

                parfor i=1:numframes
                    i
                    roim_sel(:,:,i)=(roim_sel(:,:,i)/junk2(i))*mean_all;
                end


                roibase_mean3{jj,1}=squeeze(mean(mean(roim_sel(:,:,:),1,'omitnan'),'omitnan'));
                plot(roibase_mean3{jj,1})

            end
            %             params.roibasemean2=roibase_mean;
            params.roibasemean3=roibase_mean3;
            %             params.roibasemean4=roibase_mean4;
            params.roirawmean2=roiraw_mean;


        end
        base_window=200;
        numframes = size(chone,3); %fix  to set slidingwindow limits to full video
        % old version
        %         tic
        junk=squeeze(mean(mean(chone, 'omitnan'), 'omitnan'));%mean of each image, frame x 1 vector
        %         mean_all=nanmean(nanmean(nanmean(chone)));
        %         junk2=zeros(size(junk));
        %         for kk=1:length(junk)
        %             cut=junk(max(1,kk-base_window):min(numframes,kk+base_window));
        %             cutsort=sort(cut);
        %             a=round(length(cut)*.08);
        %             junk2(kk)=cutsort(a);
        %         end
        junk2 = movquant(junk,0.08,base_window,[],'omitnan','truncate');
        params.raw_mean = junk;

        parfor i=1:numframes
            chone(:,:,i)=(chone(:,:,i)/junk2(i))*mean_all;
        end
        params.base_mean = squeeze(mean(mean(chone, 'omitnan'), 'omitnan'));
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
end