% generate stim artefact binary
% generates a binary that is 0 where you did not stimulate and 1 where you
% did and appends to all params
% must select the folder containing the original tiffs and must have done
% all the way down to batch sp so part 1
% will append binary to params.

close all
mouse_id=195;
pr_dir=uipickfiles;
days_check=1:length(pr_dir);
tic
stimdifthresh = 70;
stimzone = 25:50; %rows of pixels down the stim artefact goes (or less)


 for days=days_check
     dir_s2p = struct2cell(dir([pr_dir{days} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
     days
     cd(pr_dir{days})
      list_tif=dir('*.tif');
        filenames={list_tif(:).name};
        checks=strfind(filenames,('plane'));
        plane_tif=cellfun(@(x) ~isempty(x), checks);
        ntif=find(plane_tif~=1);
        myfilename = (list_tif(ntif(1)).name);
        info=imfinfo(myfilename);
        M=info(1).Width;
        N=info(1).Height;
        stimareamean=[];
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
            stimareamean=[stimareamean; squeeze(nanmean(nanmean(chone_temp(stimzone,:,:))))];
                   clearvars chone_temp
        end
        stims = zeros(size(stimareamean));
        for allplanes =1:size(planefolders,2)
            temp = allplanes:size(planefolders,2):length(stimareamean);
            temp2 = find(diff(stimareamean(temp))>stimdifthresh)+1;
            temp2(find(abs(diff(temp2))<3)) = [];
            stims(temp(temp2)) = 1;
        end
        stims(1:10) = 0;
        temp3 = find(stims);
        stims(temp3(find(diff(temp3)<200))) = 0;
        for allplanes = 1:size(planefolders,2)
            
                    pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
 
        %%% grab 1st file/initial 3000 frames
        cd(pr_dir2)
        save('params','stims','-append')
        figure;
        plot(rescale(stimareamean));
        hold on
        plot(stims)
        saveas(gcf,'StimulationPoints.fig')
        close all
        end

 end