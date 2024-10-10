function loadVideoTiffNoSplit_Opto(src, days, lenVid, threshold)
%called by "runVideosTiff_EH_new_sbx_uint16"
%version that doesn't divide by 2 for non-suite2p analysis
% modified from moi's version.
% 200329. EH.
% 200501 EH. changed java path for this computer
% also change to imagej save file name.
% Uses sbxread to load chunks of .sbx files, limiting RAM requirements
% for some reason, suite2p turns everything into unsigned 16bit with max
% intensity value of 32767 (after motion correction). think this will actually clip high end of
% values. moi's code scaled all based on max/min of entire movie. current
% version just divides intensity value by 2 and subtracts 1 to get max value of 32767 (makes
% assumption that max value in movie is 65535)
% uses average subtraction for opto artifact

% %ZD changed for her workstation
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\ij.jar'

MIJ.start;    %calls Fiji

for day=days
    filename = dir(fullfile(src, string(day), '*2*', '*.sbx'));
    cd (filename.folder); %set path
    stripped_filename=fullfile(filename.folder,filename.name);
    stripped_filename= strtok(stripped_filename,'.');
    z = sbxread(stripped_filename,1,1);
    global info;
    % chone = sbxread(stripped_filename,0,info.max_idx+1);
    % chone = single(squeeze(chone)); %WHY DO THIS??? maybe keep as uint16 or change to double
    % chone = (squeeze(chone)); %WHY DO THIS??? maybe keep as uint16 or change to double
    % framenum=size(chone,3);

    %%
    numframes = info.max_idx+1;
    % lims(1)=min(chone(:))   %used in moi's version for scaling image
    % lims(2)=max(chone(:))
    % lims=double(lims); %lims = [min max] pixel values of chone
    stims = [];
    temps = [];
    for ii=1:ceil(numframes/lenVid) %splitting into 3000 frame chunks. ii=1:number of files
        % ii=1;
        if ii>9
            currfile=strcat(stripped_filename,'_x',num2str(ii),'.mat');
        else
            currfile=strcat(stripped_filename,'_',num2str(ii),'.mat');
        end
        chtemp=sbxread(stripped_filename,((ii-1)*lenVid),min(lenVid,(numframes-((ii-1)*lenVid))));
        chtemp_original=double(squeeze(chtemp));
        choptotemp = repmat((nanmean(chtemp_original(:,740:end,:),2)),1,size(chtemp_original,2),1);
        %         chtemp=chtemp(:,90:730,:);

        chtemp=chtemp_original(150:end,125:718,:)-choptotemp(150:end,125:718,:); % zd added option to crop etl
        % used to be: (:,90:718,:)
        % matlab order: x,y,z
        % chtemp original is with etl/opto artifact intact used to find
        % opto artifact

        chtemp=(((double(chtemp))/2)-1); %make max of movie 32767 (assuming it was 65535 before)
        chtemp=uint16(chtemp);
        % update stims
        temp =  squeeze(nanmean(squeeze(nanmean(chtemp_original(1:20,:,:)))));
        tempstims = zeros(length(temp),1);
        for p = 1:size(info.etl_table,1)
            currx = p:size(info.etl_table,1):length(temp);
            temp2 = (abs(temp(currx)/nanmean(temp(currx))-1));
            s = find(temp2>threshold);
            if ~isempty(s)
                tempstims(currx(s)) = 1;
            end
        end
        if ii == 1
            tempstims(1:10) = 0;
        end
        stims = [stims; tempstims];

        imageJ_savefilename=strrep([currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
        imageJ_savefilename=['path=[' imageJ_savefilename ']'];
        %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
        %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
        MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
        %     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
        MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
        MIJ.run('Close All');
    end

    save([stripped_filename '.mat'],'stims','-append')

end
MIJ.exit;

%
% clear chone;

