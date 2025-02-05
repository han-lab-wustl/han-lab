function loadVideoTiffNoSplit_EH2_new_sbx_uint16Opto1_stims_db_mode(varargin)
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

%ZD added for Gerardo's workstation
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\jar\ij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\jar\mij.jar'

if nargin<2
    [filename,filepath]=uigetfile('*.sbx','Choose SBX file');
    dir='bi new';
elseif nargin<3
filepath=varargin{1};
filename=varargin{2};
dir='bi new';
else
filepath=varargin{1};
filename=varargin{2};
dir=varargin{3};  
%threshold = varargin{4};
end
% win=60*Fs;

cd (filepath); %set path
stripped_filename=regexprep(filename,'.sbx','');
z = sbxread(stripped_filename,1,1);
global info;
% chone = sbxread(stripped_filename,0,info.max_idx+1);
% chone = single(squeeze(chone)); %WHY DO THIS??? maybe keep as uint16 or change to double
% chone = (squeeze(chone)); %WHY DO THIS??? maybe keep as uint16 or change to double
% framenum=size(chone,3);

MIJ.start;    %calls Fiji

%%
numframes = info.max_idx+1;
% lims(1)=min(chone(:))   %used in moi's version for scaling image
% lims(2)=max(chone(:))
% lims=double(lims); %lims = [min max] pixel values of chone
lenVid=3000;
stims = [];
temps = [];
for ii=1:ceil(numframes/lenVid) %splitting into 3000 frame chunks. ii=1:number of files
% ii=1;
    if ii>9
        currfile=strcat(stripped_filename,'_x',num2str(ii),'.mat');
    else
        currfile=strcat(stripped_filename,'_',num2str(ii),'.mat');
    end
    if strcmp(dir,'uni') || strcmp(dir,'uni new')
%         chtemp=chone(:,:,((ii-1)*lenVid+1):min(ii*lenVid,length(chone)));
        chtemp=sbxread(stripped_filename,((ii-1)*lenVid),min(lenVid,(numframes-((ii-1)*lenVid)))); %read 1st frame,#of frames to read. takes min of either lenVid or remainder of frames for last chunk
        chtemp=double(squeeze(chtemp));  %need as double for operation below.
        chtemp=chtemp(:,45:730,:);
%         chtemp=chone(:,:,((ii-1)*lenVid+1):min(ii*lenVid,length(chone)));
    elseif strcmp(dir,'bi') %cropping off different amounts for different video types
%         chtemp=sbxread(stripped_filename,((ii-1)*lenVid+1),lenVid);
        chtemp=sbxread(stripped_filename,((ii-1)*lenVid),min(lenVid,(numframes-((ii-1)*lenVid))));
        chtemp=double(squeeze(chtemp));
        chtemp=chtemp(:,110:721,:); %moi's value, not sure this is right but not many bi old movies
%         chtemp=chone(:,110:721,((ii-1)*lenVid+1):min(ii*lenVid,length(chone)));
    elseif strcmp(dir,'bi new')
%         chtemp=chone(:,110:709,((ii-1)*lenVid+1):min(ii*lenVid,length(chone)));
        chtemp=sbxread(stripped_filename,((ii-1)*lenVid),min(lenVid,(numframes-((ii-1)*lenVid))));
        chtemp_original=double(squeeze(chtemp));
%         choptotemp = repmat((nanmean(chtemp(:,750:end,:),2)),1,size(chtemp,2),1);
% %         chtemp=chtemp(:,90:730,:);
%         chtemp=chtemp(:,90:718,:)-choptotemp(:,90:718,:);
        choptotemp = repmat((nanmean(chtemp_original(:,740:end,:),2)),1,size(chtemp_original,2),1);
        %         chtemp=chtemp(:,90:730,:);
        %chtemp=chtemp_original(110:end,125:718,:)-choptotemp(110:end,125:718,:); % zd added option to crop etl
        chtemp=chtemp_original(1:20,:,:)-choptotemp(1:20,:,:); % zh have this for only have etl

    end
    
   chtemp=(((double(chtemp))/2)-1); %make max of movie 32767 (assuming it was 65535 before)
   chtemp=uint16(chtemp);
   temps =  [temps squeeze(nanmean(squeeze(nanmean(chtemp_original(1:20,:,:)))))];
  
%    stims = [stims; tempstims];
   

%     chtemp=double(chtemp);

    imageJ_savefilename=strrep([filepath,'\',currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
    imageJ_savefilename=['path=[' imageJ_savefilename ']'];
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
    MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
%     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
    MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
    MIJ.run('Close All');
end


 %   for p = 1:size(info.etl_table,1)
 %       currx = p:size(info.etl_table,1):length(temps);
 %       temp2 = (abs(temps(currx)/nanmean(temps(currx))-1));
 %       s = find(temp2>0.05);
 %       if ~isempty(s)
 %       tempstims(currx(s)) = 1;
 %       end
 %   end

 tempstims = zeros(length(temps),1);
 numStdDevs = 2; 
for p = 1:size(info.etl_table,1)
    currx = p:size(info.etl_table,1):length(temps);
    temps_currx = temps(currx);
    currxMean = mean(temps_currx);
    currxStd = std(temps_currx);
    threshold = currxMean + numStdDevs * currxStd;
    s = find(temps_currx>threshold);
    if ~isempty(s)
        tempstims(currx(s)) = 1;
    end
end
   if ii == 1
       tempstims(1:10) = 0;
   end
stims = [stims; tempstims];
save([stripped_filename '.mat'],'stims','-append')
MIJ.exit;

% 
% clear chone;

