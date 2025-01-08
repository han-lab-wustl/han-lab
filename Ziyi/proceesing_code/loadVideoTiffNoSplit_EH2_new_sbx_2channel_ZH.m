function loadVideoTiffNoSplit_EH2_new_sbx_2channel(varargin)
% modified from moi's version. 
% 200329. EH.
% 200501 EH. changed java path for this computer
% 221012 GM. changed to read 2 channels by creating two tiffs for the
% channels and combining them.calso changed line 76 to crop the correct dimensions now that it is a two
% channel also change to imagej save file name.
% 240805 ZH. Changed to saving 2 channels seperately.
% Uses sbxread to load chunks of .sbx files, limiting RAM requirements
% for some reason, suite2p turns everything into unsigned 16bit with max
% intensity value of 32767 (after motion correction). think this will actually clip high end of
% values. moi's code scaled all based on max/min of entire movie. current
% version just divides intensity value by 2 and subtracts 1 to get max value of 32767 (makes
% assumption that max value in movie is 65535)
if nargin<2
    [filename,filepath]=uigetfile('*.sbx','Choose SBX file');
    dir='uni';
elseif nargin<3
filepath=varargin{1};
filename=varargin{2};
dir='uni';
else
filepath=varargin{1};
filename=varargin{2};
dir=varargin{3};    
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

% javaaddpath 'C:\Program Files\MATLAB\R2017a\java\mij.jar' %default
% javaaddpath 'C:\Program Files\MATLAB\R2017a\java\ij.jar'
% javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\ij-1.52p.jar'
% javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\mij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\jar\ij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2023b\java\jar\mij.jar'
MIJ.start;    %calls Fiji

%%
numframes = info.max_idx+1;
% lims(1)=min(chone(:))   %used in moi's version for scaling image
% lims(2)=max(chone(:))
% lims=double(lims); %lims = [min max] pixel values of chone
lenVid=3000;

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
        chtemp=double(squeeze(chtemp));
%         chtemp=chtemp(:,90:730,:);
        chtemp=chtemp(:,:,90:718,:);
                %chtemp=chtemp(:,:,90:718,:);
    end
    
   chtemp=(((double(chtemp))/2)-1); %make max of movie 32767 (assuming it was 65535 before)
   chtemp=uint16(chtemp);
   
   
%     chtemp=double(chtemp);

    imageJ_savefilename1=strrep([filepath,'\','green_opto_corrected_tifs','\',currfile(1:end-4),'_green.tif'],'\','\\'); %ImageJ needs double slash
    [status, msg, msgID] = mkdir(fullfile(filepath, 'green_opto_corrected_tifs'));
    imageJ_savefilename1=['path=[' imageJ_savefilename1 ']'];
    imageJ_savefilename2=strrep([filepath,'\','red_opto_corrected_tifs','\',currfile(1:end-4),'_red.tif'],'\','\\'); %ImageJ needs double slash
    imageJ_savefilename2=['path=[' imageJ_savefilename2 ']'];
    [status, msg, msgID] = mkdir(fullfile(filepath, 'red_opto_corrected_tifs'));
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
%     MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
    MIJ.createImage('chone_image1',uint16(squeeze(chtemp(1,:,:,:))),true); 
    MIJ.run('Save', imageJ_savefilename1);   %saves with defined filename
    MIJ.run('Close All');
    MIJ.createImage('chone_image2',uint16(squeeze(chtemp(2,:,:,:))),true);
    MIJ.run('Save', imageJ_savefilename2);   %saves with defined filename
    MIJ.run('Close All');
end
MIJ.exit;
% 
% clear chone;

