function loadVideoTiffNoSplitOpto_2channel_woetl(varargin)
% ZD changed cropping parameters and some variables to crop out the etl
% ZD added 2 channel functionality 

%ZD added for Gerardo's workstation
javaaddpath 'C:\Program Files\MATLAB\R2022b\java\mij.jar' %added by TI 07/23/2024
javaaddpath 'C:\Program Files\MATLAB\R2022b\java\ij.jar'  %added by TI 07/23/2024
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
MIJ.start;    %calls Fiji

%%
numframes = info.max_idx+1;
% lims(1)=min(chone(:))   %used in moi's version for scaling image
% lims(2)=max(chone(:))
% lims=double(lims); %lims = [min max] pixel values of chone
lenVid=3000;
stims = [];
tempsg = []; % only collect stims from green pmt

for ii=1:ceil(numframes/lenVid) %splitting into 3000 frame chunks. ii=1:number of files
    % ii=1;
    if ii>9
        currfile=strcat(stripped_filename,'_x',num2str(ii),'.mat');
    else
        currfile=strcat(stripped_filename,'_',num2str(ii),'.mat');
    end
    chtemp=sbxread(stripped_filename,((ii-1)*lenVid),min(lenVid,(numframes-((ii-1)*lenVid))));
    chtemp_original=double(squeeze(chtemp));
    % split into green and red channel
    chtemp_original_g = squeeze(chtemp_original(1,:,:,:));
    chtemp_original_r = squeeze(chtemp_original(2,:,:,:));
    % artifact corr in both channels
    choptotempg = repmat((nanmean(chtemp_original_g(:,740:end,:),2)),1,size(chtemp_original_g,2),1);
    choptotempr = repmat((nanmean(chtemp_original_r(:,740:end,:),2)),1,size(chtemp_original_g,2),1);
    % subtract artifact
    chtempg=chtemp_original_g(110:end,125:718,:)-choptotempg(110:end,125:718,:); % zd added option to crop etl
    chtempr=chtemp_original_r(110:end,125:718,:)-choptotempr(110:end,125:718,:); % zd added option to crop etl
    % used to be: (:,90:718,:)
    % matlab order: x,y,z
    % chtemp original is with etl/opto artifact intact used to find
    % opto artifact
    tempsg =  [tempsg squeeze(nanmean(squeeze(nanmean(chtemp_original_g(1:20,:,:)))))];
    chtempg=(((double(chtempg))/2)-1); %make max of movie 32767 (assuming it was 65535 before)
    chtempg=uint16(chtempg);
    chtempr=(((double(chtempr))/2)-1); %make max of movie 32767 (assuming it was 65535 before)
    chtempr=uint16(chtempr);
    imageJ_savefilename1=strrep([filepath,'\','green_opto_corrected_tifs','\',currfile(1:end-4),'_green.tif'],'\','\\'); %ImageJ needs double slash
    [status, msg, msgID] = mkdir(fullfile(filepath, 'green_opto_corrected_tifs'));
    imageJ_savefilename1=['path=[' imageJ_savefilename1 ']'];
    imageJ_savefilename2=strrep([filepath,'\','red_opto_corrected_tifs','\',currfile(1:end-4),'_red.tif'],'\','\\'); %ImageJ needs double slash
    imageJ_savefilename2=['path=[' imageJ_savefilename2 ']'];
    [status, msg, msgID] = mkdir(fullfile(filepath, 'red_opto_corrected_tifs'));
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
%     MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
    MIJ.createImage('chone_image1',uint16(squeeze(chtempg)),true)  
    MIJ.run('Save', imageJ_savefilename1);   %saves with defined filename
    MIJ.run('Close All');
    MIJ.createImage('chone_image2',uint16(squeeze(chtempr)),true)
    MIJ.run('Save', imageJ_savefilename2);   %saves with defined filename
    MIJ.run('Close All');
end

tempstims = zeros(length(tempsg),1);
for p = 1:size(info.etl_table,1)
    currx = p:size(info.etl_table,1):length(tempsg);
    temp2 = (abs(tempsg(currx)/nanmean(tempsg(currx))-1));
    s = find(temp2>0.5);
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

