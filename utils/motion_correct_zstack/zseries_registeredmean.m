filename = 'G:\zseries\E130\200914_EH_PV_hi_speed_002_000_1.tif';
info = imfinfo(filename);
video = [];
for frame = 1:length(info)
video(:,:,frame) = imread(filename,frame);
end
%% register the 40 frames of the same plane and take the average
registerr = 0;
registeredmean = [];
if registerr == 1
for meanframe = 1:length(info)/40
    temp2 = [];
    for i = fliplr(1:39)
    temp1 = imregcorr(squeeze(video(:,:,meanframe*40-i)),squeeze(video(:,:,meanframe*40)));
    rfixed = imref2d(size(squeeze(video(:,:,meanframe*40))));
    temp2(:,:,i) = imwarp(squeeze(video(:,:,meanframe*40-i)),temp1,'OutputView',rfixed);
    end
    temp2(:,:,40) = squeeze(video(:,:,meanframe*40));
    registeredmean(:,:,meanframe) = squeeze(nanmean(temp2,3));
end
else
for meanframe = 1:length(info)/40
    registeredmean(:,:,meanframe) = squeeze(nanmean(video(:,:,meanframe*40-39:meanframe*40),3));
end
end

%% register across slices

planeregisteredmean = registeredmean;
for z = 2:size(registeredmean,3)
    temp1 = imregcorr(squeeze(planeregisteredmean(:,:,z)),squeeze(planeregisteredmean(:,:,z-1)));
    rfixed = imref2d(size(squeeze(planeregisteredmean(:,:,z-1))));
    planeregisteredmean(:,:,z) = imwarp(squeeze(planeregisteredmean(:,:,z)),temp1,'OutputView',rfixed);
end



%% save registered mean version
javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\ij-1.52p.jar'
javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\mij.jar'
MIJ.start;    %calls Fiji

chtemp=registeredmean;
chtemp=uint16(chtemp);
filepath = 'F:\E195\230315_GM\230315_GM_001_000';
currfile = '230315_GM_001_000_1_registeredmean.mat';

imageJ_savefilename=strrep([filepath,'\',currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
    imageJ_savefilename=['path=[' imageJ_savefilename ']'];
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
    MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
%     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
    MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
    MIJ.run('Close All');
MIJ.exit;

%% save registered mean version Plane Version
javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\ij-1.52p.jar'
javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\mij.jar'
MIJ.start;    %calls Fiji

% chtemp=planeregisteredmean;
chtemp = test1;
chtemp=uint16(chtemp);
filepath = 'G:\zseries\E130';
currfile = '200914_EH_PV_hi_speed_002_000_SPhighlighted2_registeredAcrossPlanes.mat';

imageJ_savefilename=strrep([filepath,'\',currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
    imageJ_savefilename=['path=[' imageJ_savefilename ']'];
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
%     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
    MIJ.createImage('chone_image', chtemp, true); %creates ImageJ file with 'name', matlab variable name
%     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
    MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
    MIJ.run('Close All');
    MIJ.exit;
