% process z stacks for motion correct and mean across planes
% zahra edited for tiled images
% july 2024
clear all; close all
frame_num = 20;
pth = "X:\rna_fish_alignment_zstacks\240702\e217\240702_ZD_001_001";
fls = dir(fullfile(pth, 'tile*.tif'));
for fl=1:length(fls)
    % filename = "X:\zstacks\240701_ZD_001_004\tile_000.tif";
    clearvars filename
    filename = fullfile(fls(fl).folder, fls(fl).name);
    info = imfinfo(filename);
    video = [];
    for frame = 1:length(info)
        video(:,:,frame) = imread(filename,frame);
    end
    % register the 40 frames of the same plane and take the average
    registerr = 0;
    registeredmean = [];
    if registerr == 1
        for meanframe = 1:length(info)/frame_num
            temp2 = [];
            for i = fliplr(1:frame_num-1)
                temp1 = imregcorr(squeeze(video(:,:,meanframe*40-i)),squeeze(video(:,:,meanframe*frame_num)));
                rfixed = imref2d(size(squeeze(video(:,:,meanframe*frame_num))));
                temp2(:,:,i) = imwarp(squeeze(video(:,:,meanframe*frame_num-i)),temp1,'OutputView',rfixed);
            end
            temp2(:,:,frame_num) = squeeze(video(:,:,meanframe*frame_num));
            registeredmean(:,:,meanframe) = squeeze(nanmean(temp2,3));
        end
    else
        for meanframe = 1:length(info)/frame_num
            registeredmean(:,:,meanframe) = squeeze(nanmean(video(:,:,meanframe*frame_num-(frame_num-1):meanframe*frame_num),3));
        end
    end

    %register across slices

    planeregisteredmean = registeredmean;
    for z = 2:size(registeredmean,3)
        temp1 = imregcorr(squeeze(planeregisteredmean(:,:,z)),squeeze(planeregisteredmean(:,:,z-1)));
        rfixed = imref2d(size(squeeze(planeregisteredmean(:,:,z-1))));
        planeregisteredmean(:,:,z) = imwarp(squeeze(planeregisteredmean(:,:,z)),temp1,'OutputView',rfixed);
    end

    % save registered mean version as mat file
    chtemp=registeredmean;
    chtemp=uint16(chtemp);
    savenm = string(strcat(filename(1:end-4), '_registered.mat'));
    save(savenm, 'chtemp')
end

%% if want to save as tif
% %% save registered mean version
% javaaddpath 'C:\Program Files\MATLAB\R2023b\java\mij.jar'
% javaaddpath 'C:\Program Files\MATLAB\R2023b\java\ij.jar'
% MIJ.start;    %calls Fiji
%
% chtemp=registeredmean;
% chtemp=uint16(chtemp);
% filepath = "X:\zstacks\240701_ZD_001_004";
% currfile = "tile_000_registered.mat";
%
% imageJ_savefilename=strrep([filepath,'\',currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
%     imageJ_savefilename=['path=[' imageJ_savefilename ']'];
% %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
% %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
%     MIJ.createImage('chone_image', uint16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
% %     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
%     MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
%     MIJ.run('Close All');
% MIJ.exit;
%
% %% save registered mean version Plane Version
% javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\ij-1.52p.jar'
% javaaddpath 'C:\Users\Han Lab\Downloads\fiji-win64\Fiji.app\jars\mij.jar'
% MIJ.start;    %calls Fiji
%
% % chtemp=planeregisteredmean;
% chtemp = test1;
% chtemp=uint16(chtemp);
% filepath = 'G:\zseries\E130';
% currfile = '200914_EH_PV_hi_speed_002_000_SPhighlighted2_registeredAcrossPlanes.mat';
%
% imageJ_savefilename=strrep([filepath,'\',currfile(1:end-4),'.tif'],'\','\\'); %ImageJ needs double slash
%     imageJ_savefilename=['path=[' imageJ_savefilename ']'];
% %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,[0 32767])), true);
% %     MIJ.createImage('chone_image', gray2ind(mat2gray(chtemp,double(lims)),double(round(ceil(lims(2))/2))), true);
%     MIJ.createImage('chone_image', chtemp, true); %creates ImageJ file with 'name', matlab variable name
% %     MIJ.createImage('chone_image', int16(chtemp), true); %creates ImageJ file with 'name', matlab variable name
%     MIJ.run('Save', imageJ_savefilename);   %saves with defined filename
%     MIJ.run('Close All');
%     MIJ.exit;
%
