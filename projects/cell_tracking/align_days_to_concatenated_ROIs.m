% ***need to fix all_F output, if any 0s, set entire row to NaN


% For concatenating ROIs across days
% 210702 Notes on data pipeline
% run "runVideosTiff_EH_new_sbx_setCrop_planes" to crop FOVs (important for limiting size) and split planes 
% I crop aggressively for each plane. if we lose some cells at edges, that's ok
% calls "loadVideoTiffNoSplit_EH2_new_sbx_setCrop_planes"
% then this file, which also allows you to check alignments

% Aligns cell ROIs across days based on aligning single suite2p day to all days concatenated suite2p data
% Uses individual day ROI fits which should have better S/N than across day ROIs
% 1st, make tifs for each day and plane, then into folders "d1" to "d(x)"
% Within each day folder split planes to folders "d1\p1" to "d1\p4"
% This is important because you don't want to run suite2p on multiple planes at once, just 1!
% Run suite2p on each day and plane individually. 
% not sure we need to save bin or run two step registration for these. however, not tested
% The original folder and subfolders, "d1\p1", etc for day 1, plane1
% have the single day/single plane analyses.
% I do minimal curation on these, just a quick check over.
% Do this for all days and planes. Can do batch mode.
% 
% Make a concatenated suite2p analysis. only one plane at a time. 
% for ex, make "d1-d11_p1" folder.
% Then run suite2p by adding all the individual day, p1 folders to the path
% set the save path to the d1-d11_p1 folder.
% This registers all the tifs from the individual folders together
% This is used as a template to align individual days. the actual dF is not used for data.
% Spend time to check that your parameters are optimal for motion corr and aligning movies.
% two step registration helps and save bin etc to look at motion corr
% In particular "maxregshiftNR" may need to be changed. 
% on CA1 movies across 11 days, i found 10 pix to be optimal. assume this will vary
% as always it's good to run on a shorter test data set to optimize parameters
% other default settings, threshold 0.9, max_iterations 30. gets lots of rois, not all good
% after running, move "suite2p" folder of concatenated movie (will be in "d1\p1" to e.g. "d1-11_p1"
% Check quality and curate these ROIs carefully
% should have individual days and curated concatenated suite2p for all planes
% 
% this script asks you number of days to align and the locations of concatenated and individual day files
% it will save this data in a file list in case you want to re-run the analysis

%then goes through and selects best fits to template based on F, plots matrix and r vs. cent_dist.
% can look at template vs single day ROIs and traces, or skip
% if centroid dist is too far, writes as NaN
%saves file list and file
%moved code to assess ROI fits to separate script that can be run later
%"template_to_single_day_ROI_comparison", however not working


% Saves data in "Fall_xd_align.mat" in suite2p folder of concatenated movie
% file list also save here

close all
clear all

%variables for 
max_cent_dist=20;%maximum centroid dist in pixels, should convert to um based on mag
% in test experiment, was 2X maxshift for non rigid registration (10pix)
min_r=0.2; %min r value of matching roi. probably doesn't matter much
% in testing, centroid distance was the most critical variable for finding borderline matches
% at low r values and distances, may or may not be same cell. 
% if it's the same cell, very little signal anyway

answer = questdlg('Do you have a saved file list?',	'Loading F files and paths','Yes','No','No');
switch answer
    case 'Yes'
        [pre_list,pre_path]=uigetfile('*_file_list.mat','Select file list');
        cd (pre_path); %set path
        load(pre_list);%in concat day folder
    case'No'
        prompt = {'Number of Files:'};
        dlgtitle = 'Days to align';
        dims = [1 35];
        definput = {'5'}; %default files
        answer = inputdlg(prompt,dlgtitle,dims,definput);
        numDays=str2num(answer{1});
        single_files=cell(numDays,1);
        single_paths=cell(numDays,1);

        [all_file,all_path]=uigetfile('*.mat','All days concatenated file ');
        cd (all_path); %set path

        for j=1:numDays
            [single_files{j,1},single_paths{j,1}]=uigetfile('*.mat',['Single day analysis, Day ',num2str(j)]);
            cd (single_paths{j,1}); %set path
        end
end
cd (all_path); %set path
templ_all=load('Fall.mat');%in concat day folder
F_all=templ_all.F';
rois = [1:size(F_all',1)]';
%ignore iscell distinguish, zd edit 
%find(templ_all.iscell(:,1));%Nx1 vector with index of iscell=1, template cell s
%preallocate, variables below filled at end of loop
frames=zeros(numDays,1);%keep track of #frames per day
dayStart=zeros(numDays,1);%keep track 1st frame of days
dayEnd=zeros(numDays,1);%keep track of last frame of days
%below filled each loop, prior to filtering for bad fits
all_M=zeros(size(rois,1),numDays);%max r per roi across days
all_I=zeros(size(rois,1),numDays);%index of max r cell for rois across days
all_cent_dist=zeros(size(rois,1),numDays);
%after filtering for bad ROI fits, look up table below
LUT=zeros(size(rois,1),numDays);%look up table of all roi to single across days. not found or poor fit = NaN

%all has stitched single F from cross correlated ROIs to template ROIs
%could do Fc2/3 on individual days or across all. need to think about
%for now, is commented out
all_F= NaN(size(templ_all.F));
all_Fneu= NaN(size(templ_all.F));
all_spks= NaN(size(templ_all.F));
% all_Fc2= NaN(size(templ_all.F));
% all_Fc3= NaN(size(templ_all.F));
%structure with corrcoef results for each all day ROI
% day.corrs(day,1)
%structure with single.stat for each single day, ROIs info
% day.stat(day,1);
endIdx=0;

src = 'Z:\cellreg1month_Fmats\E186\';
%daysnm={'221206_YC', '221207_YC', '221208_YC', '221209_YC'};
%daysnm={'221211_YC', '221212_YC', '221213_YC', '221214_YC','221215_YC', '221216_YC', '221217_YC'};
%daysn=5:11;
daysnm={'221220_YC', '221221_YC', '221222_YC'};
daysn=12:14;
% daysnm={'221225_YC', '221226_YC', '221227_YC', '221228_YC', '221229_YC', '221230_YC'};
% daysn=15:20;
for j = 1:numDays
    %inside loop with numDays, cd to dir with file
    cd (single_paths{j,1}); %set path
    single=load(join([src,sprintf('d%03d',daysn(j)),'\',daysnm{j},'_Fall.mat'],''));%load single day, in single day folder
    %save Fall.mat for each day with as d1_Fall, d2_Fall, etc
    singles_all(j,:)=single;%gave mismatch error once when "redcell" was field in one file but not others

    singleFrames=size(single.F,2);%num frames per day
    startIdx=endIdx+1;%start of day, within all frames stack
    endIdx=endIdx+singleFrames;%end  of day, within all frames stack
    F_single=single.F';%transpose needed for corrcoef
    %could also do Fneu
    F_all_single=F_all(startIdx:endIdx,:);%size F_all to single day to single day
    %loop variables
    M=zeros(size(rois,1),1);%max r value per single all_roi to all d1_roi correlations
    I=zeros(size(rois,1),1);%Index of max r value (corresponding to best d1_roi match
    cent_all_xy=zeros(size(rois,1),2);%first column x, second y
    cent_single_xy=zeros(size(rois,1),2);
    cent_dist=zeros(size(rois,1),1);
    single_r_p=zeros(size(rois,1),1);

    concat_all_single=([F_all_single F_single]);%concatenate all and single F traces
    [single_r,single_p]=corrcoef(concat_all_single);%get r and p values for correlations of all to single
    figure, imagesc(single_r(1:size(F_all_single,2),(size(F_all_single,2)+1):end));%corr of all template cells found, not just iscell
    title('Corrcoef of All Template Cells Found');

    for i = 1:size(rois,1) %get single with highest r to all, +centroid dist
        [M(i,1),I(i,1)]=max(single_r(rois(i,1),(size(F_all_single,2)+1):end));
        single_r_p(i,1)=single_p(I(i,1));
        cent_all_xy(i,1)=mean(templ_all.stat{1,rois(i,1)}.xpix);
        cent_all_xy(i,2)=mean(templ_all.stat{1,rois(i,1)}.ypix);
        cent_single_xy(i,1)=mean(single.stat{1,I(i,1)}.xpix);
        cent_single_xy(i,2)=mean(single.stat{1,I(i,1)}.ypix);
        cent_dist(i,1)=pdist([cent_all_xy(i,:);cent_single_xy(i,:)],'euclidean');
        %into array across days, j=day, i=roi
        all_M(i,j)=M(i,1);
        all_I(i,j)=I(i,1);
        all_cent_dist(i,j)=cent_dist(i,1);
        %fill in look up table across days      
        if cent_dist(i,1) < max_cent_dist && M(i,1)>min_r%if centroids close and r high, write into LUT
            LUT(i,j)=I(i,1);
            all_F(rois(i,1),startIdx:endIdx)= single.F(I(i,1),:);
            all_Fneu(rois(i,1),startIdx:endIdx)= single.Fneu(I(i,1),:);
            all_spks(rois(i,1),startIdx:endIdx)= single.spks(I(i,1),:);
    %         all_Fc2(rois(i,1),startIdx:endIdx)= single.Fc2(I(i,1),:);
    %         all_Fc3(rois(i,1),startIdx:endIdx)= single.Fc3(I(i,1),:);    
        else%if centroids far or r low, LUT = NaN and F = 0
            LUT(i,j)=NaN;  
            all_F(rois(i,1),startIdx:endIdx)= 0;
            all_Fneu(rois(i,1),startIdx:endIdx)= 0;
            all_spks(rois(i,1),startIdx:endIdx)= 0;
    %         all_Fc2(rois(i,1),startIdx:endIdx)= 0; %uncomment when have Fc2
    %         all_Fc3(rois(i,1),startIdx:endIdx)= 0;
        end   
    end

    figure,scatter(M(:,1),cent_dist(:,1));%max r vs. centroid dist, most with high r have dist<15pix
    title('Max r vs. ROI centroid distance');
    ylabel('ROI Centroid Distance (pixels)')
    xlabel('Max R')
    frames(j,1)= singleFrames;%keep track of #frames per day
    dayStart(j,1)=startIdx;%keep track 1st frame of days
    dayEnd(j,1)=endIdx;%keep track of last frame of days   
    
    
    %uncomment to see ROI fits in loop. Can also do posthoc with...
    
    %option to see ROI fits or not    
    prompt = {['Inspect ROIs, Day ',num2str(j)]};
    dlgtitle = 'Inspect ROI alignment?';
    dims = [1 35];
    definput = {'1'}; %default files
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    inspect_rois=str2num(answer{1});
    
    for i = 1:size(rois,1) %graphs to loop through r, F comparison, rois
        if inspect_rois==1 %option to skip looking at roi graphs
%             f=figure;
%             f.WindowState='maximized';
%             subplot(3,2,1);
            subplot(2,2,1);
            annotation('textbox', [0.5, 0.9, 0.1, 0.1], 'String', ['Template ROI ' num2str(rois(i,1))]);
            annotation('textbox', [0.5, 0.85, 0.1, 0.1], 'String', ['Single ROI ' num2str(I(i,1))]);
            plot(single_r(rois(i,1),(size(F_all_single,2)+1):end));
            ylim([0 1])
            ylabel('corr')
            xlabel('ROIs')
            str=sprintf('r = %f',round(M(i,1),2));
            annotation('textbox', [0.3, 0.8, 0.1, 0.1], 'String', str);%working on inserting text box with r and later dist
%             str=sprintf('p = %f', single_r_p(i,1));
%             annotation('textbox', [0.3, 0.8, 0.1, 0.1], 'String', str);%working on inserting text box with r and later dist
%             subplot(3,2,3);
%             p1=plot(F_all_single(:,rois(i,1))); hold on;
%             p2=plot(F_single(:,I(i,1)));
%             p1.Color(4) = 1;
%             p2.Color(4) = 0.1; hold off;
%             %     plot(1:singleFrames,F_all_single(:,rois(i,1)),1:singleFrames,F_single(:,I(i,1)));
%             str=sprintf('centroid dist = %f pixels',cent_dist(i,1));
%             annotation('textbox', [0.6, 0.8, 0.1, 0.1], 'String', str);
%             subplot(3,2,[2,4]);
            subplot(2,2,2);
            imshow((single.ops.meanImg))%display white, if want auto bright: imshow((single.ops.meanImg),[])
            hold on
            scatter(templ_all.stat{1,rois(i,1)}.xpix,templ_all.stat{1,rois(i,1)}.ypix,'.','MarkerFaceAlpha',0.01,'MarkerEdgeAlpha',0.01)
            scatter(single.stat{1,I(i,1)}.xpix,single.stat{1,I(i,1)}.ypix,'.','MarkerFaceAlpha',0.01,'MarkerEdgeAlpha',0.01)
            title('Template and Single ROI');
            hold off
            %plot original F and all_F, either single F or zeros
%             subplot(3,2,5);
            subplot(2,2,3);
            plot(all_F(rois(i,1),startIdx:endIdx));
            title('Template ROI');
%             subplot(3,2,6);
            subplot(2,2,4);
            plot(F_all_single(:,rois(i,1)),'r');
             title('Single ROI');
            ylabel('dF/F')
            xlabel('frames')
            pause
            close all
        end
    end    
end

% all_F= all_F';%transpose back to original
% all_Fc2= all_Fc2';
% all_Fc3= all_Fc3';

stripped_name=regexprep(all_file,'.mat','');
currfile=strcat(stripped_name,'_%dd_align.mat');
currfile=sprintf(currfile,numDays);
currfilename=[all_path currfile];
save(currfilename, '-v7.3');    %need -v7.3 MAT file or variable is too big to save

file_list=strcat(stripped_name,'_%dd_file_list.mat');
file_list=sprintf(file_list,numDays);
file_list_full=[all_path file_list];
save(file_list_full,'all_file','all_path','single_files','single_paths','numDays', '-v7.3');    %need -v7.3 MAT file or variable is too big to save


