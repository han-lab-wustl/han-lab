%runVideos

% zahra modified on 5/1/24 to not pick files and run through her day
% directories

clear all; close all;
days=[1]; 
% src = "X:\lc_chr2_grabda\e278";
src = "X:\chrimson_snc_grabda\e294";
lenVid=3000;
threshold = 0.4; % a tunable parameter to find stims, set at 0.4 for chr2 data 
crop_etl = 0; % 1 if cropping etl, put 1, may want to leave if signal is low for motion corr
loadVideoTiffNoSplit_Opto(src, days, lenVid, threshold, crop_etl);




