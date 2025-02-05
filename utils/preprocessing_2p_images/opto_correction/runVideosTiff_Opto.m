%runVideos

% zahra modified on 5/1/24 to not pick files and run through her day
% directories

clear all; close all;
days=[10]; 
src = "X:\lc_chr2_grabda\e278";
% src = "Y:\halo_grabda\e241";
lenVid=3000;
threshold = 0.4; % a tunable parameter to find stims, set at 0.4 for chr2 data 
crop_etl = 1; % 1 if cropping etl, may want to leave if signal is low for motion corr
loadVideoTiffNoSplit_Opto(src, days, lenVid, threshold, crop_etl);




