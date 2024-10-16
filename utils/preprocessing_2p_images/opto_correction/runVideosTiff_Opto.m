%runVideos
%200820 EH. points to "loadVideoTiffNoSplit_EH2_new_sbx" for new sbx form.
%bi offsets different in new scanbox and sbx files also changed
% 200329 EH. modified from moi's code
% 200501 EH, make 'bi new' default
% also stopped removing last 4 from files{f} to make compatible with
% loadVideoTiffNoSplit_EH2.
% calls "loadVideoTiffNoSplit_EH2". see that file for changes
% changes loading, processing, and saving of tif files for suite2p

%Allows you to select multiple sbx files, separates out and stabilizes each
%plane, currently using the HMM method from SIMA

% zahra modified on 5/1/24 to not pick files and run through her day
% directories

clear all; close all;
days=[6]; 
src = 'Y:\halo_grabda\e243';
% src = 'X:\vipcre\e217';
lenVid=3000;
threshold = 0.055; % a tunable parameter to find stims, set at 0.4 for chr2 data 
loadVideoTiffNoSplit_Opto(src, days, lenVid, threshold);




