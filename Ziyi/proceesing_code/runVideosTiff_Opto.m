%runVideos

% zahra modified on 5/1/24 to not pick files and run through her day
% directories
% mods on 10/18/24 to implement standard deviation filter for blue/red LED
% triggered optogenetics

clear all; close all;
days=[]; 
src = 'E:\Ziyi\Data\241018_ZH\241018_ZH_000_001';
% src = 'X:\vipcre\e217';
lenVid=3000;
% threshold = 0.07; % a tunable parameter to find stims, set at 0.4 for chr2 data 
artifact_type=-1; %negative -1 if red laser opto to detect pmt block; if blue laser opto, keep 1
bandlimit=14; % top y dim of image to detect artifact
loadVideoTiffNoSplit_Opto(src, days, lenVid, artifact_type, bandlimit);




