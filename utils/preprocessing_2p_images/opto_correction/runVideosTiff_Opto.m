
%runVideos

% zahra modified on 5/1/24 to not pick files and run through her day
% directories
% mods on 10/18/24 to implement standard deviation filter for blue/red LED
% triggered optogenetics

clear all; close all;
days=['\Day_01_2nd session']; 
src = "\\storage1.ris.wustl.edu\ebhan\Active\Tarikul\E219_opto\Long opto control";
% src = 'X:\vipcre\e217';
lenVid=3000;
threshold = 0.4; % a tunable parameter to find stims, set at 0.4 for chr2 data 
loadVideoTiffNoSplit_Opto(src, days, lenVid, threshold);



