% Zahra
% cluster cells hehe
% add function path
addpath(fullfile(pwd, "utils"));
pth = uigetfile('*.mat','MultiSelect','on');
load(pth)
dff=redo_dFF(day.F, 31.25, 20, day.Fneu);