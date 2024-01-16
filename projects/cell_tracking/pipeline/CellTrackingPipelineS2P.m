% pipeline to track cells across weekly concatenated videos in suite2p and
% track back to days
% relies on s2p fall.mat output at this point
% Dhanerawala et al. (2024)
%% 1 - convert fall to appropriate format
% Choosing the files for conversion:
[file_names,files_path]=uigetfile('*.mat','MultiSelect','on',...
    'Choose the spatial footprints from all the sessions: ' );
input_format='Suite2p'; % could be either 'CNMF-e' or 'Suite2p'
convert_fall_suite2p_cnmf(file_names,files_path,input_format)
%% 2 - run weekly tracking
pth = 'Y:\analysis\celltrack';
% we need to find the path up-two levels
[fileroot,~,~] = fileparts(pth);
animal = 'e218'; % CHANGE
planes = [0]; % CHANGE
weekst = 1; weekend = 6;
% define path of sample data for all weeks
fls = dir(fullfile(fileroot, sprintf("fmats\\%s\\converted*week*plane%i_Fall.mat", animal, plane)));
% run
tracking_per_week(pth, animal, planes, weekst, weekend, fls)
%% 3 - visualize weekly tracked rois
%% 4 - run week to day mapping
%% 5 - combine days across weeks and visualize
