% pipeline to track cells across weekly concatenated videos in suite2p and
% track back to days
% relies on s2p fall.mat output at this point
% Dhanerawala et al. (2024)
%% 0 - setup folder paths
clear all
dst = 'Y:\analysis\celltrack'; % destination dir
src = 'Y:\analysis'; % source dir with animal specific folder ('e216')
animal = 'e190'; 
planes = [0,1,2,3]; % e.g. [0 1 2 3]
weekst = 1; weekend = 7; % week number

%% 1 - convert fall to appropriate format
% Choosing the files for conversion:
[file_names,files_path]=uigetfile('*.mat','MultiSelect','on',...
    'Choose the spatial footprints from all the sessions: ' );
input_format='Suite2p'; % could be either 'CNMF-e' or 'Suite2p'
convert_fall_suite2p_cnmf(file_names,files_path,input_format)
%% optional - run daily tracking
%%%%%%%%%% write daily tracking function here %%%%%%%%%% 
%% 2 - run weekly tracking
for plane=planes    
    % define path of sample data for all weeks
    fld = sprintf("fmats\\%s\\converted*week*plane%i_Fall.mat", animal, plane);

    % run
    tracking_per_week(src, animal, planes, weekst, weekend, fld)
end
%% 3 - visualize weekly tracked rois per plane
for plane=planes
weekfld = sprintf('week%02d-%02d_plane%i', weekst, weekend, plane);
fls = dir(fullfile(src, 'fmats', animal, sprintf('%s_week*_plane%i*.mat',animal, plane))); 
[savepth, commoncells] = plot_tracked_cells_week(src, animal, weekfld, fls);
end
%% 4 - run week to day mapping
weekall = [weekst:weekend]; % iterate through all the weeks
for week=weekall
    for plane=planes
        pthstr = sprintf("fmats\\%s\\week%02d_plane%i\\converted_*.mat",animal, week, plane); % edit if need be
        tracking_week2day(src, animal, planes, week, pthstr)
    end
end
%% 5 - combine maps and visualize
weekall = [weekst:weekend]; % iterate through all the weeks
for plane=planes
days_fls = dir(fullfile(src, sprintf("fmats\\%s\\days\\converted*plane%i*", animal, plane)));
sessions_total = length(days_fls); % total number of days tracked    
weekfld = sprintf('week%02d-%02d_plane%i', weekst, weekend, plane);
[savepth,cellmap2dayacrossweeks] = plot_tracked_cells_week2day(src, animal, ....
    weekfld, weekall, sessions_total);
end