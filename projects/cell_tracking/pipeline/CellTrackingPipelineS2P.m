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
dst = 'Y:\analysis\celltrack';
src = 'Y:\analysis';
animal = 'e218'; 
planes = [0]; % e.g. [0 1 2 3]
weekst = 1; weekend = 6; % week number
% define path of sample data for all weeks
fld = sprintf("fmats\\%s\\converted*week*plane%i_Fall.mat", animal, plane);
for plane=planes    
    % run
    tracking_per_week(dst, animal, planes, weekst, weekend, fld)
end
%% 3 - visualize weekly tracked rois per plane
for plane=planes
weekfld = sprintf('week%02d-%02d_plane%i', weekst, weekend, plane);
fls = dir(fullfile(src, 'fmats', animal, sprintf('%s_week*_plane%i*.mat',animal, animal))); 
[savepth, commoncells] = plot_tracked_cells_week(src, animal, weekfld, fls);
end
%% 4 - run week to day mapping
weekall = [weekst:weekend]; % iterate through all the weeks
pthstr = "fmats\\%s\\week%02d_plane%i\\converted_*.mat"; % edit if need be
for week=weekall
tracking_week2day(pth, animal, planes, week, pthstr)
end
%% 5 - combine maps and visualize
sessions_total = 31; % total number of days tracked
for plane=planes
weekfld = sprintf('week%02d-%02d_plane%i', weekst, weekend, plane);
[savepth,cellmap2dayacrossweeks] = plot_tracked_cells_week2day(src, animal, ....
    weekfld, weeknms, sessions_total);
end