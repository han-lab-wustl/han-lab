 %% Master script to analyze Pyrs cell data from HRZ task
 
 %%
% Requires that your Fall structure has the aligned beahvioral variables
% attached to it, in the format of VRselectstartendsplit. <-- align with
% behavior
                            
%% MAIN SETTINGS ----------------------------------------------------------

Settings.paths = dir("Y:\sstcre_analysis\fmats\e201\days\*day055*"); % you can set a specific day by substituting D* with D1 for example
% % formatting for zahra's path
% for i=1:length(Settings.paths)
%     [parent,nm,~] = fileparts(fileparts(fileparts(fileparts(Settings.paths(i).folder))));
%     dy{i} = str2num(nm);
% end 
% dys = cell2mat(dy);
% Settings.hrz_days = [55];%[55:73,75:80];%[62:74, 76, 78:81]; % get days after this day; ZD added for her folder structure
% Settings.paths = Settings.paths(ismember(dys,Settings.hrz_days)); % only certain days
Settings.Fs = 31.25; % Hz
Settings.level_mouse_name = 3; % at which level of the folder .path is the mouse name contained, 
% e.g. Z:\sstcre_imaging\e201, mouse name would be level 3
Settings.level_day = 4; % at which level of the folder .path is the day N contained
Settings.gainVR = 2/3;%0.66; % which gain was used in these recordings
Settings.bin_size = 2 / Settings.gainVR; % cm
Settings.UL_track = 180 / Settings.gainVR; % Upper Limit of the track
Settings.numIterations = 1000; % how many iterations for shuffled distribution

%--------------------------------------------------------------------------
%% Create the Epoch Table and figures
% creates a folder with epoch table and create remapping anlysis figures
% for each epoch.
% Settings.paths = dir('/home/gaia/Desktop/E146/E146/D14/Fall.mat'); % if you want to work/add on one specific day

clearvars -except Settings

Settings.saving_path = 'Y:\sstcre_analysis\hrz\opto\e201\' ; % please just change the path where you want the folder 'CS_table_last' to be created/saved.
Settings.probe_trials = 'exclude'; % DO NOT change. Probe trials are excluded from the analysis.
Settings.trials_2compare = 8; % take the last 8 trials of one epoch
Settings.I_want2reanlyze = true; % start table from scratch 

makeEpochTable(Settings)
%--------------------------------------------------------------------------
%% Create Trial Table
% creates a folder with trials table 

Settings.probe_trials = 'intra_all'; % DO NOT change. Probe trials are included in the analysis.
Settings.I_want2reanlyze = true;

make_trial_Tab(Settings)
%--------------------------------------------------------------------------
%% Final figures
% check settings and saving paths

plot_from_trialsTable_epochTable(Settings)
