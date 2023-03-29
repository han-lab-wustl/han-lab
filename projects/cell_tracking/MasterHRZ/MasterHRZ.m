 %% Master script to analyze Pyrs cell data from HRZ task
 
 %%
% Requires that your Fall structure has the aligned beahvioral variables
% attached to it, in the format of VRselectstartendsplit. <-- align with
% behavior
%%
% coomment for Gerardo : when using VRselectstartendsplit, would be great
% to have the date and the VR settings attached to it too! So we can keep
% this info along.
addpath('C:\Users\Han\Documents\MATLAB\pyrca2_tracking\HRZ_master\scripts_dFF_Fc3')
addpath('C:\Users\Han\Documents\MATLAB\pyrca2_tracking\HRZ_master\create_tables_scripts')
addpath('C:\Users\Han\Documents\MATLAB\pyrca2_tracking\HRZ_master\behavior')
addpath('C:\Users\Han\Documents\MATLAB\pyrca2_tracking\HRZ_master\')
do_quality_control = true; % sometimes Suite2p finds cells with F = 0 that are difficult to identify. This allow to remove them post hoc if any and keeps track of which of them were removed.
                            % this also creates the 'all' structure. and
                            % appends it to the Fall.   
                            
                            
%% MAIN SETTINGS ----------------------------------------------------------

Settings.paths = dir("Z:\cellreg1month_Fmats\221207_YC_Fall.mat"); % you can set a specific day by substituting D* with D1 for example
Settings.Fs = 32; % Hz
Settings.level_mouse_name = 2; % at which level of the folder .path is the mouse name contained
Settings.level_day = 2; % at which level of the folder .path is the day N contained
Settings.gainVR = 0.66; % which gain was used in these recordings
Settings.bin_size = 5 * Settings.gainVR; % cm
Settings.UL_track = 180; % Upper Limit of the track
Settings.numIterations = 1000; % how many iterations for shuffled distribution

%--------------------------------------------------------------------------
%% Quality control / create dff and Fc3
% Suite2p identifies ROI where the F = 0 troughout the whole session.
% We want to eliminare these cells from the selection. You'll see them
% being plotted when identified. This also creates the 'all' structure if
% present.

if do_quality_control
recreate_iscell_and_make_all_struct(Settings)  
end


%--------------------------------------------------------------------------
%% Create the Epoch Table and figures
% creates a folder with epoch table and create remapping anlysis figures
% for each epoch.
% Settings.paths = dir('/home/gaia/Desktop/E146/E146/D14/Fall.mat'); % if you want to work/add on one specific day

clearvars -except Settings

Settings.saving_path = 'Z:\HRZ_master_output' ; % please just change the path where you want the folder 'CS_table_last' to be created/saved.
Settings.probe_trials = 'exclude'; % DO NOT change. Probe trials are excluded from the analysis.
Settings.trials_2compare = 8; % take the last 8 trials of one epoch
Settings.I_want2save_figures = true; % save figures for each epoch
Settings.I_want2reanlyze = true; % start table from scratch 

makeEpochTable(Settings)
%--------------------------------------------------------------------------
%% Create Trial Table
% creates a folder with trials table 

Settings.saving_path = 'Z:\HRZ_master_output';
Settings.probe_trials = 'intra_all'; % DO NOT change. Probe trials are included in the analysis.
Settings.I_want2reanlyze = true;

make_trial_Tab(Settings)
%--------------------------------------------------------------------------
%% Final figures
% check settings and saving paths

plot_from_trialsTable_epochTable

%plot_from_table_cs_table
%plot_from_table_trial_by_trials_probesCS
