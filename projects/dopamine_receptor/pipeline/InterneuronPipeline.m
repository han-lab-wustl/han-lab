% %zahra's modified IN pipeline for DRD+ cells
% %full pipeline run each section in order
% %all of this happens from the sbx
% 
% %% section 1: run Videos tiff, makes tiffs from the .sbx files for as many
% %days as you want
% 
% % runVideosTiff_EH_new_sbx1
% %inputs are just the number of sbx to run and then you hand pick each one
% % zahra skips this as her python script makes tifs automatically
% 
% %% section 2: run suite 2p
% 
% %open up an anaconda prompt and change environments using the command :
% % conda activate suite2p
% 
% %then the command :'suite2p'
% 
% %then run the videos using the appropriate number of planes and don't do
% %roi detection
% 
%% section 3 get_stabilized_mat_file_per_day
% 
%run this section to create a mat file of containing all of the registered
%tiffs put together
% 
pr_dir={};
days = [28];
src ='Y:\drd';
animal='e262';
for i=1:length(days)
    pr_dir{i} = fullfile(src, animal, string(days(i)));
end
% 

for dy=1:length(pr_dir) % per day
    close all
    regtifs = dir(fullfile(pr_dir{dy}, '**', 'plane*'));
    get_stabilized_mat_file_per_day(length(regtifs), regtifs);

    %run this section to select cells. start by picking your file made from the
    %last script. input  a frequency that is the recording frequency of one
    %plane only. (31.25/nplanes). when a figure pops up. write 1 to adjust the
    %clip brightness. Adjusting the brightness may impact what is determined to
    %be an roi so keep this in mind. a new figure will pop up with blue circled
    %rois. click all of the rois you believe are cells and then simply click on
    %the background when you are done selecting.
    %it will ask you if there are any cells it missed that you would like to
    %add by hand. if you say yes, a new figure will pop up where you can draw
    %lines around the first cell you would like to add. when you complete a
    %polygon double click on the center and it will assign that polygon a
    %number and prompt you in the command window if you would like to add
    %another. repeat until you have drawn all cells and hit 0 for no.
    time=300; %size of moving avg window (s)    
    mats = dir(fullfile(pr_dir{dy}, '**', 'file*XC_plane*.mat'));
    Fs=31.25/length(mats);
    for m=1:length(mats)
        click_ROIs(time, Fs, mats(m)) % per plane
    end
end
%% 
% pr_dir=uipickfiles;
for dy=1:length(pr_dir) % per day
    % align images to behavior

    src = pr_dir{dy};
    daypth = dir(fullfile(src, "**\behavior", "vr\*.mat"));
    fmatfl = dir(fullfile(src, '**\*roibyclick_F.mat')); 
    savepthfmat = VRalign_INpipeline(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
    disp(savepthfmat)
end
% % 
% %% section 5 elim_oversample_multiplane
% % 
% % %run this section once you have all planes cell selected. This will ask you
% % %to verify all cell rois look like they have a real signal. then it will
% % %ask to review cells in similar xy position and high correlation accross
% % %planes. If you believe that these cells are the same, it will remove them
% % %from one of the two planes ( it will keep the one with higher average
% % %intensity)
% % % 
% % elim_oversample_multiplane
% % 
% %% section 6 make_procs
% % 
% % %this script simply prepares the data in a format to be ready to be read by
% % %registers2p to align accross days
% % 
% % %don't worry too much about the day number when it asks. its just to help
% % %you remember later which day's planes you are looking for, nothing hard
% % %coded.
% % 
% % % zd - iterates through all days and planes you pick
% pr_dir=uipickfiles;
% make_proc_files_Ints_per_day(pr_dir)
% % % 
% %% section 7 registers2p
% % 
% registers2p
% % 
%% problematic sections - indexing issue
% 10/10/24
%% section 8 multiple_days_reg_file

% zd - iterates through all days and planes you pick
% pr_dir=uipickfiles;
% template_day = 1;
% multiple_days_reg_file_GM(pr_dir,template_day)
% %% section 9 main_getall_days_registered
% close all
% % pr_dir=uipickfiles;
% days_involved = 'd12d13d14d15'; % change to fit your days tracked
% template_day = 1;
% get_all_cells_registered(pr_dir, days_involved,template_day)

%% vralign

% pr_dir=uipickfiles;
% for dy=1:length(pr_dir) % per day
%     % align images to behavior
% 
%     src = pr_dir{dy};
%     daypth = dir(fullfile(src, "**\behavior", "vr\*.mat"));
%     % daypth = dir(fullfile(src, "**\*).mat"));
% 
% %     sprintf('%i',day), sprintf('%s*mat', mouse_name)));%, 
%     fmatfl = dir(fullfile(src, '**\plane*all_days.mat')); % now aligns to aligned cell mat
%     savepthfmat = VRalign_INpipeline(fullfile(daypth.folder, daypth.name),fmatfl, length(fmatfl));
%     disp(savepthfmat)
% end