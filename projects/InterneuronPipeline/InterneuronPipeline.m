<<<<<<< HEAD
%full pipeline run each section in order
%all of this happens from the sbx

%section 1: run Videos tiff, makes tiffs from the .sbx files for as many
%days as you want

runVideosTiff_EH_new_sbx1
%inputs are just the number of sbx to run and then you hand pick each one


%% section 2: run suite 2p

%open up an anaconda prompt and change environments using the command :
% conda activate suite2p

%then the command :'suite2p'

%then run the videos using the appropriate number of planes and don't do
%roi detection

%% section 3 get_stabilized_mat_file_per_day

%run this section to create a mat file of containing all of the registered
%tiffs put together

get_stabilized_mat_file_per_day

%just put in the number of planes and then select the reg_tif folder for
%each of those planes. Since this does a for loop for each plane you can do
%multiple days by say number of planes = dxn where d is number of days and
%n is number of planes. then simply select all of the reg_tiff folders

%% section 4 click_rois_GM

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

click_ROIs_GM

%% section 5 elim_oversample_multiplane

%run this section once you have all planes cell selected. This will ask you
%to verify all cell rois look like they have a real signal. then it will
%ask to review cells in similar xy position and high correlation accross
%planes. If you believe that these cells are the same, it will remove them
%from one of the two planes ( it will keep the one with higher average
%intensity)
% 
% elim_oversample_multiplane

%% section 6 make_procs

%this script simply prepares the data in a format to be ready to be read by
%registers2p to align accross days

%don't worry too much about the day number when it asks. its just to help
%you remember later which day's planes you are looking for, nothing hard
%coded.

make_proc_files_Ints_per_day

%% section 7 registers2p

registers2p

%% section 8 multiple_days_reg_file

startnew = input('are you starting from scratch? (0 - no , 1 - yes) :');

if startnew == 1

multiple_days_reg_file_GM
else
    multiple_days_reg_file_cont
end

%% section 9 main_getall_days_registered

main_get_all_cells_registered_GM

% %
% %
% %
% % passed this point is HRZ specific
%% section 10 vrselect startend split

% VRselectstartendsplit
abffileSelectStartEndSplit

%% section 11 Interneuron Struture PV


%% section 11 Interneuron Struture

m=1; %do not change
curr=3; %current day
[mouseP(m).Falls{curr},mouseP(m).Forwards{curr},...
    mouseP(m).CPP{curr},mouseP(m).ybinned{curr},mouseP(m).rewards{curr},mouseP(m).F0{curr},...
    mouseP(m).masks{curr},mouseP(m).mimg{curr},mouseP(m).Fallnb{curr},mouseP(m).name{curr},mouseP(m).time{curr},mouseP(m).rewardLocation{curr},mouseP(m).licks{curr},mouseP(m).trialNum{curr}]=... 
    HRZNeuronStructor(7.8,4,'E137',...
    {'file000_cha_XC_plane_1_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_2_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_3_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_4_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days'},...
    {'E:\E137\D14\suite2p\plane0\reg_tif\',...
    'E:\E137\D14\suite2p\plane1\reg_tif\',...
    'E:\E137\D14\suite2p\plane2\reg_tif\',...
    'E:\E137\D14\suite2p\plane3\reg_tif\'}...
   ,0,'new',160); %ETL Was 3600 but thats past sat range
=======
%full pipeline run each section in order
%all of this happens from the sbx

%section 1: run Videos tiff, makes tiffs from the .sbx files for as many
%days as you want

runVideosTiff_EH_new_sbx1
%inputs are just the number of sbx to run and then you hand pick each one


%% section 2: run suite 2p

%open up an anaconda prompt and change environments using the command :
% conda activate suite2p

%then the command :'suite2p'

%then run the videos using the appropriate number of planes and don't do
%roi detection

%% section 3 get_stabilized_mat_file_per_day

%run this section to create a mat file of containing all of the registered
%tiffs put together

get_stabilized_mat_file_per_day

%just put in the number of planes and then select the reg_tif folder for
%each of those planes. Since this does a for loop for each plane you can do
%multiple days by say number of planes = dxn where d is number of days and
%n is number of planes. then simply select all of the reg_tiff folders

%% section 4 click_rois_GM

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

click_ROIs_GM

%% section 5 elim_oversample_multiplane

%run this section once you have all planes cell selected. This will ask you
%to verify all cell rois look like they have a real signal. then it will
%ask to review cells in similar xy position and high correlation accross
%planes. If you believe that these cells are the same, it will remove them
%from one of the two planes ( it will keep the one with higher average
%intensity)
% 
% elim_oversample_multiplane

%% section 6 make_procs

%this script simply prepares the data in a format to be ready to be read by
%registers2p to align accross days

%don't worry too much about the day number when it asks. its just to help
%you remember later which day's planes you are looking for, nothing hard
%coded.

make_proc_files_Ints_per_day

%% section 7 registers2p

registers2p

%% section 8 multiple_days_reg_file

startnew = input('are you starting from scratch? (0 - no , 1 - yes) :');

if startnew == 1

multiple_days_reg_file_GM
else
    multiple_days_reg_file_cont
end

%% section 9 main_getall_days_registered

main_get_all_cells_registered_GM

% %
% %
% %
% % passed this point is HRZ specific
%% section 10 vrselect startend split

% VRselectstartendsplit
abffileSelectStartEndSplit

%% section 11 Interneuron Struture PV


%% section 11 Interneuron Struture

m=1; %do not change
curr=3; %current day
[mouseP(m).Falls{curr},mouseP(m).Forwards{curr},...
    mouseP(m).CPP{curr},mouseP(m).ybinned{curr},mouseP(m).rewards{curr},mouseP(m).F0{curr},...
    mouseP(m).masks{curr},mouseP(m).mimg{curr},mouseP(m).Fallnb{curr},mouseP(m).name{curr},mouseP(m).time{curr},mouseP(m).rewardLocation{curr},mouseP(m).licks{curr},mouseP(m).trialNum{curr}]=... 
    HRZNeuronStructor(7.8,4,'E137',...
    {'file000_cha_XC_plane_1_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_2_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_3_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days',...
    'file000_cha_XC_plane_4_roibyclick_F_D1_D2_D3_D5_DR1_DR3_DR21_DR22_DR23_registered_all_days'},...
    {'E:\E137\D14\suite2p\plane0\reg_tif\',...
    'E:\E137\D14\suite2p\plane1\reg_tif\',...
    'E:\E137\D14\suite2p\plane2\reg_tif\',...
    'E:\E137\D14\suite2p\plane3\reg_tif\'}...
   ,0,'new',160); %ETL Was 3600 but thats past sat range
>>>>>>> 754f532e47d152334ffae033cf3e5763ab9bf2c0
