edclear all
close all
mouse_id=181;
addon = '_RRloccor';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
pr_dir0 = uipickfiles;

% oldbatch=input('if oldbatch press 1 else 0=');
oldbatch = 0;


 roiloccor = [];
  roiloccorP = [];
  
% dop_allsuc_stop_no_reward = NaN(length(pr_dir0),4,79);
for alldays = 1:length(pr_dir0)%[3:12 14:19]%[3:12 13:19]%[3 5:1]%[5:12 14]%[5:12 14]%1:5%26:30%1:33%31:32%30%27:30%21%[1:21]%[1:2 4:5 12:22]%[8:21]%[2:4 6:11]%[1:2 4:5 12:20]%%[1:21]%%
%     clearvars -except alldays addon mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr ...
%         mov_corr_success stop_corr_success mov_corr_prob stop_corr_prob mov_corr_fail stop_corr_fail...
%         c  alldays mouse_id mov_corr stop_corr mov_stop conc_coeff_rew conc_coeff_nr dop_suc_movint dop_suc_stopint roe_suc_movint roe_suc_stopint ...
%         dop_allsuc_mov dop_allsuc_stop roe_allsuc_mov roe_allsuc_stop ...
%         dop_allfail_mov dop_allfail_stop roe_allfail_mov roe_allfail_stop...
%         dop_alldays_planes_success_mov dop_alldays_planes_fail_mov dop_alldays_planes_success_stop dop_alldays_planes_fail_stop...
%         roe_alldays_planes_success_mov roe_alldays_planes_fail_mov roe_alldays_planes_success_stop roe_alldays_planes_fail_stop...
%         subp days_check cnt dop_alldays_planes_perireward roe_alldays_planes_perireward dop_allsuc_perireward roe_allsuc_perireward...
%         pr_dir0 dop_alldays_planes_success_stop_no_reward dop_alldays_planes_success_stop_reward...
%         dop_allsuc_stop_no_reward dop_allsuc_stop_reward roe_allsuc_stop_no_reward roe_allsuc_stop_reward...
%         dop_alldays_planes_double_0 roe_alldays_planes_double_0 dop_alldays_planes_perireward_double roe_alldays_planes_perireward_double...
%         dop_allsuc_perireward_double roe_allsuc_perireward_double dop_allsuc_perireward_double_se roe_allsuc_perireward_double_se...
%         day_labels roi_dop_alldays_planes_success_mov roi_dop_alldays_planes_success_stop roi_dop_alldays_planes_success_stop_no_reward...
%         roi_dop_alldays_planes_success_stop_reward roi_dop_allsuc_mov roi_dop_allsuc_stop roi_dop_allsuc_stop_no_reward roi_dop_allsuc_stop_reward...
%         roi_dop_alldays_planes_perireward_0 roi_dop_alldays_planes_perireward_double_0 roi_dop_alldays_planes_perireward roi_dop_alldays_planes_perireward_double...
%         roi_dop_allsuc_perireward roi_dop_allsuc_perireward_double roi_dop_allsuc_perireward_se roi_dop_allsuc_perireward_double_se...
%         roe_alldays_planes_perireward_0 roe_alldays_planes_success_stop_no_reward roe_alldays_planes_stop_reward roe_alldays_planes_perireward_double_0...
%         dop_alldays_planes_perireward_double_0 dop_alldays_planes_perireward_0 roe_alldays_planes_success_stop_reward...
%         roi_dop_alldays_planes_periCS roi_dop_alldays_planes_peridoubleCS roi_roe_alldays_planes_periCS roi_roe_alldays_planes_peridoubleCS roi_dop_allsuc_perirewardCS...
%         roi_dop_allsuc_perireward_doubleCS roi_roe_allsuc_perirewardCS roi_roe_allsuc_perireward_doubleCS...
%         roi_dop_alldays_planes_perinrlicks_0  roi_dop_alldays_planes_perinrlicks...
%         roi_dop_alldays_planes_periUS roi_dop_alldays_planes_peridoubleUS roi_dop_allsuc_perirewardUS roi_dop_allsuc_perireward_doubleUS...
%         roi_roe_alldays_planes_periUS roi_roe_alldays_planes_peridoubleUS roi_roe_allsuc_perirewardUS roi_roe_allsuc_perireward_doubleUS...
%         roi_dop_alldays_planes_unreward_single_0 roi_dop_alldays_planes_unreward_single roi_dop_allsuc_unreward_single roi_dop_allsuc_unreward_single_se...
%         roi_roe_alldays_planes_unrewardCS roi_roe_allsuc_perireward_unrewardCS oldbatch  roi_dop_alldays_planes_rewusonly_single_0...
%         roi_dop_alldays_planes_rewusonly_single_0 roi_dop_alldays_planes_rewusonly_single roi_dop_allsuc_rewusonly_single   roi_dop_allsuc_rewusonly_single_se....
%         roi_roe_alldays_planes_rewusonly roi_roe_allsuc_rewusonly roi_dop_alldays_planes_perireward_usonly_0   roi_dop_alldays_planes_perireward_usonly...
%         roi_dop_allsuc_perireward_usonly  roi_dop_allsuc_perireward_usonly_se roi_dop_alldays_planes_perirewarddoublemUS_CS_0    roi_dop_alldays_planes_perirewarddoublemUS_CS...
%          roi_dop_allsuc_perirewarddoublemUS_CS  roi_dop_allsuc_perirewarddoublemUS_CS_se roi_roe_alldays_planes_perirewarddoublemUS_CS    roi_roe_allsuc_perirewarddoublemUS_CS
    
    clearvars lickVoltage forwardvel
%     
    
    close all
    
    
    planeroicount = 0;
    cnt=cnt+1;
    Day=alldays;
    for allplanes=1:4
        plane=allplanes;
        
        pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
        pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','')
        
        
        
        if exist( pr_dir, 'dir')
            
            cd (pr_dir)
            
            load('params.mat')
            if isfield(params,'base_mean')
                base_mean = params.base_mean;
            else
%                 oldversionfile = dir('file*.mat');
%                 load(oldversionfile.name)
                base_mean = params.roibasemean{1};
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end
            
            if ~exist('lickVoltage')
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end
%             if allplanes == 1 %%% CHANGE 1 FOR DEFAULT
%                 forwardvel1= forwardvel;
%                 rewards1 = rewards;
%             else
%                 forwardvel = forwardvel1;
%                 rewards = rewards1;
%             end
            %%%%%%%%%%%%%%%
            numplanes=4;
            gauss_win=5;
            frame_rate=31.25;
            lickThresh=-0.085;%-0.085; ZD changed to -0.07 because code was crashing otherwise...
            rew_thresh=0.001;
            sol2_thresh=1.5;
            num_rew_win_sec=5;%window in seconds for looking for multiple rewards
            rew_lick_win=10;%window in seconds to search for lick after rew. could be long in naive animals but likely short latency in trained
            pre_win=5;%pre window in s for rewarded lick average
            post_win=5;%post window in s for rewarded lick average
            exclusion_win=10;%exclusion window pre and post rew lick to look for non-rewarded licks
            speed_thresh = 12; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 4; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            CSUStimelag = 0.5; %seconds between
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time/numplanes);
            post_win_framesALL=round(post_win/frame_time);
            pre_win_framesALL = round(pre_win/frame_time);
            pre_win_frames=round(pre_win/frame_time/numplanes);
            exclusion_win_frames=round(exclusion_win/frame_time);
            CSUSframelag_win_frames=round(CSUStimelag/frame_time);
            
            
            mean_base_mean=mean(base_mean);
            
            norm_base_mean=base_mean;
            
            if exist('forwardvelALL')
                speed_binned=forwardvelALL;
            end
            reward_binned=rewardsALL;
            % temporary artefact check and remove
            temp= find(reward_binned);
            reward_binned(temp(find(diff(temp) == 1))) = 0; 
            speed_smth_1=smoothdata(speed_binned,'gaussian',gauss_win)';
            dop_smth=smoothdata(norm_base_mean,'gaussian',gauss_win);
            speed_smth_2 = smoothdata(forwardvel,'gaussian',gauss_win);
            
            
            
    
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%% for regions of interest
            
            if ~isfield(params,'roibasemean3')
                
                df_f=params.roibasemean2;
            else
                df_f=params.roibasemean3;
            end
            
            for roii = 1:size(df_f,1)
                planeroicount = planeroicount + 1;
                    planeroicount
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                
                [temp,tempP] = corrcoef(roibase_mean,speed_smth_2);
                
                
                roiloccor(planeroicount,alldays) = temp(1,2);
                roiloccorP(planeroicount,alldays) = tempP(1,2);
                           
                
            end
 
        end
        
        
    end
    
end

% SAVING WHO ACCROSS DAY WORKSPACE
k = strfind(pr_dir0{1},'\');
allsavedir = pr_dir0{1}(1:k(end));
filenamestrip = [num2str(mouse_id) addon];
allsavefilename = [filenamestrip '_workspace'];
filetype = '.mat';
startCount = 0;
numDigits = '2';
if exist([allsavedir allsavefilename filetype],'file') == 2 %file exists already, check for alternative
    checker = true; %check for alternate file names
    Cnt = startCount; %counter for file name
    
    while checker
        testPath = [allsavedir allsavefilename '_' num2str(Cnt, ['%0' numDigits 'i']) filetype];
        
        if exist(testPath,'file') == 2
            Cnt = Cnt + 1; %increase counter until a non-existing file name is found
        else
            checker = false;
        end
        
        if Cnt == 10^numDigits-1 && checker
            numDigits = numDigits+1;
            warning(['No unused file found at given number of digits. Number of digits increased to ' num2str(numDigits) '.']);
        end
    end
    outFile = testPath;
    
else
    outFile = [allsavedir allsavefilename filetype];
end
close all
save(outFile)

