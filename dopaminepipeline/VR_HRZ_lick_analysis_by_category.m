clear all
close all
mouse_id=168;
addon = 'lick_HRZ';
mov_corr=[]; stop_corr=[]; mov_stop=[];
mov_corr_success=[]; stop_corr_success=[];
mov_corr_prob=[]; stop_corr_prob=[];
mov_corr_fail=[]; stop_corr_fail=[]; cnt=0;
pr_dir0 = uipickfiles;

% oldbatch=input('if oldbatch press 1 else 0=');
oldbatch = 0;



differentlicks = {'success_pre_rew_far_licksind',... % naming all the different lick categories
    'success_pre_rew_near_licksind',...
    'success_in_rew_licksind_pre_us'...
    'success_in_rew_licksind_post_us'...
    'success_post_rew'...
    'probe_pre_rew_far_licksind'...
    'probe_pre_rew_near_licksind'...
    'probe_in_rew_licksind'...
    'probe_post_rew'...
    'failed_pre_rew_far_licksind'...
    'failed_pre_rew_near_licksind'...
    'failed_post_rew'};

for d = 1:length(differentlicks) % for each category setup a storing variable to compile across days
    temp1 = differentlicks{d};
    temp = temp1;
    [outstart,outstop] = regexp(temp,'_licksind');
    if ~isempty(outstart) % remove mention of licks ind for shorter precise variable names
        temp(outstart:outstop) = [];
    end
    eval(['ALL_dop_allday_' temp ' = cell(length(pr_dir0),1);']); %initialize dopamine variables
    eval(['ALL_dopnorm_allday_' temp ' = cell(length(pr_dir0),1);']); %initialize pre window normalized dopamine variables
    eval(['ALL_doptime_allday_'  temp ' = cell(length(pr_dir0),1);']); %initialize time matrix for dopamine variables
    eval(['ALL_Spd_allday_' temp ' = cell(length(pr_dir0),1);']); % initialize speed variable
    eval(['ALL_Spdtime_allday_' temp ' = cell(length(pr_dir0),1);']); % initialize speed time variable
end


for Day = 1:length(pr_dir0)  
    clearvars lickVoltage forwardvel  
    close all
  
    for allplanes=1:4
        plane=allplanes;   
        pr_dir1 = strcat(pr_dir0{Day},'\suite2p');
        pr_dir=strcat(pr_dir1,'\plane',num2str(plane-1),'\reg_tif\','');
        
        if exist( pr_dir, 'dir')
            
            cd (pr_dir)
            
            load('params.mat')
            if isfield(params,'base_mean')
                base_mean = params.base_mean;
            else
                oldversionfile = dir('file*.mat');
                load(oldversionfile.name)
                if ~exist('forwardvel')
                    forwardvel = speed_binned;
                end
            end
            
            
            %% daily - split and find lick precise lick indices and categorize into multiple lick binaries
            %for now numprobes is manually written. please replace with
            %VR.settings if you have this in the future
            numprobes = 3;
            nearzone = 20;%cm before reward zone check
            uybinned_c1 = uybinned/VR.scalingFACTOR; % convert to 270 cm change with (uybinned/vr.settings.gain)
            uchangeRewLoc_c1 = uchangeRewLoc/VR.scalingFACTOR;
            rewzonesize = (VR.settings.rewardZone/VR.scalingFACTOR)/2; %change with (VR.settings.rewzonesize/VR.settings.gain)/2;
            usolenoid2 = VR.reward(scanstart:scanstop)==0.5;
            if ~exist('urewards','var')
                urewards = VR.reward(scanstart:stop) == 1;
            end
            
            
            
            if allplanes == 1
            
            [~,temp] = findpeaks(-1*ulickVoltage,'MinPeakHeight',0.065,'MinPeakDistance',2);
            filteredlicks = zeros(1,length(ulickVoltage));
            filteredlicks(temp) = 1;
            filteredlicks(uybinned_c1<2) = 0;
            
            utrialnum_c1 = utrialnum;
            fixes = find(utrialnum_c1-[utrialnum(1) utrialnum_c1(1:end-1)] ~=0 & utrialnum_c1-[utrialnum_c1(2:end) utrialnum_c1(end)]~=0);
            
            utrialnum_c1(fixes) = utrialnum_c1(fixes+1); % removes one point of between epoch that is its own trial for no reason
            utrialnum_c1(find(diff(utrialnum_c1))+1) = utrialnum_c1(find(diff(utrialnum_c1))); %moves the change of reward loc to match teleportation
           
            
            %%Probe Lickversions
            probeindices = consecutive_stretch(find(utrialnum_c1<3));
            
            %remove probe trials for the null distribution, i.e. in front
            %of the first epoch
            temp = find(uchangeRewLoc_c1);
            if ~isempty(probeindices{1}) && length(temp)>1
            probeindices(cellfun(@(x) x(1)<temp(2),probeindices,'UniformOutput',1)) = []; % delete probes that start before the change from epoch 1 to epoch 2
            end
            probe_pre_rew_far_licksind = [];
            probe_pre_rew_near_licksind = [];
            probe_in_rew_licksind = [];
            probe_post_rew = [];
            
            if ~isempty(probeindices{1})
            for p = 1:length(probeindices)
                currentreward = uchangeRewLoc_c1(temp(p));
                temp1 = find(uybinned_c1(probeindices{p})<currentreward-nearzone-rewzonesize)+probeindices{p}(1)-1; %find for this epoch, indices in the far before category
                if ~isempty(temp1)
                    probe_pre_rew_far_licksind = [probe_pre_rew_far_licksind temp1(find(filteredlicks(temp1)))];
                end
                
                temp1 = find((uybinned_c1(probeindices{p})<currentreward-rewzonesize) & (uybinned_c1(probeindices{p})>=currentreward-rewzonesize-nearzone))+probeindices{p}(1)-1; %find for this epoch, indiceds in the near reward zone pre category
                if ~isempty(temp1)
                    probe_pre_rew_near_licksind = [probe_pre_rew_near_licksind temp1(find(filteredlicks(temp1)))];
                end
                
                temp1 = find(uybinned_c1(probeindices{p})>=currentreward-rewzonesize & uybinned_c1(probeindices{p})<=currentreward+rewzonesize)+probeindices{p}(1)-1; %find for this epoch, indices in the reward zone category
                if ~isempty(temp1)
                    probe_in_rew_licksind = [probe_in_rew_licksind temp1(find(filteredlicks(temp1)))];
                end
                
                temp1 = find(uybinned_c1(probeindices{p})>currentreward+rewzonesize)+probeindices{p}(1)-1; %find for this epoch, indices in the reward zone category
                if ~isempty(temp1)
                    probe_post_rew = [probe_post_rew temp1(find(filteredlicks(temp1)))];
                end
            end
            else
            end
            
            
            
            % failed Lickversion
            cumtrialnum = utrialnum_c1; %first setup a trial counter that isn't epoch reset
            temp = find(diff(cumtrialnum)<-1);
            for t = 1:length(temp)
                cumtrialnum(temp(t)+1:end) = utrialnum_c1(temp(t)+1:end)+cumtrialnum(temp(t))+1;
            end
            successindices = find(ismember(cumtrialnum,cumtrialnum(find(urewards)))); %find the trials with rewards
            probeindex = find(ismember(utrialnum_c1,0:numprobes-1)); %find the trials with trialnumber less than number of probes
            [~,failedindices] = setxor(1:length(cumtrialnum),[successindices probeindex]); %find the rest which must be failed trials
            
            
            failed_pre_rew_far_licksind = [];
            failed_pre_rew_near_licksind = [];
            failed_post_rew = [];
            
            
            if ~isempty(failedindices)
            
            temp = [find(uchangeRewLoc_c1) length(uchangeRewLoc_c1)];
            for r = 1:length(temp)-1
                tempfail = failedindices((failedindices>=temp(r)&failedindices<temp(r+1)));
                if ~isempty(tempfail)
                temp1 = tempfail(find(uybinned_c1(tempfail)<uchangeRewLoc_c1(temp(r))-rewzonesize-nearzone));
                if ~isempty(temp1)
                    failed_pre_rew_far_licksind = [failed_pre_rew_far_licksind temp1(find(filteredlicks(temp1)))'];
                end
                
                temp1 = tempfail(find(uybinned_c1(tempfail)>=uchangeRewLoc_c1(temp(r))-rewzonesize-nearzone&uybinned_c1(tempfail)<uchangeRewLoc_c1(temp(r))-rewzonesize));
                if ~isempty(temp1)
                    failed_pre_rew_near_licksind = [failed_pre_rew_near_licksind temp1(find(filteredlicks(temp1)))'];
                end
                
                
                temp1 = tempfail(find(uybinned_c1(tempfail)>uchangeRewLoc_c1(temp(r))+rewzonesize));
                if ~isempty(temp1)
                    failed_post_rew = [failed_post_rew temp1(find(filteredlicks(temp1)))'];
                end
                end
                
            end
            
            end
            
            
            
            %success lickversion
            cumtrialnum = utrialnum_c1; %first setup a trial counter that isn't epoch reset
            temp = find(diff(cumtrialnum)<-1);
            for t = 1:length(temp)
                cumtrialnum(temp(t)+1:end) = utrialnum_c1(temp(t)+1:end)+cumtrialnum(temp(t))+1;
            end
            successindices = find(ismember(cumtrialnum,cumtrialnum(find(urewards)))); %find the trials with rewards
            probeindex = find(ismember(utrialnum_c1,0:numprobes-1)); %find the trials with trialnumber less than number of probes
            [~,failedindices] = setxor(1:length(cumtrialnum),[successindices probeindex]); %find the rest which must be failed trials
            
            
            success_pre_rew_far_licksind = [];
            success_pre_rew_near_licksind = [];
            success_in_rew_licksind_pre_us = [];
            success_in_rew_licksind_post_us = [];
            success_post_rew = [];
            
            if ~isempty(successindices)
                temp = [find(uchangeRewLoc_c1) length(uchangeRewLoc_c1)];
                for r = 1:length(temp)-1
                    tempsuccess = successindices((successindices>=temp(r)&successindices<temp(r+1)));
                    
                    temp1 = tempsuccess(find(uybinned_c1(tempsuccess)<uchangeRewLoc_c1(temp(r))-rewzonesize-nearzone));
                    if ~isempty(temp1)
                        success_pre_rew_far_licksind = [success_pre_rew_far_licksind temp1(find(filteredlicks(temp1)))];
                    end
                    
                    
                    temp1 = tempsuccess(find(uybinned_c1(tempsuccess)>=uchangeRewLoc_c1(temp(r))-rewzonesize-nearzone&uybinned_c1(tempsuccess)<uchangeRewLoc_c1(temp(r))-rewzonesize));
                    if ~isempty(temp1)
                        success_pre_rew_near_licksind = [success_pre_rew_near_licksind temp1(find(filteredlicks(temp1)))];
                    end
                    
%                     temprew = consecutive_stretch(tempsuccess(find(uybinned_c1(tempsuccess)>=uchangeRewLoc_c1(temp(r))-rewzonesize & uybinned_c1(tempsuccess)<=uchangeRewLoc_c1(temp(r))+rewzonesize)));
                    temprew = num2cell((find(urewards(temp(r):temp(r+1))==1)+temp(r)-1));
                    for tr = 1:length(temprew)
                        currenttrialind = find(cumtrialnum == cumtrialnum(temprew{tr}));
                        temprew{tr} = (find(uybinned_c1(currenttrialind)>=uchangeRewLoc_c1(temp(r))-rewzonesize,1):find(uybinned_c1(currenttrialind)>uchangeRewLoc_c1(temp(r))+rewzonesize,1,'last'))+currenttrialind(1)-1;
                        temp1 = temprew{tr}(1:find(urewards(temprew{tr})==1)-1);
                        if ~isempty(temp1)
                            success_in_rew_licksind_pre_us = [success_in_rew_licksind_pre_us temp1(find(filteredlicks(temp1)))];
                        end
                        
                        
                        temp1 = temprew{tr}(find(urewards(temprew{tr})==1):end);
                        if ~isempty(temp1)
                            success_in_rew_licksind_post_us = [success_in_rew_licksind_post_us temp1(find(filteredlicks(temp1)))];
                        end
                        
                        
                    end
                    
                    
                    temp1 = tempsuccess(find(uybinned_c1(tempsuccess)>uchangeRewLoc_c1(temp(r))+rewzonesize));
                    if ~isempty(temp1)
                        success_post_rew = [success_post_rew temp1(find(filteredlicks(temp1)))];
                    end
                    
                end
                
                
                
            end
          
            
            differentlicks = {'success_pre_rew_far_licksind',...
            'success_pre_rew_near_licksind',...
            'success_in_rew_licksind_pre_us'...
            'success_in_rew_licksind_post_us'...
            'success_post_rew'...
            'probe_pre_rew_far_licksind'...
            'probe_pre_rew_near_licksind'...
            'probe_in_rew_licksind'...
            'probe_post_rew'...
            'failed_pre_rew_far_licksind'...
            'failed_pre_rew_near_licksind'...
            'failed_post_rew'};
        
            figure; plot(uybinned_c1);
            hold on; plot(find(filteredlicks),uybinned_c1(find(filteredlicks)),'r.')
            colors = distinguishable_colors(length(differentlicks));
            lickexist = zeros(size(differentlicks));
            for d = 1:length(differentlicks)
                plot(eval(differentlicks{d}),uybinned_c1(eval(differentlicks{d})),'o','Color',colors(d,:))
                if ~isempty(eval(differentlicks{d}))
                    lickexist(d) = 1;
                end
            end
            
            plot(find(usolenoid2),uybinned_c1(find(usolenoid2)),'m*','MarkerSize',10)
            plot(find(urewards),uybinned_c1(find(urewards)),'y*','MarkerSize',10)
            temp = [find(uchangeRewLoc_c1) length(uchangeRewLoc_c1)];
            for r = 1:length(temp)-1
                plot([temp(r) temp(r+1)-1],[uchangeRewLoc_c1(temp(r)) uchangeRewLoc_c1(temp(r))] - rewzonesize,'k--')
                
                plot([temp(r) temp(r+1)-1],[uchangeRewLoc_c1(temp(r)) uchangeRewLoc_c1(temp(r))] + rewzonesize,'k--')
            end
            legend([{'yposition','filtered licks'} differentlicks(lickexist==1) {'CS','Rewards','Zones'}],'Interpreter','none','Location','northeastoutside')
            
            saveas(gcf,[pr_dir0{Day} '\Lick_Categories_vs_position.fig'])
            close
            
            end
   %%         
            
            
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
            speed_thresh = 5; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 5; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
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
            speedftol=10;
            max_nrew_stop_licktol=2*frame_rate;
            
            
            uforwardvel_c1 = smoothdata(uforwardvel,'gaussian',30);
            
            
%             mean_base_mean=mean(base_mean);
%             
%             norm_base_mean=base_mean;
            

            %%%%%%%%%%%%%%%%%%%%%%%%%%% for regions of interest
            
            if oldbatch==1
                
                df_f=params.roibasemean2;
            else
                df_f=params.roibasemean3;
            end
            
            
            % initialize peri variables for all lick types
            for d = 1:length(differentlicks)
                temp1 = differentlicks{d};
                temp = temp1;
                [outstart,outstop] = regexp(temp,'_licksind');
                if ~isempty(outstart) % remove mention of licks ind for shorter precise variable names
                temp(outstart:outstop) = [];
                end
                eval(['dop_' temp ' = cell(size(df_f,1),length(' differentlicks{d} '));']); %initialize dopamine variables
                eval(['dopnorm_' temp ' = cell(size(df_f,1),length(' differentlicks{d} '));']); %initialize pre window normalized dopamine variables
                eval(['doptime_'  temp ' = cell(1,length(' differentlicks{d} '));']); %initialize time matrix for dopamine variables
                eval(['Spd_' temp ' = cell(1,length(' differentlicks{d} '));']); % initialize speed variable
                eval(['Spdtime_' temp ' = cell(1,length(' differentlicks{d} '));']); % initialize speed time variable
                
                
%                 temp1 = differentlicks{d};
                
                for roii = 1:size(df_f,1)
                    roibase_mean = df_f{roii,1};
                    roimean_base_mean=mean(df_f{roii,1});
                    roinorm_base_mean=roibase_mean/roimean_base_mean;
                    roidop_smth=smoothdata(roinorm_base_mean,'gaussian',gauss_win);
                    currenttrigger = eval(temp1);
                    
                    
                    
                    for i=1:length(currenttrigger)
                        currentdopindex = find(timedFF>=uVRtimebinned(currenttrigger(i))-pre_win& timedFF<= uVRtimebinned(currenttrigger(i))+post_win); % find the time points we can use for the current trigger in dopamine time
                        currentVRindex = find(uVRtimebinned>= uVRtimebinned(currenttrigger(i))-pre_win & uVRtimebinned <= uVRtimebinned(currenttrigger(i))+post_win); % find the time points we can use for the current trigger in VR time
                        eval(['doptime_'  temp '{i} = transpose(timedFF(currentdopindex)-uVRtimebinned(currenttrigger(i)));']); 
                        eval(['dop_' temp '{roii,i} = roidop_smth(currentdopindex);']); 
                        eval(['dopnorm_' temp '{roii,i} = roidop_smth(currentdopindex)/nanmean(roidop_smth(currentdopindex(find(doptime_'  temp '{i}<=0))));']); 
                        eval(['Spd_' temp '{i} = transpose(uforwardvel_c1(currentVRindex));']); 
                        eval(['Spdtime_' temp '{i} = transpose(uVRtimebinned(currentVRindex)-uVRtimebinned(currenttrigger(i)));']);
                  
                    end
    
                    %combining per day
                    
                    temp1 = differentlicks{d};
                    temp = temp1;
                    [outstart,outstop] = regexp(temp,'_licksind');
                    if ~isempty(outstart) % remove mention of licks ind for shorter precise variable names
                        temp(outstart:outstop) = [];
                    end
                    if allplanes == 1
                        eval(['dop_allday_' temp ' = cell(0);']); %initialize dopamine variables
                        eval(['dopnorm_allday_' temp ' = cell(0);']); %initialize pre window normalized dopamine variables
                        eval(['doptime_allday_'  temp ' = cell(0);']); %initialize time matrix for dopamine variables
                        eval(['Spd_allday_' temp ' = cell(0);']); % initialize speed variable
                        eval(['Spdtime_allday_' temp ' = cell(0);']); % initialize speed time variable
                    end
                    
                    
                    eval(['dop_allday_' temp ' = cat(2,dop_allday_' temp ',padcatcell2mat(dop_' temp '(roii,:)));']); %initialize dopamine variables
                    eval(['dopnorm_allday_' temp ' = cat(2,dopnorm_allday_' temp ',padcatcell2mat(dopnorm_' temp '(roii,:)));']); %initialize pre window normalized dopamine variables
                    
                    eval(['doptime_allday_' temp ' = cat(2,doptime_allday_' temp ',padcatcell2mat(doptime_' temp '));']); %initialize time matrix for dopamine variables
                    eval(['Spd_allday_' temp ' = cat(2,Spd_allday_' temp ',padcatcell2mat(Spd_' temp '));']); % initialize speed variable
                    eval(['Spdtime_allday_' temp ' = cat(2,Spdtime_allday_' temp ',padcatcell2mat(Spdtime_' temp '));']); % initialize speed time variable
                    
        
                    
                    
                    
                end
                
                
                
            end
            
            

        end  
    end
    % compiling all days
    for d = 1:length(differentlicks)
        temp1 = differentlicks{d};
        temp = temp1;
        [outstart,outstop] = regexp(temp,'_licksind');
        if ~isempty(outstart) % remove mention of licks ind for shorter precise variable names
            temp(outstart:outstop) = [];
        end
        
        
        eval(['ALL_dop_allday_' temp '{Day} = padcatcell2mat(reshape(dop_allday_' temp ',1,1,length(dop_allday_' temp ')));']); %initialize dopamine variables
        eval(['ALL_dopnorm_allday_' temp '{Day} = padcatcell2mat(reshape(dopnorm_allday_' temp ',1,1,length(dopnorm_allday_' temp ')));']); %initialize pre window normalized dopamine variables
        eval(['ALL_doptime_allday_' temp '{Day} = padcatcell2mat(reshape(doptime_allday_' temp ',1,1,length(doptime_allday_' temp ')));']); %initialize time matrix for dopamine variables
        eval(['ALL_Spd_allday_' temp '{Day} = padcatcell2mat(reshape(Spd_allday_' temp ',1,1,length(Spd_allday_' temp ')));']); % initialize speed variable
        eval(['ALL_Spdtime_allday_' temp '{Day} = padcatcell2mat(reshape(Spdtime_allday_' temp ',1,1,length(Spdtime_allday_' temp ')));']);
        
        
    end
    
end

                 
%%
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
% 

