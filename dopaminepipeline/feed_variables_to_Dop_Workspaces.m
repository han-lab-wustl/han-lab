clear all
close all
wrk_dir = uipickfiles('Prompt','Pick the Workspace you would like to add');

load(wrk_dir{1})
msgbox(pr_dir0(:))
day_dir = uipickfiles('Prompt','Please Select all the folders for the previous popup');
analysischeck = [12];
% add the numbers of the analyses you want to preform to analysischeck
% analysis 1 : CS to first lick latency over days
% analysis 2 :  CS to stop latency over days
% analysis 3 : divide loccomotion and calculate correlation
% analysis 4 : CS to transient over days attempt 1
% analysis 5 : CS to transient over days attempt 2
% analysis 7 : Plot the first 2 minutes of every day with speed and dFF
% analysis 8 : 1st Two Minutes Non Rewarded Stops with Not With Licks
% analysis 9 : (MUST HAVE 1 and 2 ON) plotting CS latency with CS-US
% average over days
% analysis 10 : non rew licks with and without stopping
% analysis 11 : unrewarded starts, unrewarded starts after licks,
% unrewarded starts after no licks
% analysis 12 : lick bout analysis
oldbatch = 0;
%%
if mouse_id == 181
    daykey = [];
    daykey(1:18,1:3) = repmat(2:4,18,1);
    daykey(19:23,1:3) = repmat(1:3,5,1);
end

if sum(analysischeck == 1)>0 % do CS to first lick latency analysis
CS_alldays_lick_gap = cell(length(day_dir),1);
test2 = cell(length(day_dir),1);
end
if sum(analysischeck == 2)>0 % do CS to stop latency analysis
CS_alldays_stop_gap = cell(length(day_dir),1);
end
 if sum(analysischeck == 3)>0 % do Chopped up Loccomition Correlation analysis
     locvars = {'stoppedblocks','deccel2stop','deccel2movfast','deccel2movslow','accelfromstop','accelfrommovfast','accelfrommoveslow','maintainspeed','bufferedmoving'};
     for va = 1:length(locvars)
      eval([locvars{va} '_corr_alldays_planes = cell(length(day_dir),1);']);
                    eval(['roi_dop_' locvars{va} '_alldays_planes = cell(length(day_dir),1);']);
                    eval(['roi_roe_' locvars{va} '_alldays_planes = cell(length(day_dir),1);']);

     end
 end
 
 if sum(analysischeck == 4)>0
     alldFFs = cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
     allpostrans = cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
     allnegtrans = cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
     allutime = cell(length(day_dir),1);
     allCS = cell(length(day_dir),1);
     alltime =  cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
 end
  if sum(analysischeck == 7)>0
     alldFFs = cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
     allutime = cell(length(day_dir),1);
     allCS = cell(length(day_dir),1);
     alltime =  cell(length(day_dir),size(roi_dop_alldays_planes_periCS,2));
     allspeed = cell(length(day_dir),1);
     alllicks = cell(length(day_dir),1);
     allstoppedblocks = cell(length(day_dir),1);
  end
 
   if sum(analysischeck == 8)>0
       roi_dop_allsuc_nolick_stop_no_reward2min = cell(size(roi_dop_alldays_planes_success_stop_no_reward));
       roi_dop_allsuc_lick_stop_no_reward2min = cell(size(roi_dop_alldays_planes_success_stop_no_reward));
       roi_roe_allsuc_nolick_stop_no_reward2min = cell(size(roe_alldays_planes_success_stop_no_reward));
       roi_roe_allsuc_lick_stop_no_reward2min = cell(size(roe_alldays_planes_success_stop_no_reward));
   end
   
   if sum(analysischeck == 9) >0
        roi_roe_alldays_planes_perilick_no_reward_stop = {};
       roi_roe_allsuc_lick_no_reward_stop=[];
       
       roi_dop_alldays_planes_perilick_no_reward_stop = {};
       roi_dop_allsuc_lick_no_reward_stop=[];
       
       
       roi_roe_alldays_planes_perilick_no_reward_mov = {};
       roi_roe_allsuc_lick_no_reward_mov=[];
       
       roi_dop_alldays_planes_perilick_no_reward_mov = {};
       roi_dop_allsuc_lick_no_reward_mov=[];
       
   end
   
   if sum(analysischeck == 11)>0

                        roi_roe_alldays_planes_success_mov_no_reward_lick = {};
                        roi_roe_allsuc_mov_no_reward_lick=[];
                 
                        roi_roe_alldays_planes_success_mov_no_reward_no_lick = {};
                        roi_roe_allsuc_mov_no_reward_no_lick=[];
                 
                        roi_dop_alldays_planes_success_mov_no_reward_lick = {};
                       roi_dop_allsuc_mov_no_reward_lick=[];

                        roi_dop_alldays_planes_success_mov_no_reward_no_lick = {};
                       roi_dop_allsuc_mov_no_reward_no_lick=[];
  
   
   end
   
   if sum(analysischeck == 12)>0
                        roi_roe_alldays_planes_success_perilickboutstart = {};
                        roi_roe_allsuc_perilickboutstart=[];
                
                        roi_dop_alldays_planes_success_perilickboutstart= {};
                       roi_dop_allsuc_perilickboutstart = [];
                       
                       number_licks_per_bout = {};
   end
   
 if ~exist('daykey')
     daykey = repmat(1:size(roi_dop_alldays_planes_periCS,2),length(day_dir),1);
 end




for d = 1:length(day_dir)
    dir_s2p = struct2cell(dir([day_dir{d} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));

    Fs = 31.25/size(planefolders,2);
    if sum(analysischeck == 4)>0 %doing CS to transient analysis
    g = 0; % roi counter for aligning accross rois and days
    end
     if sum(analysischeck == 3)>0 %doing speed corr to chopping
    j = 0; % roi counter for aligning accross rois and days
    aj = 0;
     end
      if sum(analysischeck == 7)>0 %doing 1st two min plot
    k = 0; % roi counter for aligning accross rois and days
      end
      if sum(analysischeck == 10)>0
          roiplane1 = 0; % roi counter for stopped/moving unrew licks
      end
      if sum(analysischeck == 11)>0
          roiplane2 = 0; % roi counter for non rew moving with/without licks
      end
      if sum(analysischeck == 12)>0
          roiplane3 = 0;
      end
      
for allplanes=1:size(planefolders,2)

        dir_s2p = struct2cell(dir([day_dir{d} '\**\suite2p']));
    planefolders = dir_s2p(:,~cellfun(@isempty,regexp(dir_s2p(1,:),'plane')));
        pr_dir2=strcat(planefolders{2,allplanes},'\plane',num2str(allplanes-1),'\reg_tif\')
        if sum(analysischeck == 11)>0
        clearvars nonrew_mov_success_tmpts
        end
        load([pr_dir2 'params.mat'])
        CSlickgap = [];
    if allplanes == 1
        if oldbatch == 1
            short = rewardsALL;
            licksALL = lickVoltageALL<-0.08;
        else
        short = solenoid2ALL;
        end
        if sum(analysischeck == 4)>0 %save behavior for CS to trans analysis
        allutime{d} = utimedFF;
        allCS{d} = short;
        end
         if sum(analysischeck == 7)>0 %save behavior for CS to trans analysis
             gauss_win = 5;
        speed_smth_1=smoothdata(forwardvelALL,'gaussian',gauss_win)';
        allutime{d} = utimedFF;
        allCS{d} = short;
        allspeed{d} = speed_smth_1;
        alllicks{d} = licksALL;
        end
        CSs = find(short);
        CSs = cellfun(@(x) x(1),consecutive_stretch(CSs),'UniformOutput',1);
        
        if sum(analysischeck == 1)>0 % do CS to first lick latency analysis
            CSlickgap = [];
            test = [];
            for cs = 1:length(CSs)-1
                if ~isempty(find(licksALL(CSs(cs):CSs(cs+1)),1))
                    if find(licksALL(CSs(cs):CSs(cs+1)),1)/31.25 < 10
                CSlickgap(cs) = find(licksALL(CSs(cs):CSs(cs+1)),1)/31.25;
                test(cs) = find(licksALL(CSs(cs):CSs(cs+1)),1)+CSs(cs);
                    else
                        CSlickgap(cs) = NaN;
                    end
                else
                    CSlickgap(cs) = NaN;
                    
                end
            end
            if ~isempty(find(licksALL(CSs(cs+1):end),1))
            CSlickgap(cs+1) = find(licksALL(CSs(cs+1):end),1)/31.25;
            test(cs) = find(licksALL(CSs(cs+1):end),1)+CSs(cs);
            else
                CSlickgap(cs+1) = NaN;
            end

            CS_alldays_lick_gap{d} = CSlickgap; 
            test2{d} = single_lick_gap/31.25;
        end
        
        
        if sum(analysischeck == 2)>0 % do CS to stop latency analysis
            CSstopgap = [];
            for rs = 1:length(rew_stop_success_tmpts)
                [~,nearestCS] = min(abs(CSs-rew_stop_success_tmpts(rs)));
                if CSs(nearestCS)-rew_stop_success_tmpts(rs)>=0
                    CSstopgap(rs) = 0;
                else
                    CSstopgap(rs) = (rew_stop_success_tmpts(rs)-CSs(nearestCS))/31.25;
                end
            end
            CS_alldays_stop_gap{d} = CSstopgap;
        end
        
        
    end
    
    if sum(analysischeck == 3)>0 % do Chopped up Loccomition Correlation analysis
        gauss_win = 5;
        speed_smth_1=smoothdata(forwardvelALL,'gaussian',gauss_win)';
        if allplanes == 1
            stoppedblocks = {};
            if stop_success_tmpts(1)>mov_success_tmpts(1)
                stop_success_tmpts = [1 stop_success_tmpts];
            end
            for s = 1:length(stop_success_tmpts)
                if length(mov_success_tmpts)>= s
                stoppedblocks{s} = stop_success_tmpts(s):mov_success_tmpts(s);
                else
                    stoppedblocks{s} = stop_success_tmpts(s):length(forwardvelALL);
                end
            end
            
            bufferedstoppedblocks = stoppedblocks;
            for s = 1:length(bufferedstoppedblocks)
                bufferedstoppedblocks{s} = bufferedstoppedblocks{s}(1)-round(2*31.25):bufferedstoppedblocks{s}(end)+round(2*31.25);
            end
            [~,bufferedmoving] = setxor(1:length(forwardvelALL),cell2mat(bufferedstoppedblocks));
            bufferedmoving = consecutive_stretch(bufferedmoving);
            bufferedmoving(cellfun(@length,bufferedmoving)<round(1*31.25)) = [];
            accel = [diff(smoothdata(speed_smth_1,'gaussian',round(2*31.25))); 0];
            
            decceling = consecutive_stretch(find(accel<-0.35));
            decceling(cellfun(@length,decceling)<16) = [];
            
            deccel2stop = {};
            deccel2movfast = {};
            deccel2movslow = {};
            for dcs = 1:length(decceling)
                if ~isempty(intersect(decceling{dcs},stop_success_tmpts))
                    deccel2stop = [deccel2stop {decceling{dcs}}];
                else
                    if nanmean(speed_smth_1(decceling{dcs}(end-5:end)))>10
                    deccel2movfast = [deccel2movfast {decceling{dcs}}];
                    else
                        deccel2movslow = [deccel2movslow {decceling{dcs}}];
                    end
                end
            end
            
            acceling = consecutive_stretch(find(accel>0.13));
            acceling(cellfun(@length,acceling)<31.25) = [];
            
            accelfromstop = {};
            accelfrommovfast = {};
            accelfrommovslow = {};
            for acs = 1:length(acceling)
                if ~isempty(intersect(acceling{acs}(1)-7:acceling{acs}(end),mov_success_tmpts)) 
            accelfromstop = [accelfromstop {acceling{acs}}];
                else
                    if nanmean(speed_smth_1(acceling{acs}(1:5)))>10
            accelfrommovfast = [accelfrommovfast {acceling{acs}}];
                    else
                        accelfrommovslow = [accelfrommovslow {acceling{acs}}];
                    end
                end
            end
            
            maintainspeed = consecutive_stretch(setxor(1:length(accel),[cell2mat(acceling'); cell2mat(decceling') ; cell2mat(stoppedblocks)']));
            maintainspeed(cellfun(@length,maintainspeed)<16) = [];
            
            if allplanes == 1
            figure;
            plot(accel*40,'Color',[0.6 0 0])
            hold on
            plot(speed_smth_1,'Color',[0 0.4 1])
            hold on
            colors = distinguishable_colors(20);
            for cd = 1:length(stoppedblocks)
            plot(stoppedblocks{cd},speed_smth_1(stoppedblocks{cd}),'Color',colors(10,:),'LineWidth',1.5)
            end
            for cd = 1:length(deccel2stop)
            plot(deccel2stop{cd},speed_smth_1(deccel2stop{cd}),'Color',colors(9,:),'LineWidth',2)
            end
            for cd = 1:length(deccel2movfast)
            plot(deccel2movfast{cd},speed_smth_1(deccel2movfast{cd}),'Color',colors(12,:),'LineWidth',1.5)
            end
            for cd = 1:length(deccel2movslow)
            plot(deccel2movslow{cd},speed_smth_1(deccel2movslow{cd}),'Color',colors(13,:),'LineWidth',1.5)
            end
            for cd = 1:length(accelfromstop)
            plot(accelfromstop{cd},speed_smth_1(accelfromstop{cd}),'Color',colors(14,:),'LineWidth',1.5)
            end
            for cd = 1:length(accelfrommovfast)
            plot(accelfrommovfast{cd},speed_smth_1(accelfrommovfast{cd}),'Color',colors(15,:),'LineWidth',1.5)
            end
            for cd = 1:length(accelfrommovslow)
            plot(accelfrommovslow{cd},speed_smth_1(accelfrommovslow{cd}),'Color',colors(16,:),'LineWidth',1.5)
            end
            plot(mov_success_tmpts,speed_smth_1(mov_success_tmpts),'k.','MarkerSize',20)
hold on; plot(stop_success_tmpts,speed_smth_1(stop_success_tmpts),'r.','MarkerSize',20)
            
            figure;
            plot(accel*40,'Color',[0.6 0 0])
            hold on
            plot(speed_smth_1,'Color',[0 0.4 1])
            hold on
            for cd = 1:length(bufferedmoving)
            plot(bufferedmoving{cd},speed_smth_1(bufferedmoving{cd}),'r-')
            end
            end
%             
% 
%             plot(cell2mat(maintainspeed'),speed_smth_1(cell2mat(maintainspeed')),'.')
        end
            locvars = {'stoppedblocks','deccel2stop','deccel2movfast','deccel2movslow','accelfromstop','accelfrommovslow','accelfrommovfast','maintainspeed','bufferedmoving'};
              if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
            
             for roii = 1:size(df_f,1)
                 j = j+1;
                 roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                for va = 1:length(locvars)
                    daycorr = [];
                    dopavg = [];
                    roeavg = [];
                    daychunkedcorr = [];
                    if va == 1
                    currvar = eval(locvars{va});
                    else
                        currvar = eval(['transpose(' locvars{va} ');']);
                    end
                    planeindexs = allplanes:size(planefolders,2):length(speed_smth_1);
                    
                    [currtemp,~,dopcurrtemp] = intersect(cell2mat(currvar),planeindexs);
                    if ~isempty(currvar)
                    daycorr = corr(speed_smth_1(currtemp),roinorm_base_mean(dopcurrtemp));
                        dopavg = nanmean(roinorm_base_mean(dopcurrtemp));
                        roeavg = nanmean(speed_smth_1(currtemp));
                        if strcmp(locvars{va},'bufferedmoving')
                    for bo = 1:length(currvar)
                        [currtemp,~,dopcurrtemp] = intersect(currvar{bo},planeindexs);
                        daychunkedcorr(bo) = corr(speed_smth_1(currtemp),roinorm_base_mean(dopcurrtemp));
                    end
                    end
                        
                    else
                        daycorr = NaN;
                        dopavg = NaN;
                        roeavg = NaN;
                         if strcmp(locvars{va},'bufferedmoving')
                             daychunkedcorr(1) = NaN;
                         end
                    end
                    
                    eval([locvars{va} '_corr_alldays_planes(d,j) = {daycorr};']);
                   if strcmp(locvars{va},'bufferedmoving')
                       eval([locvars{va} '_corr_alldays_planes_chunked(d,j) = {daychunkedcorr};']);
                   end
                    eval(['roi_dop_' locvars{va} '_alldays_planes(d,j) = {dopavg};']);
                    if allplanes == 1 && roii == 1
                    eval(['roi_roe_' locvars{va} '_alldays_planes(d,j) = {roeavg};']);
                    end
                end
             end
             
             %do all the corrs but for Acceleration instead
             
             locvars = {'stoppedblocks','deccel2stop','deccel2movfast','deccel2movslow','accelfromstop','accelfrommovslow','accelfrommovfast','maintainspeed','bufferedmoving'};
              if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
            
             for roii = 1:size(df_f,1)
                 aj = aj+1;
                 roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                for va = 1:length(locvars)
                    daycorr = [];
                    dopavg = [];
                    roeavg = [];
                    daychunkedcorr = [];
                    if va == 1
                    currvar = eval(locvars{va});
                    else
                        currvar = eval(['transpose(' locvars{va} ');']);
                    end
                    planeindexs = allplanes:size(planefolders,2):length(speed_smth_1);
                    
                    [currtemp,~,dopcurrtemp] = intersect(cell2mat(currvar),planeindexs);
                    if ~isempty(currvar)
                    daycorr = corr(accel(currtemp),roinorm_base_mean(dopcurrtemp));
                        dopavg = nanmean(roinorm_base_mean(dopcurrtemp));
                        roeavg = nanmean(accel(currtemp));
                        if strcmp(locvars{va},'bufferedmoving')
                    for bo = 1:length(currvar)
                        [currtemp,~,dopcurrtemp] = intersect(currvar{bo},planeindexs);
                        daychunkedcorr(bo) = corr(accel(currtemp),roinorm_base_mean(dopcurrtemp));
                    end
                    end
                        
                    else
                        daycorr = NaN;
                        dopavg = NaN;
                        roeavg = NaN;
                         if strcmp(locvars{va},'bufferedmoving')
                             daychunkedcorr(1) = NaN;
                         end
                    end
                    
                    eval([locvars{va} '_accel_corr_alldays_planes(d,aj) = {daycorr};']);
                   if strcmp(locvars{va},'bufferedmoving')
                       eval([locvars{va} '_accel_corr_alldays_planes_chunked(d,aj) = {daychunkedcorr};']);
                   end
                    eval(['roi_dop_' locvars{va} '_accel_alldays_planes(d,aj) = {dopavg};']);
                    if allplanes == 1 && roii == 1
                    eval(['roi_roe_' locvars{va} '_accel_alldays_planes(d,aj) = {roeavg};']);
                    end
                end
             end
             
             
            
    end
    
    if sum(analysischeck == 4)>0 % do CS to Transient analysis
         if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
             for roii = 1:length(df_f)
                 g = g+1;
         alldFFs{d,g} = df_f{roii};
         alltime{d,g} = timedFF;
                startbaseline = 60;
                stopbaseline = 120;
                smthtrace = smoothdata(df_f{roii},'gaussian',round(2*31.25/size(planefolders,2)));
                upperstart = nanmean(smthtrace(timedFF>startbaseline&timedFF<stopbaseline))+1.5*nanstd(smthtrace(timedFF>startbaseline&timedFF<stopbaseline));
                upperstop = nanmean(smthtrace(timedFF>startbaseline&timedFF<stopbaseline))+0.5*nanstd(smthtrace(timedFF>startbaseline&timedFF<stopbaseline));
                lowerstart = nanmean(smthtrace(timedFF>startbaseline&timedFF<stopbaseline))-1.5*nanstd(smthtrace(timedFF>startbaseline&timedFF<stopbaseline));
                lowerstop = nanmean(smthtrace(timedFF>startbaseline&timedFF<stopbaseline))-0.5*nanstd(smthtrace(timedFF>startbaseline&timedFF<stopbaseline));
                postrans = double_thresh(smthtrace,upperstart,upperstop);
                negtrans = double_thresh(-1*smthtrace,-1*lowerstart,-1*lowerstop);
                transdur = consecutive_stretch(find(postrans));
                postrans(cell2mat(transdur(cellfun(@length,transdur)<round(0.5*31.25/size(planefolders,2))))) = 0;
                transdur = consecutive_stretch(find(negtrans));
                negtrans(cell2mat(transdur(cellfun(@length,transdur)<round(0.5*31.25/size(planefolders,2))))) = 0;
                
               allpostrans{d,g} = postrans;
               allnegtrans{d,g} = negtrans;
             end
    end
    
    
    if sum(analysischeck == 7)>0 % do Start 2 min analysis
         if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
         
         if allplanes == 1
              stoppedblocks = {};
            if stop_success_tmpts(1)>mov_success_tmpts(1)
                stop_success_tmpts = [1 stop_success_tmpts];
            end
            for s = 1:length(stop_success_tmpts)
                if length(mov_success_tmpts)>= s
                stoppedblocks{s} = stop_success_tmpts(s):mov_success_tmpts(s);
                else
                    stoppedblocks{s} = stop_success_tmpts(s):length(forwardvelALL);
                end
            end
            allstoppedblocks{d} = stoppedblocks;
         end
             for roii = 1:length(df_f)
                 k = k+1;
                 alldFFs{d,k} = df_f{roii};
                 alltime{d,k} = timedFF;
             end
    end
    
    
    if sum(analysischeck == 8)>0 % do Start 2 min analysis
        if allplanes == 1
        lick_idx = find(licksALL);
                    lick_norew_stop_success_tmpts = []; nolick_norew_stop_success_tmpts = [];
                    lickstamps=0; nolickstamps=0;
                    for ii=1:length(nonrew_stop_success_tmpts)
                        
                        if ~isempty(find(licksALL(nonrew_stop_success_tmpts(ii)-max_nrew_stop_licktol:nonrew_stop_success_tmpts(ii)+  max_nrew_stop_licktol)))
                            lickstamps=lickstamps+1;
                            lick_norew_stop_success_tmpts(lickstamps)=nonrew_stop_success_tmpts(ii);
                        else
                            nolickstamps=nolickstamps+1;
                            nolick_norew_stop_success_tmpts(nolickstamps)=nonrew_stop_success_tmpts(ii);
                        end
                    end
                    
                [~,Nolick2minStops] = intersect(nonrew_stop_success_tmpts,nolick_norew_stop_success_tmpts);
                Nolick2minStops(utimedFF(nonrew_stop_success_tmpts(Nolick2minStops))>120) = [];
                 [~,lick2minStops] = intersect(nonrew_stop_success_tmpts,lick_norew_stop_success_tmpts);
                lick2minStops(utimedFF(nonrew_stop_success_tmpts(lick2minStops))>120) = [];
                if ~isempty(Nolick2minStops)
                    roi_dop_allsuc_nolick_stop_no_reward2min(d,:) = cellfun(@(x) x(Nolick2minStops,:),roi_dop_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                    roi_roe_allsuc_nolick_stop_no_reward2min(d,:) = cellfun(@(x) x(Nolick2minStops,:),roe_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                else
                    roi_dop_allsuc_nolick_stop_no_reward2min(d,:) = cellfun(@(x) nan(size(x(1,:))),roi_dop_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                    roi_roe_allsuc_nolick_stop_no_reward2min(d,:) = cellfun(@(x) nan(size(x(1,:))),roe_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                end
                if ~isempty(lick2minStops)
                    roi_dop_allsuc_lick_stop_no_reward2min(d,:) = cellfun(@(x) x(lick2minStops,:),roi_dop_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                    roi_roe_allsuc_lick_stop_no_reward2min(d,:) =  cellfun(@(x) x(lick2minStops,:),roe_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                else
                    roi_dop_allsuc_lick_stop_no_reward2min(d,:) = cellfun(@(x) nan(size(x(1,:))),roi_dop_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                    roi_roe_allsuc_lick_stop_no_reward2min(d,:) = cellfun(@(x) nan(size(x(1,:))),roe_alldays_planes_success_stop_no_reward(d,:),'UniformOutput',0);
                end
        end
                    
    end
    
    if sum(analysischeck == 10)>0 % do nonrew stopped or moving lick analysis
        
           
            numplanes=size(planefolders,2);
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
            speed_thresh = 7.5; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 5; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            if oldbatch == 1
            CSUStimelag = 0;
            else
            CSUStimelag = 0.5; %seconds between
            end
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time/numplanes);
            post_win_framesALL=round(post_win/frame_time);
            pre_win_framesALL = round(pre_win/frame_time);
            pre_win_frames=round(pre_win/frame_time/numplanes);
        
        
        
        
        if allplanes == 1
           stoppedblocks = {};
            if stop_success_tmpts(1)>mov_success_tmpts(1)
                stop_success_tmpts = [1 stop_success_tmpts];
            end
            for s = 1:length(stop_success_tmpts)
                if length(mov_success_tmpts)>= s
                stoppedblocks{s} = stop_success_tmpts(s):mov_success_tmpts(s);
                else
                    stoppedblocks{s} = stop_success_tmpts(s):length(forwardvelALL);
                end
            end
             nr_lick_idx_stop = [];
              nr_lick_idx_mov = [];
     for nr = 1:length(nr_lick_idx)
         if length(find(cell2mat(stoppedblocks) == nr_lick_idx(nr)))>0
             nr_lick_idx_stop = [nr_lick_idx_stop nr_lick_idx(nr)];
         else
             nr_lick_idx_mov = [nr_lick_idx_mov nr_lick_idx(nr)];
         end
     end
        end
        
        save([pr_dir2 'params.mat'],'nr_lick_idx_stop','nr_lick_idx_mov','-append')
    
         if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
        
        for roii = 1:size(df_f,1)
            roiplane1 = roiplane1+1;
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                
        
        % peri nr_lick_idx_stop
          roidop_success_perinonrew_lick_stop=[];    roidop_success_perinonrew_lick_stop=[];
                    
                    for stamps=1:length(nr_lick_idx_stop)
                        currentidx =  find(timedFF>=utimedFF(nr_lick_idx_stop(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_perinonrew_lick_stop(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perinonrew_lick_stop(stamps,:)= forwardvelALL(nr_lick_idx_stop(stamps)-pre_win_framesALL:nr_lick_idx_stop(stamps)+post_win_framesALL);
                            
                        end
                    end
                    
                    if ~isempty(roe_success_perinonrew_lick_stop) % saving Speed across days
                        roi_roe_alldays_planes_perilick_no_reward_stop{d,roiplane1} = roe_success_perinonrew_lick_stop;
                        roi_roe_allsuc_lick_no_reward_stop(d,roiplane1,:)=mean(roe_success_perinonrew_lick_stop,1);
                    else
                        roi_roe_alldays_planes_perilick_no_reward_stop{d,roiplane1} = NaN(1,length(1-pre_win_framesALL:1+post_win_framesALL));
                        roi_roe_allsuc_lick_no_reward_stop(d,roiplane1,:)=NaN(1,1,length(1-pre_win_framesALL:1+post_win_framesALL));
                    end
                    if ~isempty(roidop_success_perinonrew_lick_stop) % saving Dop across days
                        roi_dop_alldays_planes_perilick_no_reward_stop{d,roiplane1} = roidop_success_perinonrew_lick_stop;
                       roi_dop_allsuc_lick_no_reward_stop(d,roiplane1,:)=mean(roidop_success_perinonrew_lick_stop,1);
                    else
                        roi_dop_alldays_planes_perilick_no_reward_stop{d,roiplane1} = NaN(1,length(1-pre_win_frames:1+post_win_frames));
                       roi_dop_allsuc_lick_no_reward_stop(d,roiplane1,:)=NaN(1,length(1-pre_win_frames:1+post_win_frames));
                    end

                    
                    %peri_lick_idx_mov
                       roidop_success_perinonrew_lick_mov=[];    roe_success_perinonrew_lick_mov=[];
                    
                    for stamps=1:length(nr_lick_idx_mov)
                        currentidx =  find(timedFF>=utimedFF(nr_lick_idx_mov(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_perinonrew_lick_mov(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perinonrew_lick_mov(stamps,:)= forwardvelALL(nr_lick_idx_mov(stamps)-pre_win_framesALL:nr_lick_idx_mov(stamps)+post_win_framesALL);
                            
                        end
                    end
                    
                    
                    if ~isempty(roe_success_perinonrew_lick_mov) % saving Speed across days
                        roi_roe_alldays_planes_perilick_no_reward_mov{d,roiplane1} = roe_success_perinonrew_lick_mov;
                        roi_roe_allsuc_lick_no_reward_mov(d,roiplane1,:)=mean(roe_success_perinonrew_lick_mov,1);
                    else
                        roi_roe_alldays_planes_perilick_no_reward_mov{d,roiplane1} = NaN(1,length(1-pre_win_framesALL:1+post_win_framesALL));
                        roi_roe_allsuc_lick_no_reward_mov(d,roiplane1,:)=NaN(1,1,length(1-pre_win_framesALL:1+post_win_framesALL));
                    end
                    if ~isempty(roidop_success_perinonrew_lick_mov) % saving Dop across days
                        roi_dop_alldays_planes_perilick_no_reward_mov{d,roiplane1} = roidop_success_perinonrew_lick_mov;
                       roi_dop_allsuc_lick_no_reward_mov(d,roiplane1,:)=mean(roidop_success_perinonrew_lick_mov,1);
                    else
                        roi_dop_alldays_planes_perilick_no_reward_mov{d,roiplane1} = NaN(1,length(1-pre_win_frames:1+post_win_frames));
                       roi_dop_allsuc_lick_no_reward_mov(d,roiplane1,:)=NaN(1,length(1-pre_win_frames:1+post_win_frames));
                    end
        end
        
        
    end
    
    
    if sum(analysischeck == 11)>0 % do nonrew start lick no lick analysis
        
           
            numplanes=size(planefolders,2);
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
            speed_thresh = 7.5; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 5; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            if oldbatch == 1
            CSUStimelag = 0;
            else
            CSUStimelag = 0.5; %seconds between
            end
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time/numplanes);
            post_win_framesALL=round(post_win/frame_time);
            pre_win_framesALL = round(pre_win/frame_time);
            pre_win_frames=round(pre_win/frame_time/numplanes);
        
        
        
        
        if allplanes == 1
           stoppedblocks = {};
            if stop_success_tmpts(1)>mov_success_tmpts(1)
                stop_success_tmpts = [1 stop_success_tmpts];
            end
            for s = 1:length(stop_success_tmpts)
                if length(mov_success_tmpts)>= s
                stoppedblocks{s} = stop_success_tmpts(s):mov_success_tmpts(s);
                else
                    stoppedblocks{s} = stop_success_tmpts(s):length(forwardvelALL);
                end
            end
            
            if ~exist('nonrew_mov_success_tmpts')
                %%moving rewarded
                    rew_mov_success_tmpts=[];
                    for jj=1:length(rew_stop_success_tmpts)
                        if ~isempty(find(rew_stop_success_tmpts(jj)-mov_success_tmpts<0,1,'first'))
                        rew_mov_success_tmpts(jj) =mov_success_tmpts(find(rew_stop_success_tmpts(jj)-mov_success_tmpts<0,1,'first'));
                        end
                    end
                    
                    %%%moving unrewarded
                    
                    nonrew_mov_success_tmpts = setxor(rew_mov_success_tmpts,mov_success_tmpts);
            end
            
             nonrew_mov_success_tmpts_licks = [];
              nonrew_mov_success_tmpts_no_licks = [];
     for nr = 1:length(nonrew_mov_success_tmpts)
         if length(find(licksALL(stoppedblocks{find(cellfun(@(x) sum(x == nonrew_mov_success_tmpts(nr)),stoppedblocks,'UniformOutput',1))})))>0
            nonrew_mov_success_tmpts_licks = [nonrew_mov_success_tmpts_licks nonrew_mov_success_tmpts(nr)];
         else
            nonrew_mov_success_tmpts_no_licks = [nonrew_mov_success_tmpts_no_licks nonrew_mov_success_tmpts(nr)];
         end
     end
        end
        
        save([pr_dir2 'params.mat'],'nonrew_mov_success_tmpts_licks','nonrew_mov_success_tmpts_no_licks','-append')
    
         if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
        
        for roii = 1:size(df_f,1)
            roiplane2 = roiplane2+1;
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                
        
        % peri unrew mov with licks
          roidop_success_perinonrew_lick_stop=[];    roidop_success_perinonrew_lick_stop=[];
                    
                    for stamps=1:length(nonrew_mov_success_tmpts_licks)
                        currentidx =  find(timedFF>=utimedFF(nonrew_mov_success_tmpts_licks(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_perinonrew_lick_stop(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perinonrew_lick_stop(stamps,:)= forwardvelALL(nonrew_mov_success_tmpts_licks(stamps)-pre_win_framesALL:nonrew_mov_success_tmpts_licks(stamps)+post_win_framesALL);
                            
                        end
                    end
                    
                    if ~isempty(roe_success_perinonrew_lick_stop) % saving Speed across days
                        roi_roe_alldays_planes_success_mov_no_reward_lick{d,roiplane2} = roe_success_perinonrew_lick_stop;
                        roi_roe_allsuc_mov_no_reward_lick(d,roiplane2,:)=mean(roe_success_perinonrew_lick_stop,1);
                    else
                        roi_roe_alldays_planes_success_mov_no_reward_lick{d,roiplane2} = NaN(1,length(1-pre_win_framesALL:1+post_win_framesALL));
                        roi_roe_allsuc_mov_no_reward_lick(d,roiplane2,:)=NaN(1,1,length(1-pre_win_framesALL:1+post_win_framesALL));
                    end
                    if ~isempty(roidop_success_perinonrew_lick_stop) % saving Dop across days
                        roi_dop_alldays_planes_success_mov_no_reward_lick{d,roiplane2} = roidop_success_perinonrew_lick_stop;
                       roi_dop_allsuc_mov_no_reward_lick(d,roiplane2,:)=mean(roidop_success_perinonrew_lick_stop,1);
                    else
                        roi_dop_alldays_planes_success_mov_no_reward_lick{d,roiplane2} = NaN(1,length(1-pre_win_frames:1+post_win_frames));
                       roi_dop_allsuc_mov_no_reward_lick(d,roiplane2,:)=NaN(1,length(1-pre_win_frames:1+post_win_frames));
                    end

                    
                    %peri unrew mov with no licks
                       roidop_success_perinonrew_lick_mov=[];    roe_success_perinonrew_lick_mov=[];
                    
                    for stamps=1:length(nonrew_mov_success_tmpts_no_licks)
                        currentidx =  find(timedFF>=utimedFF(nonrew_mov_success_tmpts_no_licks(stamps)),1);
                        if currentidx+post_win_frames<length(roinorm_base_mean)
                            roidop_success_perinonrew_lick_mov(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perinonrew_lick_mov(stamps,:)= forwardvelALL(nonrew_mov_success_tmpts_no_licks(stamps)-pre_win_framesALL:nonrew_mov_success_tmpts_no_licks(stamps)+post_win_framesALL);
                            
                        end
                    end
                    
                    
                    if ~isempty(roe_success_perinonrew_lick_mov) % saving Speed across days
                        roi_roe_alldays_planes_success_mov_no_reward_no_lick{d,roiplane2} = roe_success_perinonrew_lick_mov;
                        roi_roe_allsuc_mov_no_reward_no_lick(d,roiplane2,:)=mean(roe_success_perinonrew_lick_mov,1);
                    else
                        roi_roe_alldays_planes_perilick_no_reward_mov{d,roiplane2} = NaN(1,length(1-pre_win_framesALL:1+post_win_framesALL));
                        roi_roe_allsuc_mov_no_reward_no_lick(d,roiplane2,:)=NaN(1,1,length(1-pre_win_framesALL:1+post_win_framesALL));
                    end
                    if ~isempty(roidop_success_perinonrew_lick_mov) % saving Dop across days
                        roi_dop_alldays_planes_success_mov_no_reward_no_lick{d,roiplane2} = roidop_success_perinonrew_lick_mov;
                       roi_dop_allsuc_mov_no_reward_no_lick(d,roiplane2,:)=mean(roidop_success_perinonrew_lick_mov,1);
                    else
                        roi_dop_alldays_planes_success_mov_no_reward_no_lick{d,roiplane2} = NaN(1,length(1-pre_win_frames:1+post_win_frames));
                       roi_dop_allsuc_mov_no_reward_no_lick(d,roiplane2,:)=NaN(1,length(1-pre_win_frames:1+post_win_frames));
                    end
        end
        
        
    end
    
     if sum(analysischeck == 12)>0 % Lick bout analysis
         
         numplanes=size(planefolders,2);
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
            speed_thresh = 7.5; %cm/s cut off for stopped
            Stopped_frame = 15;%frame_rate; %number of seconds for it to be considered a stop;
            max_reward_stop = 5*frame_rate; %number of seconds after reward for a stop to be considered a reward related stop * frame rate.
            frame_tol = 5; %number of frames prior to reward to check for stopping points as a tolerance for defining stopped.
            if oldbatch == 1
            CSUStimelag = 0;
            else
            CSUStimelag = 0.5; %seconds between
            end
            frame_time=1/frame_rate;
            num_rew_win_frames=round(num_rew_win_sec/frame_time);%window in frames
            rew_lick_win_frames=round(rew_lick_win/frame_time);%window in frames
            post_win_frames=round(post_win/frame_time/numplanes);
            post_win_framesALL=round(post_win/frame_time);
            pre_win_framesALL = round(pre_win/frame_time);
            pre_win_frames=round(pre_win/frame_time/numplanes);
        
            
            
         if allplanes == 1
             if oldbatch == 1
            solenoid2ALL = rewardsALL;
            licksALL = lickVoltageALL<-0.08;
        else
%         short = solenoid2ALL;
        end
         singlelick = zeros(size(licksALL));
         singlelick(cellfun(@(x) x(1),consecutive_stretch(find(licksALL)),'UniformOutput',1)) = 1;
         lickbout = bwlabel(singlelick);
         for dw = 1:max(lickbout)-1
                if utimedFF(find(lickbout==dw+1,1))-utimedFF(find(lickbout==dw,1,'last'))<=0.5
                    lickbout(find(lickbout==dw,1):find(lickbout==dw+1,1)) = dw+1;
                end
                
         end
            lickbout = lickbout>0;
            nonrewlickbout = lickbout;
            cnonrewlickbout = consecutive_stretch(find(nonrewlickbout));
            CSSs = cellfun(@(x) x(1) ,consecutive_stretch(find(solenoid2ALL)),'UniformOutput',1);
            deleteidx = [];
            for css = 1:length(CSSs)
                deleteidx = [deleteidx find(cellfun(@(x) sum((utimedFF(x)-utimedFF(CSSs(css)))<5&(utimedFF(x)-utimedFF(CSSs(css)))>0)>0,cnonrewlickbout,'UniformOutput',1))];
            end
            cnonrewlickbout(deleteidx) = [];
            cnonrewlickbout(cellfun(@(x) sum(utimedFF(x)-5.1<=0|utimedFF(x)+5.1>utimedFF(end))>0,cnonrewlickbout,'UniformOutput',1)) = [];
            numlicksperbout = cellfun(@(x) sum(singlelick(x)),cnonrewlickbout,'UniformOutput',1);
            number_licks_per_bout{d,1} = numlicksperbout;
            
            templogical = zeros(size(singlelick));
            templogical(cell2mat(cnonrewlickbout)) = 1;
%             figure;
%             plot(utimedFF,singlelick)
%             hold on
%             plot(utimedFF,lickbout)
%             plot(utimedFF,templogical)
%             plot(utimedFF,solenoid2ALL)
         end
             if oldbatch==1
                
                df_f=params.roibasemean2;
         else
                if ~isfield(params,'roibasemean3')
                    df_f=params.roibasemean2;
                else
                df_f=params.roibasemean3;
                end
         end
        
        for roii = 1:size(df_f,1)
            roiplane3 = roiplane3+1;
                roibase_mean = df_f{roii,1};
                roimean_base_mean=mean(df_f{roii,1});
                
                
                roinorm_base_mean=roibase_mean/roimean_base_mean;
                
                
        
      
%                     roiperlickboutstart = [];
                    roidop_success_perilickboutstart=[];    roe_success_perilickboutstart=[];
                    for stamps = 1:length(cnonrewlickbout)
                        currentidx = find(timedFF>=utimedFF(cnonrewlickbout{stamps}(1)),1);
%                          currentidx =  find(timedFF>=utimedFF(nonrew_mov_success_tmpts_licks(stamps)),1);
                        if currentidx+post_win_frames<=length(roinorm_base_mean)
                            roidop_success_perilickboutstart(stamps,:)= roinorm_base_mean(currentidx-pre_win_frames:currentidx+post_win_frames);
                            roe_success_perilickboutstart(stamps,:)= forwardvelALL(cnonrewlickbout{stamps}(1)-pre_win_framesALL:cnonrewlickbout{stamps}(1)+post_win_framesALL);
                            
                        end
                    end
                   if ~isempty(roe_success_perilickboutstart) % saving Speed across days
                        roi_roe_alldays_planes_success_perilickboutstart{d,roiplane3} = roe_success_perilickboutstart;
                        roi_roe_allsuc_perilickboutstart(d,roiplane3,:)=mean(roe_success_perilickboutstart,1);
                    else
                        roi_roe_alldays_planes_success_perilickboutstart{d,roiplane3} = NaN(1,length(1-pre_win_framesALL:1+post_win_framesALL));
                        roi_roe_allsuc_perilickboutstart(d,roiplane3,:)=NaN(1,1,length(1-pre_win_framesALL:1+post_win_framesALL));
                    end
                    if ~isempty(roidop_success_perilickboutstart) % saving Dop across days
                        roi_dop_alldays_planes_success_perilickboutstart{d,roiplane3} = roidop_success_perilickboutstart;
                       roi_dop_allsuc_perilickboutstart(d,roiplane3,:)=mean(roidop_success_perilickboutstart,1);
                    else
                        roi_dop_alldays_planes_success_perilickboutstart{d,roiplane3} = NaN(1,length(1-pre_win_frames:1+post_win_frames));
                       roi_dop_allsuc_perilickboutstart(d,roiplane3,:)=NaN(1,length(1-pre_win_frames:1+post_win_frames));
                    end
        end
            
            
       
         end
 
     end
    
 

end

if sum(analysischeck == 10)>0
    varchecks = {'roi_roe_alldays_planes_perilick_no_reward_stop','roi_roe_allsuc_lick_no_reward_stop','roi_dop_alldays_planes_perilick_no_reward_stop',...
       'roi_dop_allsuc_lick_no_reward_stop','roi_roe_alldays_planes_perilick_no_reward_mov',...
       'roi_roe_allsuc_lick_no_reward_mov','roi_dop_alldays_planes_perilick_no_reward_mov',...
       'roi_dop_allsuc_lick_no_reward_mov'};
   for cvar = 1:length(varchecks)
 currvar = eval(varchecks{cvar});
 clearvars newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        else
            for d = 1:size(currvar,1)
                newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
         eval([varchecks{cvar} ' = newvar;']);
   end
end

if sum(analysischeck == 11)>0
    varchecks = {'roi_roe_alldays_planes_success_mov_no_reward_no_lick','roi_roe_allsuc_mov_no_reward_no_lick',...
        'roi_dop_alldays_planes_success_mov_no_reward_no_lick',...
       'roi_dop_allsuc_mov_no_reward_no_lick','roi_roe_alldays_planes_success_mov_no_reward_lick',...
       'roi_roe_allsuc_mov_no_reward_lick','roi_dop_alldays_planes_success_mov_no_reward_lick',...
       'roi_dop_allsuc_mov_no_reward_lick'};
   for cvar = 1:length(varchecks)
 currvar = eval(varchecks{cvar});
 clearvars newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        else
            for d = 1:size(currvar,1)
                newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
         eval([varchecks{cvar} ' = newvar;']);
   end
end

if sum(analysischeck == 12)>0
    varchecks = {'roi_roe_alldays_planes_success_perilickboutstart','roi_roe_allsuc_perilickboutstart',...
        'roi_dop_alldays_planes_success_perilickboutstart',...
       'roi_dop_allsuc_perilickboutstart'};
   for cvar = 1:length(varchecks)
 currvar = eval(varchecks{cvar});
 clearvars newvar
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        else
            for d = 1:size(currvar,1)
                newvar(d,:,:) = currvar(d,daykey(d,:),:);
            end
        end
         eval([varchecks{cvar} ' = newvar;']);
   end
end


%%%%%%%%%
%%%%%%%%% Plotting Portion
%%%%%%%%%

if sum(analysischeck == 1)>0 % do CS to first lick latency analysis
save(wrk_dir{1},'CS_alldays_lick_gap','-append')
figure; errorbar(cellfun(@nanmean,CS_alldays_lick_gap),cellfun(@nanstd,CS_alldays_lick_gap)./cellfun(@(x) sqrt(sum(~isnan(x))),CS_alldays_lick_gap,'UniformOutput',1))
xlabel('days')
ylabel('Latency (seconds)')
title('Lick from CS')
CS_alldays_lick_latency = test2;
save(wrk_dir{1},'CS_alldays_lick_latency','-append')
figure; errorbar(cellfun(@nanmean,test2),cellfun(@nanstd,test2)./cellfun(@(x) sqrt(sum(~isnan(x))),test2,'UniformOutput',1))
xlabel('days')
ylabel('Latency (seconds)')
title('Lick from CS')
end


if sum(analysischeck == 2)>0 % do CS to stop latency analysis
    save(wrk_dir{1},'CS_alldays_stop_gap','-append')
figure; errorbar(cellfun(@nanmean,CS_alldays_stop_gap),cellfun(@nanstd,CS_alldays_stop_gap)./cellfun(@(x) sqrt(sum(~isnan(x))),CS_alldays_stop_gap,'UniformOutput',1),'k-')
xlabel('days')
ylabel('Latency (seconds)')
title('Stop from CS')
end

if sum(analysischeck == 3)>0 % do Chopped speed correlation
locvars = {'stoppedblocks','maintainspeed','deccel2stop','accelfromstop','deccel2movfast','accelfrommovfast','deccel2movslow','accelfrommovslow','bufferedmoving'};

    for va = 1:length(locvars)
        clear newvar
        currvar = eval([locvars{va} '_corr_alldays_planes']);
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        end
         eval([locvars{va} '_corr_alldays_planes = newvar;']);
         
         if strcmp(locvars{va},'bufferedmoving')
             
             clear newvar
        currvar = eval([locvars{va} '_corr_alldays_planes_chunked']);
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        end
         eval([locvars{va} '_corr_alldays_planes_chunked = newvar;']);
         end
    end

   locvars = {'stoppedblocks','maintainspeed','deccel2stop','accelfromstop','deccel2movfast','accelfrommovfast','deccel2movslow','accelfrommovslow'};
  for va = 1:length(locvars)
      currvar = cell2mat(eval([locvars{va} '_corr_alldays_planes']));
      subplot(4,2,va)
      plot(currvar)
      title(locvars{va})
      xlabel('Days')
      ylabel('Corr')
  end
  
  figure;
    sb(1) = subplot(1,2,1);
    currvar = cell2mat(bufferedmoving_corr_alldays_planes);
    plot(currvar)
    title('Moving Corr Concat')
    xlabel('Days')
    sb(2) = subplot(1,2,2);
    currvar = bufferedmoving_corr_alldays_planes_chunked;
    errorbar(cellfun(@nanmean,currvar),cellfun(@(x) nanstd(x)/sqrt(length(x)),currvar,'UniformOutput',1))
    title('Moving Corr Per Bout')
    xlabel('Days')
    linkaxes(sb,'xy')
    
    % do the same but for Acceleration
    
    locvars = {'stoppedblocks','maintainspeed','deccel2stop','accelfromstop','deccel2movfast','accelfrommovfast','deccel2movslow','accelfrommovslow','bufferedmoving'};

    for va = 1:length(locvars)
        clear newvar
        currvar = eval([locvars{va} '_accel_corr_alldays_planes']);
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        end
         eval([locvars{va} '_accel_corr_alldays_planes = newvar;']);
         
         if strcmp(locvars{va},'bufferedmoving')
             
             clear newvar
        currvar = eval([locvars{va} '_accel_corr_alldays_planes_chunked']);
        if length(size(currvar))==2
            for d = 1:size(currvar,1)
            newvar(d,:) = currvar(d,daykey(d,:));
            end
        end
         eval([locvars{va} '_accel_corr_alldays_planes_chunked = newvar;']);
         end
    end

   locvars = {'stoppedblocks','maintainspeed','deccel2stop','accelfromstop','deccel2movfast','accelfrommovfast','deccel2movslow','accelfrommovslow'};
  for va = 1:length(locvars)
      currvar = cell2mat(eval([locvars{va} '_accel_corr_alldays_planes']));
      subplot(4,2,va)
      plot(currvar)
      title(['accel ' locvars{va}])
      xlabel('Days')
      ylabel('Corr')
  end
  
  figure;
    sb(1) = subplot(1,2,1);
    currvar = cell2mat(bufferedmoving_accel_corr_alldays_planes);
    plot(currvar)
    title('Accel Moving Corr Concat')
    xlabel('Days')
    sb(2) = subplot(1,2,2);
    currvar = bufferedmoving_accel_corr_alldays_planes_chunked;
    errorbar(cellfun(@nanmean,currvar),cellfun(@(x) nanstd(x)/sqrt(length(x)),currvar,'UniformOutput',1))
    title('Accel Moving Corr Per Bout')
    xlabel('Days')
    linkaxes(sb,'xy')
    
end


if sum(analysischeck == 4)>0 % do CS to transient analysis
  dplot = 2;
    for d = 1:size(alldFFs,1)
        figure(1000+ceil(d/dplot));
        for ri =1:length(daykey(d,:))
             subplot(length(daykey(d,:)),dplot,dplot*(length(daykey(d,:))-ri+1)-dplot+(rem(d+dplot-1,dplot)+1))
             plot(allutime{d},rescale(allCS{d},min(alldFFs{d,daykey(d,ri)}),max(alldFFs{d,daykey(d,ri)})),'Color',[212 175 55]/255)
             hold on
             plot(alltime{d,daykey(d,ri)},alldFFs{d,daykey(d,ri)},'b-')
             postran = consecutive_stretch(find(allpostrans{d,daykey(d,ri)}));
             negtran =  consecutive_stretch(find(allnegtrans{d,daykey(d,ri)}));
             for p = 1:length(postran)
                 plot(alltime{d,daykey(d,ri)}(postran{p}),alldFFs{d,daykey(d,ri)}(postran{p}),'r-')
             end
             for n = 1:length(negtran)
                 plot(alltime{d,daykey(d,ri)}(negtran{n}),alldFFs{d,daykey(d,ri)}(negtran{n}),'g-')
             end
             xlim([60 600])
             if ri == length(daykey(d,:))
                 title(['ROI ' num2str(ri) ' Day ' num2str(d)])
             else
             title(['ROI ' num2str(ri)])
             end
        end
    end
    
end


if sum(analysischeck == 7)>0 % do CS to transient analysis
  dplot = 2;
    for d = 1:size(alldFFs,1)
        figure(1000+ceil(d/dplot));
        for ri =1:length(daykey(d,:))
             subplot(length(daykey(d,:)),dplot,dplot*(length(daykey(d,:))-ri+1)-dplot+(rem(d+dplot-1,dplot)+1))
             hold on
             for s = 1:length(allstoppedblocks{d})
                 rectangle('Position',[allutime{d}(allstoppedblocks{d}{s}(1)) min(alldFFs{d,daykey(d,ri)})-range(alldFFs{d,daykey(d,ri)})...
                     allutime{d}(length(allstoppedblocks{d}{s})) 2*range(alldFFs{d,daykey(d,ri)})],...
                     'FaceColor',[0 1 1 0.5],'EdgeColor',[0 0 0 0.5]) 
             end
             plot(allutime{d},rescale(allCS{d},min(alldFFs{d,daykey(d,ri)}),max(alldFFs{d,daykey(d,ri)})),'Color',[212 175 55]/255)
             hold on
             plot(alltime{d,daykey(d,ri)},alldFFs{d,daykey(d,ri)},'b-')
             
             xlim([0 120])
             ylim([min(alldFFs{d,daykey(d,ri)})-range(alldFFs{d,daykey(d,ri)}) max(alldFFs{d,daykey(d,ri)})])
             yticks(round(min(alldFFs{d,daykey(d,ri)})/50)*50:50:max(alldFFs{d,daykey(d,ri)}))
             yyaxis right
             plot(allutime{d},allspeed{d},'k-')
             hold on
             plot(allutime{d}(find(alllicks{d})),allspeed{d}(find(alllicks{d})),'r.')
             ylim([min(allspeed{d}) max(allspeed{d})+range(allspeed{d})])
             yticks(round(min(allspeed{d})/50)*50:50:max(allspeed{d}))
             if ri == length(daykey(d,:))
                 title(['ROI ' num2str(ri) ' Day ' num2str(d)])
             else
             title(['ROI ' num2str(ri)])
             end
        end
    end
    
end
%%

if sum(analysischeck == 8)>0 % do 2 min peristop plots
    figure;
    subplot(2,2,1)
    yax1 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(1:4,end),'UniformOutput',0)));
    seyax1 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(1:4,end),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    yax2 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(1:4,end),'UniformOutput',0)));
    seyax2 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(1:4,end),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1)); 
    shadedErrorBar(xax,yax1,seyax1,'r',1)
    hold on
    shadedErrorBar(xax,yax2,seyax2,'g',1)
    ylim([0.975 1.05])
    ylims = ylim;
    text(-4,ylims(2)*0.999,'Licking Stops','Color','r')
    text(-4,ylims(2)*0.98,'Non-Licking Stops','Color','g')
    title('SO Early Days')
    
    
     subplot(2,2,2)
    yax1 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(end-3:end,end),'UniformOutput',0)));
    seyax1 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(end-3:end,end),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    yax2 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(end-3:end,end),'UniformOutput',0)));
    seyax2 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(end-3:end,end),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1)); 
    shadedErrorBar(xax,yax1,seyax1,'r',1)
    hold on
    shadedErrorBar(xax,yax2,seyax2,'g',1)
    ylim([0.975 1.05])
    ylims = ylim;
    text(-4,ylims(2)*0.999,'Licking Stops','Color','r')
    text(-4,ylims(2)*0.98,'Non-Licking Stops','Color','g')
    title('SO Late Days')
    
     subplot(2,2,3)
    yax1 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(1:4,1:end-1),'UniformOutput',0)));
    seyax1 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(1:4,1:end-1),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    yax2 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(1:4,1:end-1),'UniformOutput',0)));
    seyax2 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(1:4,1:end-1),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1)); 
    shadedErrorBar(xax,yax1,seyax1,'r',1)
    hold on
    shadedErrorBar(xax,yax2,seyax2,'g',1)
    ylim([0.975 1.05])
    ylims = ylim;
    text(-4,ylims(2)*0.999,'Licking Stops','Color','r')
    text(-4,ylims(2)*0.98,'Non-Licking Stops','Color','g')
    title('Not SO Early Days')
    
     subplot(2,2,4)
    yax1 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(end-3:end,1:end-1),'UniformOutput',0)));
    seyax1 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_lick_stop_no_reward2min(end-3:end,1:end-1),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    yax2 = nanmean(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(end-3:end,1:end-1),'UniformOutput',0)));
    seyax2 = nanstd(cell2mat(cellfun(@(x) nanmean(x,1), roi_dop_allsuc_nolick_stop_no_reward2min(end-3:end,1:end-1),'UniformOutput',0)))/sqrt(4);
    xax = linspace(-5,5,length(yax1)); 
    shadedErrorBar(xax,yax1,seyax1,'r',1)
    hold on
    shadedErrorBar(xax,yax2,seyax2,'g',1)
    ylim([0.975 1.05])
    ylims = ylim;
    
    text(-4,ylims(2)*0.999,'Licking Stops','Color','r')
    text(-4,ylims(2)*0.98,'Non-Licking Stops','Color','g')
    title('Not SO Late Days')
    
    
    %plane Version
    planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
     ylims = ([0.98 1.05])
 speedtransf = [0 25];
 doptransf = [0.981 0.99]; 
 speedvars ={'roi_roe_allsuc_lick_stop_no_reward2min','roi_roe_allsuc_nolick_stop_no_reward2min'};

 dopvars = {'roi_dop_allsuc_lick_stop_no_reward2min','roi_dop_allsuc_nolick_stop_no_reward2min'};
    titles = {'1st2min NonRew Stop Licks','1st2min NonRew Stop no Licks'};
    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
figure;
for vars = 1:length(dopvars)
 speedvariable = eval(speedvars{vars});
 dopvariable = eval(dopvars{vars});
 dopvariable = cell2mat(cellfun(@(x) reshape(nanmean(x,1),1,1,[]),dopvariable,'UniformOutput',0));
 speedvariable = cell2mat(cellfun(@(x) reshape(nanmean(x,1),1,1,[]),speedvariable,'UniformOutput',0));
    subplot(2,2,vars+2)
    if ~isnan(sum( nanmean(dopvariable(max([1 size(dopvariable,1)-3]):end,1,:))))
for r = 1:size(dopvariable,2)
    %late days
    yax1 = nanmean(dopvariable(max([1 size(dopvariable,1)-3]):end,r,:));
    seyax1 = nanstd((dopvariable(max([1 size(dopvariable,1)-3]):end,r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
    h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
%     doptransf = [0.976 0.989];
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
%     yyaxis right

    end
    ylim( ylims)
    ylims = ylim;
end
    end
title(titles{vars})
ylabel('Late Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)

 subplot(2,2,vars)
for r = 1:size(dopvariable,2)
    %early days
    yax1 = nanmean(dopvariable(1:min([4 size(dopvariable,1)]),r,:));
    seyax1 = nanstd((dopvariable(1:min([4 size(dopvariable,1)]),r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(1:min([4 size(dopvariable,1)]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
    
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars})
ylabel('Early Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)


end

end

if sum(analysischeck == 9)>0
    
    

figure;
subplot(3,1,1)
errorbar(cellfun(@nanmean,test2),cellfun(@nanstd,test2)./cellfun(@(x) sqrt(sum(~isnan(x))),test2,'UniformOutput',1))
xlabel('days')
ylabel('Latency (seconds)')
title('Lick from CS')
subplot(3,1,2)
errorbar(cellfun(@nanmean,CS_alldays_stop_gap),cellfun(@nanstd,CS_alldays_stop_gap)./cellfun(@(x) sqrt(sum(~isnan(x))),CS_alldays_stop_gap,'UniformOutput',1),'k-')
xlabel('days')
ylabel('Latency (seconds)')
title('Stop from CS')


planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
        if exist('mouseid')
        if mouseid == 181 && isempty(roi_dop_alldays_planes_periCS{19,4})
        roi_dop_alldays_planes_periCS = [repmat({NaN(79,3)},23,1) roi_dop_alldays_planes_periCS];
        roi_dop_alldays_planes_periCS(:,5) = [];
        end
        end

        means = cellfun(@(x) nanmean(reshape(x(40:42,:),[],1)),roi_dop_alldays_planes_periCS,'UniformOutput',1);
        sems = cellfun(@(x) nanstd(reshape(x(40:42,:),[],1))/sqrt(numel(x(40:42,:))),roi_dop_alldays_planes_periCS,'UniformOutput',1);
        subplot(3,1,3)
        for p = 1:size(means,2)
            errorbar(means(:,p),sems(:,p),'Color',planecolors{p},'Capsize',0)
            hold on
        end
end


if sum(analysischeck==10)>0
 ylims = ([0.98 1.05])
 speedtransf = [0 25];
 doptransf = [0.981 0.99]; 
 speedvars ={'roi_roe_allsuc_lick_no_reward_stop','roi_roe_allsuc_lick_no_reward_mov'};

 dopvars = {'roi_dop_allsuc_lick_no_reward_stop','roi_dop_allsuc_lick_no_reward_mov'};
    titles = {'Non Rew Licks with Stops','Non Rew Licks without Stops'};
    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
figure;
for vars = 1:length(dopvars)
 speedvariable = eval(speedvars{vars});
 dopvariable = eval(dopvars{vars});
    subplot(2,2,vars+2)
for r = 1:size(dopvariable,2)
    %late days
    yax1 = nanmean(dopvariable(max([1 size(dopvariable,1)-3]):end,r,:));
    seyax1 = nanstd((dopvariable(max([1 size(dopvariable,1)-3]):end,r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
    h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
%     doptransf = [0.976 0.989];
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
%     yyaxis right

    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars})
ylabel('Late Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)

 subplot(2,2,vars)
for r = 1:size(dopvariable,2)
    %early days
    yax1 = nanmean(dopvariable(1:min([4 size(dopvariable,1)]),r,:));
    seyax1 = nanstd((dopvariable(1:min([4 size(dopvariable,1)]),r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(1:min([4 size(dopvariable,1)]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
    
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars})
ylabel('Early Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)


end
end




    varchecks = {'roi_roe_alldays_planes_success_mov_no_reward_no_lick','roi_roe_allsuc_mov_no_reward_no_lick',...
        'roi_dop_alldays_planes_success_mov_no_reward_no_lick',...
       'roi_dop_allsuc_mov_no_reward_no_lick','roi_roe_alldays_planes_success_mov_no_reward_lick',...
       'roi_roe_allsuc_mov_no_reward_lick','roi_dop_alldays_planes_success_mov_no_reward_lick',...
       'roi_dop_allsuc_mov_no_reward_lick'};
   
   
   if sum(analysischeck==11)>0
 ylims = ([0.98 1.05])
 speedtransf = [0 25];
 doptransf = [0.981 0.99]; 
 speedvars ={'roe_success_perimov_no_reward','roi_roe_allsuc_mov_no_reward_no_lick','roi_roe_allsuc_mov_no_reward_lick'};

 dopvars = {'roi_dop_allsuc_mov_no_reward','roi_dop_allsuc_mov_no_reward_no_lick','roi_dop_allsuc_mov_no_reward_lick'};
    titles = {'Non Rew Move','Non Rew Move Licks','Non Rew Move no Licks'};
    
planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
figure;
for vars = 1:length(dopvars)
 speedvariable = eval(speedvars{vars});
 dopvariable = eval(dopvars{vars});
    subplot(2,3,vars+3)
for r = 1:size(dopvariable,2)
    %late days
    yax1 = nanmean(dopvariable(max([1 size(dopvariable,1)-3]):end,r,:));
    seyax1 = nanstd((dopvariable(max([1 size(dopvariable,1)-3]):end,r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
    h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
        if vars == 1
            speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,:)));
        else
    speedyax = squeeze(nanmean(speedvariable(max([1 size(dopvariable,1)-3]):end,r,:)));
        end
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
%     doptransf = [0.976 0.989];
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
%     yyaxis right

    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars})
ylabel('Late Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)

 subplot(2,3,vars)
for r = 1:size(dopvariable,2)
    %early days
    yax1 = nanmean(dopvariable(1:min([4 size(dopvariable,1)]),r,:));
    seyax1 = nanstd((dopvariable(1:min([4 size(dopvariable,1)]),r,:)))/sqrt(4);
    xax = linspace(-5,5,length(yax1));
    
    
    h = shadedErrorBar(xax,yax1,seyax1,'r',1);
     h.mainLine.Color = planecolors{r};
    h.patch.FaceColor = planecolors{r};
    h.edge(1).Color = planecolors{r};
    h.edge(2).Color = planecolors{r};
    
    if r == 1
    speedyax = squeeze(nanmean(speedvariable(1:min([4 size(dopvariable,1)]):end,r,:)));
    speedxax = linspace(-5,5,length(speedyax));
    hold on
%     speedtransf = [0 100];
    
    transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
    plot(speedxax,transspeedyax,'k-')
    end
    ylim( ylims)
    ylims = ylim;
end
title(titles{vars})
ylabel('Early Days')
yyaxis right
ylim(ylims)
yticks(doptransf)
yticklabels(speedtransf)


end
   end
%%
    if sum(analysischeck==12)>0
 ylims = ([0.98-0.07 1.06]);
 speedtransf = [0 50];
 doptransf = [0.981-0.07 0.95];
 planecolors={[0 0 1],[0 1 0],[204 164 61]/256,[231 84 128]/256};
 mmaxlicknumcorr = [];
 mintlicknumcorr = [];
 mmaxstoplencorr = [];
 mintstoplencorr = [];
 mintprestoplencorr = [];
 
 for d = 1:size(number_licks_per_bout,1)
     find_figure(['PeriLickBout Day ' num2str(d)])
     uniquenumbers = unique(number_licks_per_bout{d});
      temp1 = number_licks_per_bout{d};
     [us,temphist] = histcounts(temp1,min(uniquenumbers):max(uniquenumbers)+1);
     [deleteidx] = ismember(uniquenumbers,temphist(find(us<5))); 
     uniquenumbers(deleteidx) = [];
     for n = 1:length(uniquenumbers)
         
         currtraces = find(number_licks_per_bout{d}==uniquenumbers(n));
         if ~isempty(currtraces)
         speedyax = roi_roe_alldays_planes_success_perilickboutstart{d,1}(currtraces,:)';
         speedxax = linspace(-5,5,size(speedyax,1));
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
         dopyax = roi_dop_alldays_planes_success_perilickboutstart{d,pll}(currtraces,:)';
         
         dopxax = linspace(-5,5,size(dopyax,1));
         roundxax = find(dopxax>=-2.5&dopxax<=-1);
         dopyax = dopyax./mean(dopyax(roundxax,:));
         subplot(4,length(uniquenumbers),n+(length(uniquenumbers))*(4-pll+1)-length(uniquenumbers))
     transspeedyax = diff(doptransf)/diff(speedtransf)*(speedyax)+doptransf(1);
        
        plot(dopxax',dopyax,'Color',[planecolors{pll} 0.2])
        hold on
        plot(dopxax',mean(dopyax,2),'Color',planecolors{pll}/2,'lineWidth',1.5)
        plot(speedxax',transspeedyax,'k-','Color',[0.3 0.3 0.3 0.02])
         plot(speedxax',mean(transspeedyax,2),'k-','Color',[0 0 0],'lineWidth',1.5)
        ylim(ylims)
        if pll == 4
            title([num2str(uniquenumbers(n)) ' Licks'])
            legend({['n = ' num2str(size(dopyax,2))]})
        end
        if pll == 1
            xlabel('Seconds (s)')
        end
        
     end
         end
     end
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
     find_figure(['Max Dop vs Number of Licks Plane ' num2str(pll)])
     subplot(ceil(sqrt(size(number_licks_per_bout,1))),ceil(sqrt(size(number_licks_per_bout,1))),d)
     temp1 = number_licks_per_bout{d};
     u = unique(temp1);
     [us,temphist] = histcounts(temp1,min(u):max(u)+1);
     [deleteidx] = ismember(temp1,temphist(find(us<5))); 
     temp2 = max(roi_dop_alldays_planes_success_perilickboutstart{d,pll}');
     temp2(deleteidx) = [];
     temp1(deleteidx) = [];
    scatter(temp1,temp2,15,'filled')
     r = corr(temp1',temp2','type','Spearman');
     mmaxlicknumcorr(d,pll) = r;
     title(['Day ' num2str(d) ' r: ' num2str(r)])
     end
     
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
     find_figure(['Int Dop vs Number of Licks Plane ' num2str(pll)])
     subplot(ceil(sqrt(size(number_licks_per_bout,1))),ceil(sqrt(size(number_licks_per_bout,1))),d)
     temp1 = number_licks_per_bout{d};
     u = unique(temp1);
     [us,temphist] = histcounts(temp1,min(u):max(u)+1);
     [deleteidx] = ismember(temp1,temphist(find(us<5))); 
     temp2 = trapz(roi_dop_alldays_planes_success_perilickboutstart{d,pll}');
     temp2(deleteidx) = [];
     temp1(deleteidx) = [];
    scatter(temp1,temp2,15,'filled')
     r = corr(temp1',temp2','type','Spearman');
     mintlicknumcorr(d,pll) = r;
     title(['Day ' num2str(d) ' r: ' num2str(r)])
     end
     
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
     find_figure(['Max Dop vs length of Stop Plane ' num2str(pll)])
     subplot(ceil(sqrt(size(number_licks_per_bout,1))),ceil(sqrt(size(number_licks_per_bout,1))),d)
     temp1 = number_licks_per_bout{d};
     u = unique(temp1);
     [us,temphist] = histcounts(temp1,min(u):max(u)+1);
     [deleteidx] = ismember(temp1,temphist(find(us<5))); 
     temp2 = max(roi_dop_alldays_planes_success_perilickboutstart{d,pll}');
     temp3 = smoothdata(roi_roe_alldays_planes_success_perilickboutstart{d,1}','gaussian',8)';
     temp4 = zeros(size(temp3,1),1);
     speedxax = linspace(-5,5,size(temp3,2));
     for n = 1:size(temp3,1)
         temp4(n) = sum(cellfun(@(x) length(x)*double(sum(find(x==floor(size(temp3,2)/2)))>0),consecutive_stretch(find(temp3(n,:)<20)),'UniformOutput',1))*(speedxax(2)-speedxax(1));
     end
     temp2(deleteidx) = [];
     temp1(deleteidx) = [];
     temp4(deleteidx) = [];
    scatter(temp4,temp2,15,'filled')
     r = corr(temp4,temp2');
     mmaxstoplencorr(d,pll) = r;
     title(['Day ' num2str(d) ' r: ' num2str(r)])
     end
     
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
     find_figure(['Int Dop vs length of Stop Plane ' num2str(pll)])
     subplot(ceil(sqrt(size(number_licks_per_bout,1))),ceil(sqrt(size(number_licks_per_bout,1))),d)
     temp1 = number_licks_per_bout{d};
     u = unique(temp1);
     [us,temphist] = histcounts(temp1,min(u):max(u)+1);
     [deleteidx] = ismember(temp1,temphist(find(us<5))); 
     temp2 = trapz(roi_dop_alldays_planes_success_perilickboutstart{d,pll}');
     temp3 = smoothdata(roi_roe_alldays_planes_success_perilickboutstart{d,1}','gaussian',8)';
     temp4 = zeros(size(temp3,1),1);
     speedxax = linspace(-5,5,size(temp3,2));
     for n = 1:size(temp3,1)
         temp4(n) = sum(cellfun(@(x) length(x)*double(sum(find(x==floor(size(temp3,2)/2)))>0),consecutive_stretch(find(temp3(n,:)<20)),'UniformOutput',1))*(speedxax(2)-speedxax(1));
     end
     temp2(deleteidx) = [];
     temp1(deleteidx) = [];
     temp4(deleteidx) = [];
    scatter(temp4,temp2,15,'filled')
     r = corr(temp4,temp2');
     mintstoplencorr(d,pll) = r;
     title(['Day ' num2str(d) ' r: ' num2str(r)])
     end
     
     
     for pll = 1:size(roi_dop_alldays_planes_success_perilickboutstart,2)
     find_figure(['Int Prev Dop vs length of Stop Plane ' num2str(pll)])
     subplot(ceil(sqrt(size(number_licks_per_bout,1))),ceil(sqrt(size(number_licks_per_bout,1))),d)
     temp1 = number_licks_per_bout{d};
     u = unique(temp1);
     [us,temphist] = histcounts(temp1,min(u):max(u)+1);
     [deleteidx] = ismember(temp1,temphist(find(us<5)));
     roundxax = find(dopxax>=-2.5&dopxax<=-1);
    temp5 = roi_dop_alldays_planes_success_perilickboutstart{d,pll};
    temp5 = temp5./mean(temp5(:,roundxax),2);
     temp3 = smoothdata(roi_roe_alldays_planes_success_perilickboutstart{d,1}','gaussian',8)';
     temp4 = zeros(size(temp3,1),1);
     starttemp = NaN(size(temp3,1),1);
     speedxax = linspace(-5,5,size(temp3,2));
     temp2 = NaN(size(temp3,1),1);
%      xtemps = cell(size(temp3,1),1);
     for n = 1:size(temp3,1)
         temp4(n) = sum(cellfun(@(x) length(x)*double(sum(find(x==floor(size(temp3,2)/2)))>0),consecutive_stretch(find(temp3(n,:)<20)),'UniformOutput',1))*(speedxax(2)-speedxax(1));
         trash = cellfun(@(x) x(1)*double(sum(find(x==floor(size(temp3,2)/2)))>0),consecutive_stretch(find(temp3(n,:)<20)),'UniformOutput',1);
        if sum(trash)>0
         starttemp(n) = trash(find(trash));
        end
         if ~isempty(starttemp(n)) && ~isnan(starttemp(n))&& (speedxax(starttemp(n))+5)>=1
             xtemp = find(dopxax<speedxax(starttemp(n)),1,'last')-6:find(dopxax<speedxax(starttemp(n)),1,'last');
%              xtemps{n} = xtemp;
temp2(n) = trapz(temp5(n,xtemp)');
         end
     end
     
     
      
     temp2(deleteidx) = [];
     temp1(deleteidx) = [];
     temp4(deleteidx) = [];
    scatter(temp4,temp2,15,'filled')
     r = corrcoef(temp4',temp2','rows','complete');
     mintprestoplencorr(d,pll) = r(1,2);
     title(['Day ' num2str(d) ' r: ' num2str(r(1,2))])
     end
     
 end
 figure;
 corrvars = {'mmaxlicknumcorr','mintlicknumcorr','mmaxstoplencorr','mintstoplencorr','mintprestoplencorr'};
 corrtitles = {'Max Dop with Lick Number','Integral Dop with Lick Number','Max Dop with Stop Length','Integral Dop with Stop Length','Integral Dop pre stop with stop len'};
 for c = 1:length(corrvars)
     currvar = eval(corrvars{c});
     subplot(1,length(corrvars),c)
     for pll = 1:size(currvar,2)
         plot(currvar(:,pll),'Color', planecolors{pll})
         hold on
     end
     ylabel('Correlation')
     xlabel('Day')
     title(corrtitles{c})
 end
 
 
    end