function [allCOM, allstdCOM, COM] = COMgeneralviewF(VR, allLicks, scalefactor)
%%
%[filename,filepath] = uigetfile('*.mat');
% file = [filepath filename];
% Date='25_Sep_2020_time(17_41_22)';
% mice_names=[{'E142'}];
% trialWindow = 3 ; % what is the number of trial size of your sliding window
% cmBeforeRewZone = 15;
% 1, if you want to use all the licks, 2 if you want to use only licks before reward (reward lick included)

% m=1; % you can create a for loop
% mouseName=mice_names{m};
mouseName_Date=VR.name_date_vr;

%     dir='J:\'; %my USB
%     
%     varname='VR_data'; %folder in my USB
%     inputdir=[dir varname '\'];
%     fn=[mouseName_Date '.mat'];
%     VR=load([inputdir fn]);
%     VR=VR.VR;
% load(file)
    
    % change in Rew Loc 
    
    changeRewLoc=find(VR.changeRewLoc);
    changeRewLoc = [changeRewLoc length(VR.changeRewLoc)];
  
    RewLoc=VR.changeRewLoc(changeRewLoc(1:end-1));
    
    trash = zeros(size(VR.lick));
    trash(VR.lick==1) = VR.ypos(VR.lick==1);
    del = find(trash<10 & trash>0);
    VR.lick(del) = 0;
    
    COM = cell(length(changeRewLoc)-1,1); %only succesfull trials
    stdCOM = cell(length(changeRewLoc)-1,1);
    allCOM = cell(length(changeRewLoc)-1,1); % all the trials
    allstdCOM = cell(length(changeRewLoc)-1,1);
for kk = 1:length(changeRewLoc)-1
    if (VR.changeRewLoc(changeRewLoc(kk)))<=93*scalefactor
    COM{kk,2} = 1; % rew loc 1
    allCOM{kk,2} = 1;
    elseif (VR.changeRewLoc(changeRewLoc(kk)))<=127*scalefactor && (VR.changeRewLoc(changeRewLoc(kk)))>93*scalefactor
    COM{kk,2} = 2;% rew loc 2
    allCOM{kk,2} = 2;
    elseif (VR.changeRewLoc(changeRewLoc(kk)))<=161*scalefactor && (VR.changeRewLoc(changeRewLoc(kk)))>127*scalefactor
    COM{kk,2} = 3; % rew loc 3
    allCOM{kk,2} = 3;
    end

    
    % trials
    difftrials = diff(VR.trialNum(changeRewLoc(kk):changeRewLoc(kk+1)-1));
    difftrials = [0 difftrials];
    startnotprobe = find(VR.trialNum(changeRewLoc(kk):end)>2,1,'first')+changeRewLoc(kk)-1;
    trials=find(difftrials>=1 & VR.trialNum(changeRewLoc(kk):changeRewLoc(kk+1)-1)>2)+changeRewLoc(kk)-1;
%     trials = [startnotprobe trials];
    for jj = 1:length(trials)-1
        if find(VR.reward(trials(jj)+1:trials(jj+1)))>0
           licking = find(VR.lick(trials(jj)+1:find(VR.reward(trials(jj)+1:trials(jj+1)))+trials(jj)))+trials(jj)-1;
           COM{kk} = [COM{kk} mean(VR.ypos(licking))-RewLoc(kk)];
           stdCOM{kk} = [stdCOM{kk} std(VR.ypos(licking))-RewLoc(kk)];
           failure{kk}(jj) = 0;
        else
            licking = find(VR.lick(trials(jj)+1:VR.lick(trials(jj+1))))+trials(jj)-1;
            failure{kk}(jj) = 1;
        end      
        trialyposlicks{kk}{jj} = VR.ypos(licking);
        allCOM{kk}(jj) =mean(VR.ypos(licking))-RewLoc(kk);
        allstdCOM{kk}(jj) = std(VR.ypos(licking));
    end
    
end
dummy = [0; diff(cell2mat(COM(:,2)))];
for i=1:length(dummy)
    if dummy(i)>0
        dummy(i)=2; %far to close
    elseif dummy(i)<0
        dummy(i)=3;%close to far
    else
        dummy(i)=1;%first rew loc
    end 
COM{i,3}=dummy(i);
end