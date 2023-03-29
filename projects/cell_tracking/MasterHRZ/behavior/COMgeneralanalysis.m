version = ver;
version = version.Release;
version = version(5:6);

%EH variables 
rewWinNeg=10;% EH -cm from rewStart peri-reward window
rewWinPos=5;% EH +cm peri-reward window
if isfield(VR,'settings')
rewSize=VR.settings.rewardZone/2; %EH 15cm rew zone + -
else
    rewSize = 15/2;
end
numTrialsStart=5; %EH number of trials to average at start and end of rew epoch
numTrialsEnd=5; %EH number of trials to average at start and end of rew epoch

% GM Variables
speedbinsize = 3;
conversion = -0.013;
slidingwindow = 5;
testlearningposition = 60; %cm
%%
changeRewLoc=find(VR.changeRewLoc); %find the change in rew locations
changeRewLoc = [changeRewLoc length(VR.changeRewLoc)]; %vector containing where each reward location ends

RewLoc=VR.changeRewLoc(changeRewLoc(1:end-1)); %find the rew locations (in cm)
RewLocStart=RewLoc-rewSize; %find the start of rew locations (in cm)%EH

VR.lick = correct_artifact_licks(VR.ypos,VR.lick); % the function can be found at the end of this script
   
    for ll =2:length(VR.ROE) %deletes large ROE spikes if missed by VR
        if (VR.ROE(ll)-VR.ROE(ll-1)) <= -45 && (VR.ROE(ll)) < -5 && (VR.ROE(ll-1)) < -5
        VR.ROE(ll) = VR.ROE(ll-1);
        end
    end
   
 %converts ROE to Speed
  VR.ROE = (conversion*VR.ROE)./[0 diff(VR.time)];
  %
  
  
% pre-allocated variables
COM = cell(length(changeRewLoc)-1,1); %COM of only succesfull trials for each rew locations (rows are in temporal order)
stdCOM = cell(length(changeRewLoc)-1,1); %std of ypos of licks for each succesfull trials
allCOM = cell(length(changeRewLoc)-1,1); %COM of all trials for each rew locations
allstdCOM = cell(length(changeRewLoc)-1,1); %std of ypos of licks for all trials
ROE = cell(length(changeRewLoc)-1,1); %binned velocity for all trials
meanROE = cell(length(changeRewLoc)-1,1); %average velocity per trial
stdROE = cell(length(changeRewLoc)-1,1); %standard deviation of velocity per trial
ROEcount = cell(length(changeRewLoc)-1,1);%number of data points per bin for debugging
binypos = cell(length(changeRewLoc)-1,1);% indices of VR.ypos for each bin
binstarted = cell(length(changeRewLoc)-1,1);% starting bin for every trial for debugging
slidingMeanCOM = cell(length(changeRewLoc)-1,1);
slidingVarCOM = cell(length(changeRewLoc)-1,1);

cutROE = cell(length(changeRewLoc)-1,1);

meancutROE = cell(length(changeRewLoc)-1,1);
semcutROE = cell(length(changeRewLoc)-1,1);
timecount = cell(length(changeRewLoc)-1,1);

InterLickInterval = [];

allRatio = cell(length(changeRewLoc)-1,1); %EH ratio of peri-reward licks for all trials for each rew locations
% allLickDist = cell(length(changeRewLoc)-1,1); %EH lick dist from rew start
%%
if sum(VR.lick>0)==0
    disp('This session has no licks')
else 
for kk = 1:length(changeRewLoc)-1 % for each reward location...
        if isfield(VR,'trialNum')
            trialNum = VR.trialNum;
        elseif isfield(VR,'trials')
            trialNum = VR.trials;
        end
        numProbe = 3; % same as numprobe from runtime code.
        if sum(VR.name_date_vr=='_')==1
            VR.reward((length(VR.reward)+1):length(VR.lick))=0;
        end
        if length(VR.reward)~=length(VR.lick)
            VR.reward((length(VR.reward)+1):length(VR.lick))=0;
        end
        
        rewStart=(RewLoc(kk)-rewSize);% EH define peri-reward window
        periLow=rewStart-rewWinNeg;% EH lower end of window
        periHigh=rewStart+rewWinPos;% EH upper end of window
        
        % trials stucture
        difftrials = diff(trialNum(changeRewLoc(kk):changeRewLoc(kk+1)-1));
        difftrials = [0 difftrials];
        if trialNum(:,1) == numProbe && kk == 1
            difftrials(1) = 1;
        end
        startnotprobe = find(trialNum(changeRewLoc(kk):end)>(numProbe-1),1,'first')+changeRewLoc(kk)-1; % find the starting of each reward locaton after the probe trials. (NB: in the cronological order in VR, when the rew location change occur, the first three trials (recognized as trial number 0,1,2) are probe trials.)
        trials=find(difftrials>=1 & trialNum(changeRewLoc(kk):changeRewLoc(kk+1)-1)>(numProbe-1))+changeRewLoc(kk)-1; % find the starting of each trial within reward location
        
        for jj = 1:length(trials)-1 %for each trial...
            if find(VR.reward(trials(jj)+1:trials(jj+1)))>0 % if is a succesfull trial
                licking = find(VR.lick(trials(jj)+1:find(VR.reward(trials(jj)+1:trials(jj+1)))+trials(jj)))+trials(jj); % find all the licks before and equal to the reward lick
                licking([2 diff(licking)]==1) = [];%EH, remove consecutive licks. concatenate 2 to keep 1st lick and shift all to maintain index
                if length(licking)>0
                    licking(length(licking)) = find(VR.reward(trials(jj)+1:trials(jj+1)))+trials(jj); % GM ensures after consecutive removal that the reward lick is on the same index as the reward received
                end
                periLick=(VR.ypos(licking)>periLow & VR.ypos(licking)<periHigh);  %EH logical of peri-reward licks
                ratio=(sum(periLick)/length(periLick));%EH ratio of peri to total licks for trial (pre reward, including reward lick)
                
%                 COM{kk} = [COM{kk} mean(VR.ypos(licking))-RewLoc(kk)]; %calculate the normalized COM for each trial and store it
%                 stdCOM{kk} = [stdCOM{kk} std(VR.ypos(licking))-RewLoc(kk)]; %calculate the std of the COM for each trial and store it
                
                COM{kk} = [COM{kk} mean(VR.ypos(licking))-RewLoc(kk)]; %calculate the normalized COM for each trial and store it
                stdCOM{kk,jj} = [stdCOM{kk} std(VR.ypos(licking))-RewLoc(kk)]; %calculate the std of the COM for each trial and store it
                
                lickDist{kk,jj}=abs(VR.ypos(licking)-RewLocStart(kk));%EH abs distace of each lick from rewStart
                
                %GM
                start = find(diff(VR.ypos(trials(jj)+1:trials(jj+1))),1,'first')+trials(jj); %defines start and stop indices for binning, start is from the moment they are allowed to move.
                stop = find(VR.reward(trials(jj)+1:trials(jj+1)))+trials(jj);
                binstart = floor(VR.ypos(start)); %defines the ypos of the bins
                binstop = ceil(VR.ypos(stop));
                          
                failure{kk}(jj) = 0; % keep track of failed trials
            else
                licking = find(VR.lick(trials(jj)+1:(trials(jj+1))))+trials(jj); % else, find all the licks of that trial
                licking([2 diff(licking)]==1) = [];%EH, remove consecutive licks. concatenate 2 to keep 1st lick and shift all to maintain index
                lickDist{kk,jj}=abs(VR.ypos(licking)-RewLocStart(kk));%EH abs distace of each lick from rewStart
                %lickDist is array with {reward epoch(rows), trial(columns)}
                %however #columns(trials) vary so some columns have []
                %trickier foe extraction, see below
                failure{kk}(jj) = 1;
                
                periLick=(VR.ypos(licking)>periLow & VR.ypos(licking)<periHigh);  %EH logical of peri-reward licks
                ratio=(sum(periLick)/length(periLick));%EH ratio of peri to total licks for trial (pre reward, including reward lick)
                
                %GM
                start = find(diff(VR.ypos(trials(jj)+1:trials(jj+1))),1,'first')+trials(jj); %defines start and stop indices for binning
                stop = (trials(jj+1));
                binstart = floor(VR.ypos(start)); %defines the ypos of the bins
                binstop = ceil(VR.ypos(stop));
                
                
            end
            %GM InterLickInterval
            
            previoussize = size(InterLickInterval,3);
            InterLickInterval(kk,jj,1:length(licking)-1) = diff(VR.time(licking)); %fill in interlick interval data per trial and epoch
            filledintervals = length(licking)-1; %count the number of lick intervals there are
            if filledintervals <= 0 && failure{kk}(jj) == 0 %if the mouse successfully licked only 1 time(rewarded lick)
                InterLickInterval(kk,jj,1) = 0; %add an interval of 0
                filledintervals = 1; %say that there was one interval
            elseif filledintervals <= 0 && failure{kk}(jj) == 1 %otherwise if the mouse failed because of not licking at all
                InterLickInterval(kk,jj,1) = NaN; %say there were no intervals
                filledintervals = 1; %and add one value to fill the rest with NaN as well to pad the  matrix
            end
            if filledintervals <size(InterLickInterval,3) %since there can be different number of intervals...
                InterLickInterval(kk,jj,filledintervals+1:end) = NaN;  %.. fill the spaces of trials with less licks with Nan to distinguish from intervals of 0 (reward lick only trials)
            end
            if filledintervals > previoussize % If this trial has more licks than all previous trials... and this is not the first trial...
                if kk == 1 && jj == 1
                else
                    if kk > 1
                        InterLickInterval(1:kk-1,:,previoussize+1:end) = NaN;
                    end
                    if jj > 1
                    InterLickInterval(kk,1:jj-1,previoussize+1:end) = NaN; %fill the previous trials with Nan
                    end
                   
                end
            end
            
                
            %Velocity Setup GM
            if rem(binstop-binstart,speedbinsize)~=0 %rewrites binstop to be a multiple of the speedbinsize
                binstop = binstop +speedbinsize-rem(binstop-binstart,speedbinsize);
            end
            [~,speedlearningbin] = min(abs((binstart:speedbinsize:binstop-speedbinsize) - testlearningposition));
            for bb = binstart:speedbinsize:binstop-speedbinsize %for every bin...
                if length(find(VR.ypos(start:stop)>=bb & VR.ypos(start:stop)<=bb+speedbinsize)+start-1) > 0
                bintime = VR.time(find(VR.ypos(start:stop)>=bb & VR.ypos(start:stop)<=bb+speedbinsize)+start-1);
                 timecount{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = bintime(end)-bintime(1);
                else
                    timecount{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = NaN;
                end
                
             
                ROEcount{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = length(find(VR.ypos(start:stop)>=bb & VR.ypos(start:stop)<=bb+speedbinsize)+start-1); %count how many points are in this bin
                ROE{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = mean((VR.ROE(find(VR.ypos(start:stop)>=bb & VR.ypos(start:stop)<bb+speedbinsize)+start-1))); %write the average vel for this bin
                if ~isnan(ROE{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize)) %collects the indices for this bin for plotting ypos later
                    binypos{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = (find(VR.ypos(start:stop)>=bb & VR.ypos(start:stop)<=bb+speedbinsize,1)+start-1);
                else
                    binypos{kk}{jj}((bb-binstart+speedbinsize)/speedbinsize) = NaN;
                end
            end
            cutROE{kk}{jj} = ROE{kk}{jj}(1:speedlearningbin);
            %velocity only up to position 60 test GM
            
            
            
            trialyposlicks{kk}{jj} = VR.ypos(licking); %store the ypos of each lick
%             allCOM{kk}(jj) =mean(VR.ypos(licking))-RewLoc(kk); %calculate the normalized COM of all the trials successfull of not. Note that for failed trials all the licks are included, not only the ones before the rew loc
            allCOM{kk}(jj) =mean(VR.ypos(licking))-RewLoc(kk); %EH shift to start of rew zone. calculate the normalized COM of all the trials successfull of not. Note that for failed trials all the licks are included, not only the ones before the rew loc
            allstdCOM{kk}(jj) = std(VR.ypos(licking))/sqrt(length(licking));
            allRatio{kk}(jj)=ratio; %EH
            
            %GM
            meanROE{kk}(jj) = nanmean(ROE{kk}{jj});
            stdROE{kk}(jj) = std(ROE{kk}{jj});
            binstarted{kk}(jj) = binstart; %saves the starting ypos for each trial for debugging
            
            meancutROE{kk}(jj) = nanmean(cutROE{kk}{jj});
            semcutROE{kk}(jj) = nanstd(cutROE{kk}{jj})/sqrt(sum(~isnan(cutROE{kk}{jj})));
            
%             allLickDist{kk}(kk,jj)=lickDist{kk}; %EH
%             allLickDist{kk}(jj,(1:length(lickDist{kk})))=lickDist{kk}; %EH


        end
   
        if length(COM{kk})-slidingwindow+1>1
            for i=1:length(COM{kk})-slidingwindow+1
                slidingMeanCOM{kk}(i) = mean(COM{kk}(i:i+slidingwindow-1));
                slidingVarCOM{kk}(i) = var(COM{kk}(i:i+slidingwindow-1));
            end
        else
            slidingMeanCOM{kk} = NaN;
            slidingVarCOM{kk} = NaN;
        end
    end
    
    

    
end 
time_min = VR.time/60; % create the x axes of the raw data plot in munutes(each VR datapoint collected with the function tic toc)
ypos = VR.ypos; % create the y axes of the raw data plot showinf the animal position
rewX = time_min(find(VR.reward));% create the x axes of the raw data plot for reward
rewYpos = VR.ypos(find(VR.reward));% create the y axes of the raw data plot for reward
islickX = time_min(find(VR.lick)); % create the x axes of the raw data plot for licks
islickYpos = VR.ypos(find(VR.lick));% create the y axes of the raw data plot for licks