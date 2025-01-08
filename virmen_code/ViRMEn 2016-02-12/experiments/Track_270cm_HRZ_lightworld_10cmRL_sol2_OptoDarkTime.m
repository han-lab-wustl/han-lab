function code = Track_270cm_HRZ_lightworld_10cmRL_sol2_OptoDarkTime
% 200610. EH modify to use contact lick sensor instead of optical

% Hidden Reward Zone experiment code - HRZ

% quakeClone1   Code for the ViRMEn experiment quakeClone1.
%   code = quakeClone1   Returns handles to the functions that ViRMEn
%   executes during engine initialization, runtime and termination.

% Begin header code - DO NOT EDIT
code.initialization = @initializationCodeFun;
code.runtime = @runtimeCodeFun;
code.termination = @terminationCodeFun;
% End header code - DO NOT EDIT

% --- INITIALIZATION code: executes before the ViRMEn engine starts.
function vr = initializationCodeFun(vr)
vr = initializeDAQ_low_view_V_conLick_ROE_imageSync(vr); %200920 EBH
def_ROE_Solenoid_2_Optogen ;

% Change parameters here --------------------------------------------------
vr.gainFactor = 2/3; % Gain in the track. The smaller, the longer the track. 1= usual gain. This value is multiplied by num rpm from ROE
p = 0.4; % probability of having probe trial at the beginning
vr.lickThreshold = -0.065; % might need to adjust
vr.goal = 90; % HERE state the automatic reward location
vr.lickControl = 1; % HERE state the if lick is necessary to get reward
vr.posControl = 1;
vr.countControl = 1;
vr.rewardZone =10*vr.gainFactor; % length of reward zone 
vr.numProbes = 3; % number of probe trials (remember for analysis that their index start from 0)
vr.random_initiation = 1; % randomly initiate reward zone
vr.automaticExperimentRewZone = 1; % automatically switch the reward location after randi(16:21) trials
vr.automaticExperimentWorld = 0; % automatically change the current World after n = vr.automaticExperimentWorldNRewZones reward locations.
vr.automaticExperimentWorld1NRewZones = 1; %how many reward zone in the world 1
vr.automaticExperimentWorld2NRewZones = 5; %how many reward zone in the world 2
vr.world2color = [0.23 0.23 0.23]; %22,27 change the background color on world 2
vr.world2darkTimeColor = [0.22 0.22 0.22]; %15,22 change the background color on 'darktime' in world 2

% Timers - seconds
vr.endTime = 0.2 ;%vr.rewardTime = 1;%time after reaching the end of the track that animal spend in VR
vr.darkTime = 3 ;%vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start, now it goes from 3:5s

%Solenoid 1 to 2 variables
vr.LED2rewDelay = 0.5;%seconds after the solen oid 2 turns on that the reward is given.
vr.LEDpulsewidth = 5/1000; %seconnds that the solenoid 2 is on for;

%GM variables initialization queing
vr.LEDTimer = vr.LED2rewDelay; % initialize the LED to reward timer
vr.LEDcount = 0; % boolean for starting or stopping te LED timer
vr.LEDcount = logical(vr.LEDcount);
vr.pin = 0;
vr.automaticReward = 0; %boolean to separate manual from automatic rewards

%GM variables opto triggers
vr.optoTrigger = {};
% vr.optoEp = [2];
% ZD added prompt for ep stim
prompt_name = 'Pick epoch to stim (2 or 3): '; 
optoEp = inputdlg(prompt_name);
vr.optoEp = [str2double(optoEp)];
vr.optoIson = 0;

% GM variables ------------------------------------------------------------
vr.manualAutoCorrection = 0; % counts the number of reward zones skipped by pressing + early in auto world change
vr.minSuccessfulTrial = 16;
vr.maxSuccessfulTrial = 21;

% Mouse name and current date ---------------------------------------------
% Pop up window asking to state the animal's name and automatically saving
% today's date

prompt_name = 'Type mouse name '; 
mouse_name = inputdlg(prompt_name);
Date = [char(datetime(now,'ConvertFrom','datenum')) '*'];
Date = regexprep(Date,'-','_');
Date = regexprep(Date,':','_');
Date = regexprep(Date,' ','_time(');
Date = regexprep(Date,'*',')');
date{1,1} = Date;
vr.name_date_vr = [mouse_name{1,1} '_' date{1,1}];

% Gain setting ------------------------------------------------------------
% following pop up windows in case someone use the same exp setting
% 
% if vr.gainFactor ~= 1
% %     prompt_automaticG = ['your gainFactor is =  ' num2str(vr.gainFactor) '. Wanna keep it? 1-yes, 0-set a new one '];
% %     answer_automaticG = inputdlg(prompt_automaticG);
%     % ZD commented out prompt - gain factor is the same for all exp
%     answer_automaticG = vr.gainFactor;
%     if answer_automaticG == 0
%         prompt_automaticGN = 'Set GAIN to 1 for normal gain, other values to change gain';
%         answer_automaticGN = inputdlg(prompt_automaticGN);
%         vr.gainFactor = str2double(answer_automaticGN);
%     end
% end


% Automatic world change setting ------------------------------------------
if vr.automaticExperimentWorld == 1
    prompt_automaticW = 'the automatic world change is activated, are you okay with that? 1-yes 0-no ';
    answer_automaticW = inputdlg(prompt_automaticW);
    vr.automaticExperimentWorld = str2double(answer_automaticW);
end

% Settings from previous experiments --------------------------------------
vr.timeSolenoid = 140; % in milliseconds
vr.topTarget = 170; % from "linearTrackTwoDirections"
vr.beginZone = 1; % where the animal start, in cm
vr.worlds{1}.objects.indices.bottomWall = []; %delete the bottom wall EB
vr.worlds{2}.objects.indices.bottomWall = []; %delete the bottom wall EB
vr.currentWorld = 1; % default world at start - we only have 2 worlds in this esperiment
vr.numRewards = 0 ; %actual earned rewards
vr.reward_multiple = 1;

% Lick box ----------------------------------------------------------------
vr.text(1).position = [-3.5 -0.8]; % lower-left corner of the screen
vr.text(1).size = 0.03; % letter size as fraction of the screen
vr.text(1).color = [1 0 0]; % red


% EB variables ------------------------------------------------------------
vr.numMainRewZone = 3; % how many big reward zones in this experiment
vr.isChange = 0 ; % the reward location will change at the end of this trial
vr.sametrial = 0 ;%define if it's new trial
vr.time2EndCheck = 0 ; % value =1 means the timer started
vr.time2End = [] ; % store the time in which the animal reached the end
vr.changeCurrentWorld = 0; % value =1 means the world is about to change
vr.nW1Changes = 1; %how many times the world1 appear. Set to 1 considering the experiment starts from W1.
vr.nW2Changes = 0; %how many times the world2 appear. 
vr.nTrials = randi([vr.minSuccessfulTrial,vr.maxSuccessfulTrial]);%random n of trials   
vr.random_time = round(rand*2,1);%random amount of time in the dark
vr.worldOff = 1; % if on thw world is invisible
vr.eligible = 1; %this trial is eligible for reward
vr.curRewEpoch =  0;

rng('shuffle') % prevent from having the same random numbers each time matlab restart

vr.i=zeros(1,300);
for i=1:3:300
vr.i(i:(i+2)) = randperm(3); %defines a random order for the three main reward zones.
end

corr = find(diff(vr.i)==0); %correct consecutive rew zones
while sum(corr>0)>0 % until there are no more consecutive zones
    for i=1:size(corr,2)
        vr.i(corr(i)+1:corr(i)+3)=nan; % define the triplette that has to be deleted 
    end
    vr.i(isnan(vr.i))=[]; % delete it
    corr = find(diff(vr.i)==0);
end

if vr.random_initiation
    a=[{67:86} {101:120} {135:154}]; % defines the possible target zone
    i=vr.i(1); % defines the region
    vr.goal(1) =a{1,i}(1,randi(length(a{1,i})));
end

tic % start the stopwatch
vr.t = 1; % index to store variables 


%queuing variables
vr.manualNegative = 0;
vr.manualDouble = 0;
vr.singleQueue =  0;
vr.doubleQueue = 0;
vr.negativeCSQueue = 0;
vr.DoubleFollowUp = 0;
vr.PastReward = clock;

% Data that we are collecting ---------------------------------------------
vr.changeRewLoc=vr.goal; %reward location
vr.time=[];
vr.lick=[];
vr.ypos=[];
vr.reward=[];
vr.imageSync=[]; %2000920. EBH
vr.trialNum=[];
if vr.numProbes>0 && rand < p
    I = (0:vr.numProbes -1) ;
    vr.trialNum(1) = I(randi(vr.numProbes )) ;
else
    vr.trialNum(1)=vr.numProbes ; % trial number lower that the one stated here are probe trials
end
vr.worldCurrent=[];
vr.worldCurrent(1)= vr.currentWorld ;
vr.wOff =[];
vr.wOff = vr.worldOff ;
vr.pressedKeys=[];
vr.lickVoltage=[]; % from lick sensor 
vr.roe=[]; % from ROE
vr.timeROE=[]; % from ROE
disp ('==============')
disp (['New session starts here ' mouse_name])
disp (['firsts probe trial probability = ' num2str(p)])
disp (['first trial number is = ' num2str(vr.trialNum(1))])
disp('===')



% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
if vr.position(2)<5 && vr.velocity(2)>0
    vr.collision = 0; % one way wall
    vr.dp(2) = vr.velocity(2)*vr.dt;
end
global lickSensor roe timeROE  imageSync
vr.t=vr.t+1;
vr.imageSync(1,vr.t)=imageSync;
vr.lickVoltage(1,vr.t)=lickSensor;  %state from where we are getting the lick
%%signal remember to update signal file
vr.time(1,vr.t)=toc;
vr.roe(1,vr.t)=roe;% Read current count from the quadrature rotary encoder.
vr.timeROE(1,vr.t)=timeROE; % Time elasped in seconds since Arduino server starts running (double)
vr.lick(1,vr.t)= lickSensor<vr.lickThreshold; %lislick;
vr.dp(2) = vr.dp(2)*vr.gainFactor;

vr.ypos(1,vr.t)=vr.position(2);
vr.trialNum(1,vr.t)=vr.trialNum(vr.t-1);
vr.reward(1,vr.t)=0;
vr.changeRewLoc(1,vr.t)=0;
vr.pressedKeys(1,vr.t)=vr.keyPressed;
vr.worldCurrent(1,vr.t)=vr.currentWorld ;
vr.isReward=0;
%vr.changeCurrentWorld=0;
vr.numRewards=sum(vr.reward);
vr.wOff(1,vr.t)=vr.worldOff;
vr.worlds{1, 1}.backgroundColor = vr.world2color; % change the VR background color; ZD changed for world 1

if vr.currentWorld == 2
    vr.worlds{1, 2}.backgroundColor = vr.world2color; % change the VR background color
end

if vr.worldOff == 1
    vr.worlds{vr.currentWorld}.surface.visible(:) = false ;
elseif vr.worldOff == 0
    vr.worlds{vr.currentWorld}.surface.visible(:) = true ;
end


%%Quing Rewards to after CS-US pair Three kinds :single reward/ double
%%reward/ negative CS (no US)
if vr.singleQueue == 1 % single reward
    if ~vr.LEDcount
        vr.isReward=0.5; %store a signal for the CS
        if vr.DoubleFollowUp ~= 1 %play the CS only when this isn't the consecutive reward from a double queue
        writeDigitalPin(vr.arduino,vr.chSol2,1) %playing cs
        vr.pin = 1;
        else
            vr.DoubleFollowUp = 0; 
        end
        vr.LEDcount = 1; %start the count down from CS to US
        vr.LEDTimer = vr.LED2rewDelay; %reset the timer just in case to 500 ms
        vr.reward(1,vr.t) = vr.isReward; %make sure you store for saving behavior
        vr.singleQueue =  0;%make sure all queues are reset at least for the first frame of a queue
        vr.doubleQueue = 0;
        vr.negativeCSQueue = 0;
        vr.PastReward = clock; %setup the current time to compare for 500 ms
    end
end
if vr.doubleQueue == 1 %double reward
    if ~vr.LEDcount
        vr.isReward=0.5;
        writeDigitalPin(vr.arduino,vr.chSol2,1)
        vr.pin = 1;
        vr.LEDcount = 1;
        vr.LEDTimer = vr.LED2rewDelay;
        vr.manualDouble = 1;
        vr.reward(1,vr.t) = vr.isReward;
        vr.singleQueue =  0;
        vr.doubleQueue = 0;
        vr.negativeCSQueue = 0;
        vr.PastReward = clock;
        vr.DoubleFollowUp = 1;
    end
end
if vr.negativeCSQueue == 1
    if ~vr.LEDcount
        vr.isReward=0.5;
        writeDigitalPin(vr.arduino,vr.chSol2,1)
        vr.pin = 1;
        vr.LEDcount = 1;
        vr.LEDTimer = vr.LED2rewDelay;
        vr.manualNegative = 1;
        vr.reward(1,vr.t) = vr.isReward;
        vr.singleQueue =  0;
        vr.doubleQueue = 0;
        vr.negativeCSQueue = 0;
        vr.PastReward = clock;
    end
end


%%Post LED to Reward ---
% if vr.LEDcount
%     vr.LEDTimer = vr.LEDTimer - vr.dt;
% end
if etime(clock,vr.PastReward) >= vr.LEDpulsewidth && vr.LEDcount && vr.pin
    writeDigitalPin(vr.arduino,vr.chSol2,0)
    vr.pin = 0;
end

if etime(clock,vr.PastReward) >= vr.LED2rewDelay && vr.LEDcount
   
    if  vr.manualNegative == 0
        if  vr.manualDouble == 1
            vr.isReward = 2;
            %             reward_double(vr,vr.timeSolenoid)
            reward(vr,vr.timeSolenoid)
            vr.singleQueue = 1;
            vr.manualDouble = 0;
            
        else
            vr.isReward = 1;
            reward(vr,vr.timeSolenoid)
        end
    end
     vr.manualNegative = 0;
%     vr.timerSaved(length(vr.timerSaved)+1) = vr.timer;
    vr.LEDcount = 0;
    vr.LEDTimer = vr.LED2rewDelay;
end

%Hidden Reward Zone Check System
vr.text(1).color = [1 0 0];
if vr.trialNum(1,vr.t)>vr.numProbes-1 && ((vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2))) || (vr.posControl==0))
    if (vr.eligible==1) || (vr.countControl==0)
        vr.text(1).color = [0 1 0];
        if vr.lickControl==0
                vr.singleQueue = 1;
                vr.eligible=0;
                vr.curRewEpoch =vr.curRewEpoch + 1;
    %             reward(vr,vr.timeSolenoid);
        elseif vr.lickControl==1 && vr.lick(1,vr.t) == 1 
                vr.singleQueue = 1;
                vr.eligible=0;
                vr.curRewEpoch = vr.curRewEpoch + 1;
    %             reward(vr,vr.timeSolenoid);
        end
    else
        vr.text(1).color = [1 1 0];
    end
    
end

if vr.optoIson == 1 % shutting off opto
    if vr.worlds{vr.currentWorld}.surface.visible(:) == true
%     if vr.position(2)>(vr.goal+(vr.rewardZone/2)) || vr.eligible == 0 ZD
%     changed for dark time only opto
        vr.optoIson = 0;
        write(vr.optodeviceObj,[1 0],'uint16')
        vr.optoTrigger = [vr.optoTrigger {[vr.t,1,0]}];        
    end
end
vr.reward(1,vr.t) = vr.isReward;


if vr.lick(1,vr.t) == 1 %display the detected lick
vr.text(1).string = 'L'; 
elseif vr.lick(1,vr.t) == 0
vr.text(1).string = 'N'; 
end

% Key Press section -------------------------------------------------------
% Invisible world
if double(vr.keyPressed == 48)
    if vr.worlds{vr.currentWorld}.surface.visible(:) == false
        vr.worldOff = 0;
    elseif vr.worlds{vr.currentWorld}.surface.visible(:) == true
        vr.worldOff = 1;
    end
end
%give a single reward
if double(vr.keyPressed == 331)
 vr.singleQueue = 1;
end
% Automatic Experiment Rew Zone
if double(vr.keyPressed == 49) && vr.automaticExperimentRewZone == 1 %ascii code for "1"
    vr.automaticExperimentRewZone = 0;
elseif double(vr.keyPressed == 49) && vr.automaticExperimentRewZone == 0
    vr.automaticExperimentRewZone = 1;
end

% Change lickThreshold - increase
if double(vr.keyPressed == 52) %ascii code for "4"
    vr.lickThreshold = vr.lickThreshold+0.001;
    disp(vr.lickThreshold)
end

% Change lickThreshold - decrease
if double(vr.keyPressed == 53) %ascii code for "5"
    vr.lickThreshold = vr.lickThreshold-0.001;
    disp(vr.lickThreshold)
end

% Decrease length rew zone
if double(vr.keyPressed == 55) && vr.rewardZone>=0 %ascii code for "7"
 disp('decrease')
 disp(num2str(vr.rewardZone))
 vr.rewardZone =  vr.rewardZone - 5*vr.gainFactor ;
 disp(num2str(vr.rewardZone))
end

% Increase length rew zone
if double(vr.keyPressed == 56) && vr.rewardZone<vr.topTarget %ascii code for "8"
 disp('increase')
 disp(num2str(vr.rewardZone))
 vr.rewardZone =  vr.rewardZone + 5*vr.gainFactor  ;
  disp(num2str(vr.rewardZone))
end

% Lick control
if double(vr.keyPressed == 57) && vr.lickControl==1 %ascii code for "9"
   vr.lickControl=0; 
elseif double(vr.keyPressed == 57) && vr.lickControl==0
    vr.lickControl=1; 
end

% Pos control
if double(vr.keyPressed == 81) && vr.posControl==1 %ascii code for "q"
   vr.posControl=0; 
elseif double(vr.keyPressed == 81) && vr.posControl==0
    vr.posControl=1; 
end

% Count control
if double(vr.keyPressed == 69) && vr.countControl==1 %ascii code for "e"
   vr.eligible=1; 
end

% Change World
if double(vr.keyPressed == 334)% ascii code for "+ numpad"
    currentNRewLoc=sum(vr.changeRewLoc>0);
    if vr.automaticExperimentWorld == 1 && vr.changeCurrentWorld == 0
        vr.manualAutoCorrection = vr.manualAutoCorrection + vr.automaticExperimentWorld1NRewZones*vr.nW1Changes...
            + vr.automaticExperimentWorld2NRewZones*vr.nW2Changes-currentNRewLoc;
    end
    vr.isChange=1;
    vr.changeCurrentWorld =1;
 
end

% Change reward location - note: it will chamge when the animal reach the
% end of the trial, and it's set at least 30 cm apart from the previous one
if double(vr.keyPressed) == 45 || double(vr.keyPressed) == 95  %ascii code for "-/_"
    vr.isChange=1;
end


%optogenetics transmition 
if double(vr.keyPressed)== 72 % if "h" the key right under escape is pressed
   write(vr.optodeviceObj,[1 1000],'uint16')
   vr.optoTrigger = [vr.optoTrigger {[vr.t,1,1000]}];
end

% End Key Press section ---------------------------------------------------
tempNumSuccTrials = vr.curRewEpoch;
% tempNumSuccTrials = sum(vr.reward(find(vr.changeRewLoc,1,'last'):vr.t)==1); % count the number of succesful trial in the current reward location

if vr.automaticExperimentRewZone == 1 && tempNumSuccTrials == vr.nTrials
    vr.isChange=1; 
    vr.nTrials = randi([vr.minSuccessfulTrial,vr.maxSuccessfulTrial]);
    vr.curRewEpoch = 0;
end

if  vr.isChange==1 && vr.position(2)==vr.beginZone
    a=[{67:86} {101:120} {135:154}];%defines the regions, here we have three
    indx=vr.i(sum(vr.changeRewLoc>=1)+1); %choose the region
    tempdist = abs(vr.changeRewLoc(find(vr.changeRewLoc,1,'last'))-a{1,indx}); % which locations are 30 cm apart from the previous one
    rewZoneIdx = find (tempdist > 30); % find only the good indexes
    temp = a{1,indx}(rewZoneIdx(randi(length(rewZoneIdx)))); %randomly choose one of them
    vr.goal = temp; %set the new reward zone
    vr.isChange=0;
    disp('===')
    disp(['Total Trial = ' num2str(vr.trialNum(1,vr.t)+1)])
    pastRewLoc = vr.changeRewLoc(vr.changeRewLoc>0);
    pastRewLoc = pastRewLoc(length(pastRewLoc-1));
    disp(['Rew Locations = ' num2str(pastRewLoc)])
    disp(['Reward total = ' num2str(sum(vr.reward(find(vr.changeRewLoc>0,1,'last'):vr.t)))])
    disp('===')
    vr.trialNum(1,vr.t)=0;
    vr.changeRewLoc(1,vr.t)=temp;
end

if  vr.collision && vr.position(2)>vr.topTarget  && vr.time2EndCheck == 0 %vr.position(2)>= vr.topTarget
    vr.time2End=vr.time(1,vr.t); % at what time the animal reached the top target
    vr.time2EndCheck = 1 ; % value =1 means the timer started
end 

if vr.automaticExperimentWorld == 1  
    currentNRewLoc=sum(vr.changeRewLoc>0);
    if currentNRewLoc>vr.automaticExperimentWorld1NRewZones*vr.nW1Changes + vr.automaticExperimentWorld2NRewZones*vr.nW2Changes-vr.manualAutoCorrection && vr.currentWorld == 1
         vr.changeCurrentWorld = 1; 
    end
    if currentNRewLoc>vr.automaticExperimentWorld1NRewZones*vr.nW1Changes + vr.automaticExperimentWorld2NRewZones*vr.nW2Changes-vr.manualAutoCorrection && vr.currentWorld == 2
        vr.changeCurrentWorld = 1; 
    end
end

if  vr.time2EndCheck == 1
    if  (vr.time(1,vr.t)-vr.time2End)>=vr.endTime % chech the animal stayed at the end for the desired time
        vr.worlds{vr.currentWorld}.surface.visible(:) = false;
        vr.worldOff = 1;
        vr.position(2) = vr.beginZone;
        vr.position(1) = 0;
        vr.position(4) = 0;
        vr.dp(1:4) = [0 0 0 0];
        vr.sametrial = 0;
        vr.eligible=1;
        vr.trialNum(1,vr.t)=vr.trialNum(1,vr.t)+1;
        vr.time2dark=vr.time(1,vr.t);
        vr.time2EndCheck = 0; % stop timer for the end of the track
        vr.random_time=round(rand*2,1); %choose rnd amount of time in the dark
        
        if  vr.changeCurrentWorld == 1 && vr.trialNum(1,vr.t)==vr.numProbes
            vr.currentWorld = abs(floor(vr.currentWorld/2)-1)+1;
            if vr.currentWorld ==1
                vr.nW1Changes = vr.nW1Changes+1;
            elseif vr.currentWorld == 2
                vr.nW2Changes = vr.nW2Changes+1;
            end        
            vr.changeCurrentWorld=0;
            vr.trialNum(1,vr.t) = 0; %EB 21/02/15 three more probe trials
        end
        %GM 11/13/2023 Start Opto at start of dark time for sepcific epoch
        if sum(vr.optoEp == sum(vr.changeRewLoc~=0))>0 && vr.trialNum(1,vr.t)>vr.numProbes-1
            write(vr.optodeviceObj,[1 65535],'uint16')
            vr.optoTrigger = [vr.optoTrigger {[vr.t,1,65535]}];
            vr.optoIson = 1;
        end
        
        
    end
end
if vr.position(2)== vr.beginZone && vr.time(1,vr.t)>=vr.time2dark+((vr.darkTime-1)+vr.random_time) % variable length in dark
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
    vr.worldOff = 0;
end

if vr.worlds{vr.currentWorld}.surface.visible(:) == false % prevent the VR from moving when the screen is black
    vr.dp(1:4) = [0 0 0 0];
    check = find(diff(vr.trialNum)==1,1,'last');
    if vr.currentWorld == 2 && vr.worldCurrent(check) ~= 1
    vr.worlds{1, 2}.backgroundColor = vr.world2darkTimeColor; % change the VR background color
    else
    vr.worlds{1, 2}.backgroundColor = [0 0 0]; 
    end
end




% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)
    write(vr.optodeviceObj,[1 0],'uint16') %make sure opto is off

    disp('===')
    disp(['Total Trial = ' num2str(vr.trialNum(1,vr.t)+1)])
    pastRewLoc = vr.changeRewLoc(vr.changeRewLoc>0);
    pastRewLoc = pastRewLoc(length(pastRewLoc));
    disp(['Rew Locations = ' num2str(pastRewLoc)])
    disp(['Reward total = ' num2str(sum(vr.reward(find(vr.changeRewLoc>0,1,'last'):vr.t)))])
    disp('===')
name_date_vr=vr.name_date_vr;
VR.name_date_vr=name_date_vr;
VR.ROE=vr.roe ;
VR.lickThreshold = vr.lickThreshold ;
VR.reward=vr.reward ;
VR.time=vr.time ;
VR.lick=vr.lick ;
VR.ypos=vr.ypos ;
VR.lickVoltage=vr.lickVoltage ;
VR.trialNum=vr.trialNum ;
VR.timeROE=vr.timeROE ;
VR.optoTrigger = vr.optoTrigger;
VR.changeRewLoc=vr.changeRewLoc ;
VR.pressedKeys=vr.pressedKeys ;
VR.world=vr.worldCurrent ;
VR.imageSync=vr.imageSync ;%2000920. EBH
VR.scalingFACTOR=vr.gainFactor ;
VR.wOff = vr.wOff ;
vr.gainFactor = 1; % Gain in the track. The smaller, the longer the track. 1= usual gain. This value is multiplied by num rpm from ROE

settings.name = vr.exper.name; % name of the actual experiment
settings.lickThreshold = vr.lickThreshold; % might need to adjust
settings.goal = vr.goal; % HERE state the automatic reward location
settings.lickControl = vr.lickControl; % HERE state the if lick is necessary to get reward
settings.rewardZone = vr.rewardZone; % length of reward zone 
settings.numProbes = vr.numProbes; % number of probe trials (remember for analysis that their index start from 0)
settings.random_initiation = vr.random_initiation; % randomly initiate reward zone
settings.automaticExperimentRewZone = vr.automaticExperimentRewZone; % automatically switch the reward location after randi(22:27) trials
settings.automaticExperimentWorld = vr.automaticExperimentWorld; % automatically change the current World after n = vr.automaticExperimentWorldNRewZones reward locations.
settings.automaticExperimentWorld1NRewZones = vr.automaticExperimentWorld1NRewZones; %how many reward zone in the world 1
settings.automaticExperimentWorld2NRewZones = vr.automaticExperimentWorld2NRewZones; %how many reward zone in the world 2
% Timers - seconds
settings.endTime = vr.endTime ;%vr.rewardTime = 1;%time after reaching the end of the track that animal spend in VR
settings.darkTime = vr.darkTime;
settings.minSuccessfulTrial = vr.minSuccessfulTrial;
settings.maxSuccessfulTrial = vr.maxSuccessfulTrial;
settings.OptoEp = vr.optoEp;

VR.settings = settings;

if exist('name_date_vr','var')
    save(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\' name_date_vr '.mat'],'VR') 
    load('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date');
    name_date{size(name_date,1)+1,1}=name_date_vr;
    save('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date')
end
% plotVR %200925 eh. crash, Unable to read file
disp (['first trial number was = ' num2str(vr.trialNum(1))])
