 function code = Track_180cm_teleport_ROE_conLick_tillTheEnd_phase1Training0427
% 200610. EH modify to use contact lick sensor instead of optical
% last updated 20200203
% - world 1: unidirectional run, reward at end, teleport to beginning

% - licking n times in t secs gives licksuccess
% - gamemode (0/1) decides whether run pass target counts as success (0) or
% has to lick in a zone(1)
% - reward is given when licksuccess and positionsuccess
% - teleport is delayed: rewardTime states the amount of time after reward for animal to
% explore,rewarddarkTime states the amount of the time in dark before
% teleport

% more reward is given for gamemode to because they don't get as many
% reward




% Caradini progressive portocol equivalent:
% 1. numbertolick = 0; % always reward after pass the point
% 2. gamemode = 1, numbertolick = 1,timetolick = 0.5 (doesn't really matter when numbertolick = 1?),
% progressively reduce vr.rewardzone
% 3. progressively increase numbertolick?
% 4. Introduce punishment for licking outside reward zone in the future

%switches rotation gain in track in new world
%number of rewards can by modified during run
%"3" = 3 rew, "2"= 2. gives rew at switch.
%default set at 3 rew in initialization of vr.reward_multiple
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

vr = initializeDAQ_low_view_V_conLick_ROE_imageSync(vr); %200610 EH
def_ROE_Solenoid_2

% Number of reward zones(whole track =
% 12)---------------------------------
vr.gainFactor = 2/3;
vr.numRewZones=2; %set the desired number of reward zones
vr.rewardZone = 5*vr.gainFactor; % length of reward zone after currentGoal
vr.allGoals =15:(vr.rewardZone):180; % set default goals [world 1 world 2].
vr.legalRewZones=find(vr.allGoals,1,'last'); % rew zones where the animal can get reward
vr.legalRewZones=randperm(vr.legalRewZones,vr.numRewZones);

%Solenoid 1 to 2 variables
vr.LED2rewDelay = 0.5;%seconds after the solen oid 2 turns on that the reward is given.
vr.LEDpulsewidth = 5/1000; %seconnds that the solenoid 2 is on for;

%GM variables initialization
vr.LEDTimer = vr.LED2rewDelay; % initialize the LED to reward timer
vr.LEDcount = 0; % boolean for starting or stopping te LED timer
vr.LEDcount = logical(vr.LEDcount);
vr.pin = 0;
vr.automaticReward = 0; %boolean to separate manual from automatic rewards

%--------------------------------------------------------------------------
vr.t=1; %iteration index
vr.lickThreshold = -0.1;% might need to adjust, unfortunately the clampex recording is not reliable, use licksensor_toy
prompt_name='type mouse name ';
mouse_name=inputdlg(prompt_name);
Date=char(today('datetime'));
expression='-';
replace='_';
Date=regexprep(Date,expression,replace);
date{1,1}=Date;
vr.name_date_vr=[mouse_name{1,1} '_' date{1,1}];
vr.gamemode = 1; % decide if the lick outside "reward zone" counts % =1 EB 0630
vr.changeRewLoc{1}=vr.allGoals; %save the initial reward locations
vr.timeSolenoid = 140; %in milliseconds
vr.topTarget = 170;%from "linearTrackTwoDirections"
vr.bottomTarget = 10;%new term
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
vr.RotGainFactor= 1;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1; % default world at start
vr.totaltrialcount =0; % initialize
vr.numRewards = 0;%actual earned rewards
vr.currentGain = 1;%gain world. 1=default, 2=gainFactor
vr.rewardTime = 1;%time after reaching the end of the track that animal still in VR?
vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start, now it goes from 3:5s
vr.sametrial = 0;%define if it's new trial
vr.rewardTimer = 0.5; 
vr.reward=0;
vr.trialNum=0; 
vr.lickVoltage=[];
vr.reward_multiple = 1;
vr.positionsuccess=0; %%EB!
vr.endtimecheck =0;% reach the end without successful lick/pass target EB!
vr.firsttimecheck = 0;% initialize

%queuing variables
vr.manualNegative = 0;
vr.manualDouble = 0;
vr.singleQueue =  0;
vr.doubleQueue = 0;
vr.negativeCSQueue = 0;
vr.DoubleFollowUp = 0;
vr.PastReward = clock;


tic
vr.time(vr.t)=toc;
vr.roe=[];
vr.timeROE=[];
vr.lick=[];
vr.ypos=[];
vr.rewZone=1;
%200610 EH
%vr.licksensor = readVoltage(arduino('COM3','Uno')); %calles the arduino for lick sensor % check device manager for COMx
vr.numbertolick = 1; % integer,n licks to be successful, set to 0 to make licksuccess always 1
vr.timetolick = 1; % secs,the first lick and nth lick has to be less than t sec apart, set to 0 to make licksuccess always 0
%don't think vr.timeElapsed is defined in function
vr.lick_t = [];%vr.timeElapsed for each lick. reset to [] at  vr.timeElapsed not defined!
% vr.framerateMin = 0.035; %seconds % make it wait until 0.03 if it is faster than that
vr.startTime = vr.time;
% Lick box ----------------------------------------------------------------
vr.text(1).position = [-3.5 -0.8]; % lower-left corner of the screen
vr.text(1).size = 0.03; % letter size as fraction of the screen
vr.text(1).color = [1 0 0]; % 
vr.worldOff = 1;
vr.timer = 0;
vr.darkautomaticturnon = 1; %whether or not to turn on the world automatically aft initialdarktime seconds
vr.initialdarktime = 120; %seconds before world turns on automatically
vr.worlds{1}.surface.visible(:) = false;



% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
global lickSensor roe timeROE
vr.t=vr.t+1;
vr.lickVoltage(vr.t)=lickSensor;  %state from where we are getting the lick
%%signal remember to update signal file
vr.time(vr.t)=toc;
vr.roe(vr.t)=roe;% Read current count from the quadrature rotary encoder.
vr.timeROE(vr.t)=timeROE; % Time elasped in seconds since Arduino server starts running (double)
vr.lick(vr.t)=vr.lickVoltage(vr.t)<vr.lickThreshold;
vr.ypos(vr.t)=vr.position(2);
vr.trialNum(vr.t)=vr.trialNum(vr.t-1);
vr.numRewZonest(vr.t) = vr.numRewZones;
vr.isReward=0;
vr.reward(vr.t)=0;
rewZone=abs(vr.allGoals-vr.position(2));
one = find(rewZone==min(rewZone)); %make sure there are not two
vr.rewZone(vr.t)= one(1,1); %vector that store the animal position relatively to all the reward zones
vr.dp(2) = vr.dp(2)*vr.gainFactor;

if vr.lick(1,vr.t) == 1 %display the detected lick
vr.text(1).string = 'L'; 
elseif vr.lick(1,vr.t) == 0
vr.text(1).string = ' '; 
end
if vr.worldOff == 1
    vr.worlds{1}.surface.visible(:) = false ;
elseif vr.worldOff == 0
    vr.worlds{1}.surface.visible(:) = true ;
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
    vr.timerSaved(length(vr.timerSaved)+1) = vr.timer;
    vr.LEDcount = 0;
    vr.LEDTimer = vr.LED2rewDelay;
end

%multiple reward system
if sum(vr.rewZone(vr.t)==vr.legalRewZones)==1
    if vr.lick(vr.t) && vr.position(2)>10 && sum(vr.reward(vr.trialNum==vr.trialNum(vr.t)))==0
        vr.singleQueue = 1;
    elseif vr.lick(vr.t) && vr.position(2)>10 && sum(vr.reward)>0
        a=find(vr.reward,1,'last');
        if vr.rewZone(a)==vr.rewZone(vr.t)
            vr.isReward=0;
        elseif vr.position(2)>vr.ypos(a) && (vr.time(vr.t)-vr.time(a))>=vr.rewardTimer
            vr.singleQueue = 1;
        end
    end
end

if toc > vr.initialdarktime && vr.darkautomaticturnon == 1
    vr.worldOff = 0;
    vr.darkautomaticturnon = 0;
end

if  vr.position(2)>= vr.topTarget && vr.collision
    vr.worlds{vr.currentWorld}.surface.visible(:) = false;
    vr.worldOff = 1;
    vr.position(2) = vr.beginZone;
    vr.position(1) = 0;
    vr.position(4) = 0;
    vr.dp(1:4) = [0 0 0 0];
    vr.sametrial = 0;
    vr.trialNum(vr.t)=vr.trialNum(vr.t)+1;
    vr.time2dark=vr.time(vr.t);
    vr.legalRewZones=find(vr.allGoals,1,'last'); % change rew zones where the animal can get reward
    vr.legalRewZones=randperm(vr.legalRewZones,vr.numRewZones);
end
    
if vr.position(2)== vr.beginZone && vr.time(vr.t)>=vr.time2dark+(vr.rewarddarkTime+(randi([1,2])+(randi([0,10])/10))) % variable length in dark
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
    vr.worldOff = 0;
end

if vr.worlds{vr.currentWorld}.surface.visible(:) == false
    vr.dp(1:4) = [0 0 0 0];
end
% key press----------------------------------------------------------------
% Invisible world
if double(vr.keyPressed == 48)
    if vr.worlds{vr.currentWorld}.surface.visible(:) == false
        vr.worldOff = 0;
    elseif vr.worlds{vr.currentWorld}.surface.visible(:) == true
        vr.worldOff = 1;
    end
end
if double(vr.keyPressed == 50) && vr.numRewZones>1 %ascii code for "2"
vr.numRewZones=vr.numRewZones-1; %decrease the number of reward zones
vr.legalRewZones=find(vr.allGoals,1,'last'); % rew zones where the animal can get reward
vr.legalRewZones=randperm(vr.legalRewZones,vr.numRewZones);
vr.changeRewLoc{vr.t}=vr.allGoals(vr.legalRewZones); %save new legal reward locations
end

if double(vr.keyPressed == 51) && vr.numRewZones<12 %ascii code for "3"
vr.numRewZones=vr.numRewZones+1; %increase the number of reward zones
vr.legalRewZones=find(vr.allGoals,1,'last'); % rew zones where the animal can get reward
vr.legalRewZones=randperm(vr.legalRewZones,vr.numRewZones);
vr.changeRewLoc{vr.t}=vr.allGoals(vr.legalRewZones); %save new legal reward locations
end

if double(vr.keyPressed == 52) %ascii code for "4"
    vr.lickThreshold = vr.lickThreshold+0.01;
    disp(vr.lickThreshold)
end

if double(vr.keyPressed == 53) %ascii code for "5"
    vr.lickThreshold = vr.lickThreshold-0.01;
    disp(vr.lickThreshold)
end
%--------------------------------------------------------------------------
% if vr.isReward
%     switch vr.reward_multiple
%         case 3
%             reward_triple(vr,vr.timeSolenoid);
%         case 2
%             reward_double(vr,vr.timeSolenoid);
%         case 1
%             reward(vr,vr.timeSolenoid);
%     end
% end

%     if vr.isReward
        vr.reward(vr.t)=vr.isReward; 
%     end



% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)
name_date_vr=vr.name_date_vr;
VR.name_date_vr=name_date_vr;
VR.ROE=vr.roe;
VR.lickThreshold = vr.lickThreshold;
VR.reward=vr.reward;
VR.time=vr.time;
VR.lick=vr.lick;
VR.ypos=vr.ypos;
VR.lickVoltage=vr.lickVoltage;
VR.trials=vr.trialNum;
VR.timeROE=vr.timeROE;
VR.rewZone=vr.rewZone;
VR.changeRewLoc=vr.changeRewLoc;
VR.numRewZones = vr.numRewZonest;
save(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\phase1\' name_date_vr '.mat'],'VR')

