function code = Track_180cm_teleport_ROE_conLick_tillTheEnd_EB0826
% 200610. EH modify to use contact lick sensor instead of optical

% Hidden Reward Trainig experiment code - HRT


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
vr = initializeDAQ_low_view_V_conLick_ROE(vr); %200610 EH
def_ROE ;

% Change parameters here --------------------------------------------------

vr.lickThreshold = -0.032; % might need to adjust
vr.goal = 90; % HERE state the automatic reward location
vr.lickControl = 0; % HERE state the if lick is necessary to get reward
vr.rewardZone = 15; % length of reward zone 
vr.numProbes = 3; % number of probe trials
vr.random_initiation = 1; % randomly initiate reward zone

% Timers - seconds
vr.endTime = 1 ;%vr.rewardTime = 1;%time after reaching the end of the track that animal spend in VR
vr.darkTime = 3 ;%vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start, now it goes from 3:5s

% Mouse name and current date ---------------------------------------------
% Pop up window asking to state the animal's name and automatically saving
% today's date

prompt_name = 'type mouse name ';
mouse_name = inputdlg(prompt_name);
Date = char(today('datetime'));
Date = regexprep(Date,'-','_');
date{1,1} = Date;
vr.name_date_vr = [mouse_name{1,1} '_' date{1,1}];

%--------------------------------------------------------------------------
vr.timeSolenoid = 140; % in milliseconds
vr.topTarget = 170; % from "linearTrackTwoDirections"
vr.bottomTarget = 10; 
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
vr.RotGainFactor= 1;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1; % default world at start
vr.numRewards = 0 ; %actual earned rewards
vr.currentGain = 1;%gain world. 1=default, 2=gainFactor
vr.rewardTimer = 0.5; 
vr.reward_multiple = 1;
% EB
vr.numMainRewZone = 3; % how many big reward zones in this experiment
vr.isChange = 0 ; % the reward location will change at the end of this trial
vr.sametrial = 0 ;%define if it's new trial
vr.time2EndCheck = [] ; % value =1 means the timer started
rng('shuffle') % prevent from having the same random numbers each time matlab restart
vr.i = randperm(vr.numMainRewZone,3); %defines a random order for the three main reward zones.
if vr.random_initiation
    a=[{60:93} {94:127} {128:161}]; % defines the possible target zone
    i=vr.i(1); % defines the region
    vr.goal(1) =a{1,i}(1,randi(length(a{1,i})));
end

tic % start the stopwatch
vr.t=1; % index to store variables

% data that we are collecting ---------------------------------------------

vr.changeRewLoc=vr.goal; %reward location
vr.time=toc;
vr.lick=[];
vr.ypos=[];
vr.reward=[];
vr.trialNum=vr.numProbes ; % trial number lower that the one stated here are probe trials 
vr.lickVoltage=[]; % from lick sensor 
vr.roe=[]; % from ROE
vr.timeROE=[]; % from ROE



% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
global lickSensor islick roe timeROE
vr.t=vr.t+1;
vr.lickVoltage(vr.t)=lickSensor;  %state from where we are getting the lick
%%signal remember to update signal file
vr.time(vr.t)=toc;
vr.roe(vr.t)=roe;% Read current count from the quadrature rotary encoder.
vr.timeROE(vr.t)=timeROE; % Time elasped in seconds since Arduino server starts running (double)
vr.lick(vr.t)=islick;
vr.ypos(vr.t)=vr.position(2);
vr.trialNum(vr.t)=vr.trialNum(vr.t-1);
vr.isReward=0;
vr.reward(vr.t)=0;
vr.changeRewLoc(vr.t)=0;

if vr.lickControl==0
    if vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward(vr.trialNum==vr.trialNum(vr.t)))==0
        vr.isReward=1;
    elseif vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward)>0
        a=find(vr.reward,1,'last');
        if vr.trialNum(a)==vr.trialNum(vr.t)
            vr.isReward=0;
        elseif vr.trialNum(vr.t)>2 %probe trials = [ 0 1 2 ].
            vr.isReward=1;
        end
    end
elseif vr.lickControl==1
    if islick && vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward(vr.trialNum==vr.trialNum(vr.t)))==0
        vr.isReward=1;
    elseif islick && vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward)>0
        a=find(vr.reward,1,'last');
        if vr.trialNum(a)==vr.trialNum(vr.t)
            vr.isReward=0;
        else
            vr.isReward=1;
        end
    end
end


if  vr.position(2)>= vr.topTarget
    vr.time2EndCheck=vr.time(vr.t);
end 

if  vr.time(vr.t)-vr.time2EndCheck>=0.2
    vr.worlds{vr.currentWorld}.surface.visible(:) = false;
    vr.position(2) = vr.beginZone;
    vr.position(1) = 0;
    vr.position(4) = 0;
    vr.dp(1:4) = [0 0 0 0];
    vr.sametrial = 0;
    vr.trialNum(vr.t)=vr.trialNum(vr.t)+1;
    vr.time2dark=vr.time(vr.t);
end
    
if vr.position(2)== vr.beginZone && vr.time(vr.t)>=vr.time2dark+(vr.rewarddarkTime+(randi([1,2])+(randi([0,10])/10))) % variable length in dark
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end

if vr.worlds{vr.currentWorld}.surface.visible(:) == false
    vr.dp(1:4) = [0 0 0 0];
end

% key press----------------------------------------------------------------

%200610 EH. Change lickThreshold increments
if double(vr.keyPressed == 52) %ascii code for "4"
    vr.lickThreshold = vr.lickThreshold+0.001;
    disp(vr.lickThreshold)
end

% Change lickThreshold decrements
if double(vr.keyPressed == 53) %ascii code for "5"
    vr.lickThreshold = vr.lickThreshold-0.001;
    disp(vr.lickThreshold)
end

% Decrement length rew zone
if double(vr.keyPressed == 55) && vr.rewardZone>=15 %ascii code for "7"
 vr.rewardZone =  vr.rewardZone - 5;
end

% Increment length rew zone
if double(vr.keyPressed == 55) && vr.rewardZone<vr.topTarget %ascii code for "8"
 vr.rewardZone =  vr.rewardZone + 5;
end

% Lick control
if double(vr.keyPressed == 57) && vr.lickControl==1 %ascii code for "9"
   vr.lickControl=0; 
elseif double(vr.keyPressed == 57) && vr.lickControl==0
    vr.lickControl=1; 
end

% Change reward location
if double(vr.keyPressed) == 45 || double(vr.keyPressed) == 95  %ascii code for "-/_"
    vr.isChange=1;
end

if  vr.isChange==1 && vr.position(2)==vr.beginZone
    if sum(vr.changeRewLoc>=1)<=vr.numMainRewZone-1
        a=[{60:80} {81:120} {121:160}];%defines the regions, here we have three
        indx=vr.i(sum(vr.changeRewLoc>=1)+1); %choose the region
        temp =a{1,indx}(1,randi(size(a{1,indx},2))); % defines the target within region
    else
        temp = randi([60,160]);
    end
    vr.changeRewLoc(vr.t)=temp;
    checkIndx=find(vr.changeRewLoc > 0); % make sure the new rewarded location is set at least 20 cm apart from the previous one
    diff_rewZ=abs(vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))-temp);
    if diff_rewZ < 30
        if sum(vr.changeRewLoc>=1)<=vr.numMainRewZone
            a=[{60:80} {81:120} {121:160}];%defines the regions, here we have three
            indx=vr.i(sum(vr.changeRewLoc>=1)); 
            if vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))>temp %if the difference between the two regions is >30 and temp is smaller than the previous one and we are still in the first three rewaeded zones
                temp=randi([min(a{1,indx}),temp-(diff_rewZ+1)]);
            elseif vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))<temp
                temp=randi([temp+(diff_rewZ+1),max(a{1,indx})]);
            end
        elseif sum(vr.changeRewLoc>=1)>3 %when location is in a random spot, (after three switches of rew zone)
            if vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))>temp
                temp=randi([60,temp-(diff_rewZ+1)]);
            elseif vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))<temp
                temp=randi([temp+(diff_rewZ+1),160]);
            end
        end
    end
  
    vr.changeRewLoc(vr.t)=temp;
    vr.goal = temp; %set the new reward zone
    vr.isChange=0;
    disp('===')
    disp(['Total Trial = ' num2str(vr.trialNum(vr.t))])
    pastRewLoc = vr.changeRewLoc(vr.changeRewLoc>0);
    pastRewLoc = pastRewLoc(length(pastRewLoc-1));
    disp(['Rew Locations = ' num2str(pastRewLoc)])
    disp(['Reward total = ' num2str(sum(vr.reward(pastRewLoc(length(pastRewLoc-1)):vr.t)))])
    vr.trialNum(vr.t)=0;
end

if vr.isReward
    switch vr.reward_multiple
        case 3
            reward_triple(vr,vr.timeSolenoid);
        case 2
            reward_double(vr,vr.timeSolenoid);
        case 1
            reward(vr,vr.timeSolenoid);
    end
end

    if vr.isReward
        vr.reward(vr.t)=vr.isReward; 
    end



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
VR.changeRewLoc=vr.changeRewLoc;
if exist('name_date_vr','var')
    save(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\' name_date_vr '.mat'],'VR')
    load('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date');
    name_date{size(name_date,1)+1,1}=name_date_vr;
    save('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date')
end
plotVR