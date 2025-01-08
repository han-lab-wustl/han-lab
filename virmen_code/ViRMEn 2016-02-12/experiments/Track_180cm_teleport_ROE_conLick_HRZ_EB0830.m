function code = Track_180cm_teleport_ROE_conLick_HRZ_EB0830
% 200610. EH modify to use contact lick sensor instead of optical

% Hidden Reward Zone experiment code - HRZ


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
global scalingFACTOR

% Change parameters here --------------------------------------------------
scalingFACTOR = 1; % Gain in the track. The smaller, the longer the track. 1= usual gain. This value is multiplied by num rpm from ROE
p = 0.4 ;%probability of having probe trial at the beginning
vr.lickThreshold = -0.031; % might need to adjust
vr.goal = 90; % HERE state the automatic reward location
vr.lickControl = 1; % HERE state the if lick is necessary to get reward
vr.rewardZone = 15*scalingFACTOR; % length of reward zone 
vr.numProbes = 3; % number of probe trials (remember for analysis that their index start from 0)
vr.random_initiation = 1; % randomly initiate reward zone
vr.automaticExperimentRewZone = 1; % automatically switch the reward location after randi(22:27) trials
vr.automaticExperimentWorld = 0; % automatically change the current World after n = vr.automaticExperimentWorldNRewZones reward locations.
vr.automaticExperimentWorld1NRewZones = 1; %how many reward zone in the world 1
vr.automaticExperimentWorld2NRewZones = 2; %how many reward zone in the world 2

% Timers - seconds
vr.endTime = 0.2 ;%vr.rewardTime = 1;%time after reaching the end of the track that animal spend in VR
vr.darkTime = 3 ;%vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start, now it goes from 3:5s

% Mouse name and current date ---------------------------------------------
% Pop up window asking to state the animal's name and automatically saving
% today's date

prompt_name = 'Type mouse name ';
mouse_name = inputdlg(prompt_name);
Date = char(today('datetime'));
Date = regexprep(Date,'-','_');
date{1,1} = Date;
vr.name_date_vr = [mouse_name{1,1} '_' date{1,1}];

% Gain setting ------------------------------------------------------------

if scalingFACTOR ~= 1
    prompt_automaticG = ['your scalingFactor is =  ' num2str(scalingFACTOR) '. Wanna keep it? 1-yes, 0-set a new one '];
    answer_automaticG = inputdlg(prompt_automaticG);
    answer_automaticG = str2double(answer_automaticG);
    if answer_automaticG == 0
        prompt_automaticGN = 'Set GAIN to 1 for normal gain, other values to scale it';
        answer_automaticGN = inputdlg(prompt_automaticGN);
        scalingFACTOR = str2double(answer_automaticGN);
    end
end


% Automatic world change setting ------------------------------------------
if vr.automaticExperimentWorld == 1
    prompt_automaticW = 'the atomatic world change is activated, are you okay with that? 1-yes 0-no ';
    answer_automaticW = inputdlg(prompt_automaticW);
    vr.automaticExperimentWorld = str2double(answer_automaticW);
end

% Settings from previous experiments --------------------------------------
vr.timeSolenoid = 140; % in milliseconds
vr.topTarget = 170; % from "linearTrackTwoDirections"
vr.bottomTarget = 10; 
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
vr.RotGainFactor = 2;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1; % default world at start - we only have 2 worlds in this esperiment
vr.numRewards = 0 ; %actual earned rewards
vr.currentGain = 2;%gain world. 1=default, 2=gainFactor
vr.reward_multiple = 1;

% EB variables ------------------------------------------------------------
vr.numMainRewZone = 3; % how many big reward zones in this experiment
vr.isChange = 0 ; % the reward location will change at the end of this trial
vr.sametrial = 0 ;%define if it's new trial
vr.time2EndCheck = 0 ; % value =1 means the timer started
vr.time2End = [] ; % store the time in which the animal reached the end
vr.changeCurrentWorld = 0; % value =1 means the world is about to change
vr.nW1Changes = 1; %how many times the world1 appear. Set to 1 considering the experiment starts from W1.
vr.nW2Changes = 0; %how many times the world2 appear. 
vr.nTrials = randi([22,27]);%random n of trials  
vr.random_time = round(rand*2,1);%random amount of time in the dark

rng('shuffle') % prevent from having the same random numbers each time matlab restart

vr.i=zeros(1,300);
for i=1:3:100
vr.i(1*i:(1*i+2)) = randperm(vr.numMainRewZone); %defines a random order for the three main reward zones.
end

corr = diff(vr.i)==0; %correct consecutive rew zones
while sum(corr)>0
vr.i(corr)=[];
corr = diff(vr.i)==0;
end

if vr.random_initiation
    a=[{60:93} {94:127} {128:161}]; % defines the possible target zone
    i=vr.i(1); % defines the region
    vr.goal(1) =a{1,i}(1,randi(length(a{1,i})));
end

tic % start the stopwatch
vr.t = 1; % index to store variables 

% Data that we are collecting ---------------------------------------------

vr.changeRewLoc=vr.goal; %reward location
vr.time=toc;
vr.lick=[];
vr.ypos=[];
vr.reward=[];
if vr.numProbes>0 && rand < p
    I = (0:vr.numProbes -1) ;
    vr.trialNum = I(randi(vr.numProbes )) ;
else
    vr.trialNum=vr.numProbes ; % trial number lower that the one stated here are probe trials
end
vr.worldCurrent= vr.currentWorld ;
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
global lickSensor islick roe timeROE scalingFACTOR
vr.t=vr.t+1;
vr.lickVoltage(vr.t)=lickSensor;  %state from where we are getting the lick
%%signal remember to update signal file
vr.time(vr.t)=toc;
vr.roe(vr.t)=roe;% Read current count from the quadrature rotary encoder.
vr.timeROE(vr.t)=timeROE; % Time elasped in seconds since Arduino server starts running (double)
vr.lick(vr.t)=islick;
vr.ypos(vr.t)=vr.position(2);
vr.trialNum(vr.t)=vr.trialNum(vr.t-1);
vr.reward(vr.t)=0;
vr.changeRewLoc(vr.t)=0;
vr.pressedKeys(vr.t)=vr.keyPressed;
vr.worldCurrent(vr.t)=vr.currentWorld ;
vr.isReward=0;
vr.changeCurrentWorld=0;
vr.numRewards=sum(vr.reward);

if vr.lickControl==0
    if vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward(vr.trialNum==vr.trialNum(vr.t)))==0 && vr.trialNum(vr.t)>vr.numProbes-1 %probe trials = [ 0 1 2 ].
        vr.isReward=1;
    elseif vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward)>0
        a=find(vr.reward,1,'last');
        if vr.trialNum(a)==vr.trialNum(vr.t)
            vr.isReward=0;
        elseif vr.trialNum(vr.t)>vr.numProbes-1 %probe trials = [ 0 1 2 ].
            vr.isReward=1;
        end
    end
elseif vr.lickControl==1
    if islick && vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward(vr.trialNum==vr.trialNum(vr.t)))==0 && vr.trialNum(vr.t)>vr.numProbes-1 %probe trials = [ 0 1 2 ].
        vr.isReward=1;
    elseif islick && vr.position(2)>(vr.goal-(vr.rewardZone/2)) && vr.position(2)<(vr.goal+(vr.rewardZone/2)) && sum(vr.reward)>0
        a=find(vr.reward,1,'last');
        if vr.trialNum(a)==vr.trialNum(vr.t)
            vr.isReward=0;
        elseif vr.trialNum(vr.t)>vr.numProbes-1 %probe trials = [ 0 1 2 ].
            vr.isReward=1;
        end
    end
end

if vr.isReward
    vr.reward(vr.t)=vr.isReward;
    switch vr.reward_multiple
        case 3
            reward_triple(vr,vr.timeSolenoid);
        case 2
            reward_double(vr,vr.timeSolenoid);
        case 1
            reward(vr,vr.timeSolenoid);
    end
end   

% Key Press section -------------------------------------------------------

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

% Decrement length rew zone
if double(vr.keyPressed == 55) && vr.rewardZone>=15 %ascii code for "7"
 vr.rewardZone =  vr.rewardZone - 5*scalingFACTOR ;
end

% Increment length rew zone
if double(vr.keyPressed == 55) && vr.rewardZone<vr.topTarget %ascii code for "8"
 vr.rewardZone =  vr.rewardZone + 5*scalingFACTOR ;
end

% Lick control
if double(vr.keyPressed == 57) && vr.lickControl==1 %ascii code for "9"
   vr.lickControl=0; 
elseif double(vr.keyPressed == 57) && vr.lickControl==0
    vr.lickControl=1; 
end

% Change World
% if double(vr.keyPressed == 61)%ascii code for "="
%     if vr.currentWorld == 1 
%         vr.changeCurrentWorld = 2; 
%         vr.isChange=1;
%     elseif vr.currentWorld == 2
%         vr.changeCurrentWorld = 1; % default world at start - we only have 2 worlds in this esperiment
%         vr.isChange=1;
%     end
% end  

% Change reward location - note: it will chamge when the animal reach the
% end of the trial, and it's set at least 30 cm apart from the previous one
if double(vr.keyPressed) == 45 || double(vr.keyPressed) == 95  %ascii code for "-/_"
    vr.isChange=1;
end

% End Key Press section ---------------------------------------------------

tempNumSuccTrials = sum(vr.reward(find(vr.changeRewLoc,1,'last'):vr.t)); % count the number of succesful trial in the current reward location

if vr.automaticExperimentRewZone == 1 && tempNumSuccTrials == vr.nTrials
    vr.isChange=1;
    vr.nTrials=randi([22,27]);
end

if  vr.isChange==1 && vr.position(2)==vr.beginZone
    a=[{60:93} {94:127} {128:161}];%defines the regions, here we have three
    indx=vr.i(sum(vr.changeRewLoc>=1)+1); %choose the region
    tempdist = abs(vr.changeRewLoc(find(vr.changeRewLoc,1,'last'))-a{1,indx}); % which locations are 30 cm apart from the previous one
    rewZoneIdx = find (tempdist > 30); % find only the good indexes
    temp = a{1,indx}(rewZoneIdx(randi(length(rewZoneIdx)))); %randomly choose one of them
    vr.goal = temp; %set the new reward zone
    vr.isChange=0;
    disp('===')
    disp(['Total Trial = ' num2str(vr.trialNum(vr.t)+1)])
    pastRewLoc = vr.changeRewLoc(vr.changeRewLoc>0);
    pastRewLoc = pastRewLoc(length(pastRewLoc-1));
    disp(['Rew Locations = ' num2str(pastRewLoc)])
    disp(['Reward total = ' num2str(sum(vr.reward(find(vr.changeRewLoc>0,1,'last'):vr.t)))])
    disp('===')
    vr.trialNum(vr.t)=0;
    vr.changeRewLoc(vr.t)=temp;
end

if  vr.collision && vr.time2EndCheck == 0 %vr.position(2)>= vr.topTarget
    vr.time2End=vr.time(vr.t); % at what time the animal reached the top target
    vr.time2EndCheck = 1 ; % value =1 means the timer started
end 

if vr.automaticExperimentWorld == 1  
    currentNRewLoc=sum(vr.changeRewLoc>0);
    if currentNRewLoc>vr.automaticExperimentWorld1NRewZones*vr.nW1Changes + vr.automaticExperimentWorld2NRewZones*vr.nW2Changes && vr.currentWorld == 1
         vr.changeCurrentWorld = 2; 
    end
    if currentNRewLoc>vr.automaticExperimentWorld1NRewZones*vr.nW1Changes + vr.automaticExperimentWorld2NRewZones*vr.nW2Changes && vr.currentWorld == 2
        vr.changeCurrentWorld = 1; 
    end
end

if  vr.time2EndCheck == 1
    if  (vr.time(vr.t)-vr.time2End)>=vr.endTime % chech the animal stayed at the end for the desired time
        vr.worlds{vr.currentWorld}.surface.visible(:) = false;
        vr.position(2) = vr.beginZone;
        vr.position(1) = 0;
        vr.position(4) = 0;
        vr.dp(1:4) = [0 0 0 0];
        vr.sametrial = 0;
        vr.trialNum(vr.t)=vr.trialNum(vr.t)+1;
        vr.time2dark=vr.time(vr.t);
        vr.time2EndCheck = 0; % stop timer for the end of the track
        vr.random_time=round(rand*2,1); %choose rnd amount of time in the dark
        if  vr.changeCurrentWorld == 1 && vr.trialNum(vr.t)>vr.numProbes-1
            vr.currentWorld = 1;
            vr.nW1Changes = vr.nW1Changes+1;
        elseif vr.changeCurrentWorld == 2 && vr.trialNum(vr.t)>vr.numProbes-1
            vr.currentWorld = 2;
            vr.nW2Changes = vr.nW2Changes+1;
        end
    end
end
if vr.position(2)== vr.beginZone && vr.time(vr.t)>=vr.time2dark+((vr.darkTime-1)+vr.random_time) % variable length in dark
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end

if vr.worlds{vr.currentWorld}.surface.visible(:) == false % prevent the VR from moving when the screen is black
    vr.dp(1:4) = [0 0 0 0];
end







% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)
global scalingFACTOR
    disp('===')
    disp(['Total Trial = ' num2str(vr.trialNum(vr.t)+1)])
    pastRewLoc = vr.changeRewLoc(vr.changeRewLoc>0);
    pastRewLoc = pastRewLoc(length(pastRewLoc));
    disp(['Rew Locations = ' num2str(pastRewLoc)])
    disp(['Reward total = ' num2str(sum(vr.reward(find(vr.changeRewLoc>0,1,'last'):vr.t)))])
    disp('===')
name_date_vr=vr.name_date_vr;
VR.name_date_vr=name_date_vr;
VR.ROE=vr.roe;
VR.lickThreshold = vr.lickThreshold;
VR.reward=vr.reward;
VR.time=vr.time;
VR.lick=vr.lick;
VR.ypos=vr.ypos;
VR.lickVoltage=vr.lickVoltage;
VR.trialNum=vr.trialNum;
VR.timeROE=vr.timeROE;
VR.changeRewLoc=vr.changeRewLoc;
VR.pressedKeys=vr.pressedKeys;
VR.world=vr.worldCurrent;
VR.scalingFACTOR=scalingFACTOR;
if exist('name_date_vr','var')
    save(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\' name_date_vr '.mat'],'VR')
    load('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date');
    name_date{size(name_date,1)+1,1}=name_date_vr;
    save('C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\summary','name_date')
end
plotVR
disp (['first trial number was = ' num2str(vr.trialNum(1))])