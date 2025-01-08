function code = Track_180cm_teleport_ROE_conLick_tillTheEnd_phase1Training
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
% vr = initializeDAQ(vr);
vr = initializeDAQ_low_view_V_conLick_ROE(vr); %200610 EH
% try %200610 EH
%     load('C:\Users\imaging_VR\Documents\MATLAB\lickThreshold.mat','lickThreshold');
%     vr.lickThreshold = lickThreshold;
% catch
vr.lickThreshold = -0.029;% might need to adjust, unfortunately the clampex recording is not reliable, use licksensor_toy
% end
prompt_name='type mouse name ';
mouse_name=inputdlg(prompt_name);
%prompt_date='type todays date in the following format: YY/MM/DD ';
%date=inputdlg(prompt_date);
Date=char(today('datetime'));
expression='-';
replace='_';
Date=regexprep(Date,expression,replace);
date{1,1}=Date;
vr.name_date_vr=[mouse_name{1,1} '_' date{1,1}];
vr.gamemode = 1; % decide if the lick outside "reward zone" counts % =1 EB 0630
vr.rewardZone = 15; % length of reward zone after currentGoal
vr.random_initiation = 1; % randomly initiate reward for world 2 %=1 EB 0630
vr.allGoals =15:15:180; %set default goals [world 1 world 2].
vr.t=1;
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
vr.rewarddarkTime = 5; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start
vr.sametrial = 0;%define if it's new trial
vr.rewardTimer = 0; 
vr.reward=0;
vr.trialNum=0; 
vr.lickVoltage=[];
vr.reward_multiple = 1;
vr.positionsuccess = 1;
vr.licksuccess = 1; 
vr.endtimecheck =0;% reach the end without successful lick/pass target
tic
vr.time(vr.t)=toc;
vr.roe=[];
vr.timeROE=[];
vr.isChange=0; %test if a change in the reward location is occurring during this trial
vr.lick=[];
vr.ypos=[];
vr.rewZone=1;
vr.worlds{vr.currentWorld}.surface.visible(:) = true; %false;
vr.firsttimecheck = 0;% initialize
def_ROE;
%200610 EH
%vr.licksensor = readVoltage(arduino('COM3','Uno')); %calles the arduino for lick sensor % check device manager for COMx
vr.numbertolick = 1; % integer,n licks to be successful, set to 0 to make licksuccess always 1
vr.timetolick = 1; % secs,the first lick and nth lick has to be less than t sec apart, set to 0 to make licksuccess always 0
%don't think vr.timeElapsed is defined in function
vr.lick_t = [];%vr.timeElapsed for each lick. reset to [] at  vr.timeElapsed not defined!
% vr.framerateMin = 0.035; %seconds % make it wait until 0.03 if it is faster than that
vr.startTime = vr.time;

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
rewZone=abs(vr.allGoals-vr.position(2));
vr.rewZone(vr.t)=find(rewZone==min(rewZone)); %vector that store the animal position relatively to all the reward zones

if islick && vr.position(2)>10 && sum(vr.reward)==0
    vr.isReward=1;
elseif islick && vr.position(2)>10 && sum(vr.reward)>0
    a=find(vr.reward,1,'last');
    if vr.rewZone(a)==vr.rewZone(vr.t)
        vr.isReward=0;
    elseif vr.position(2)>vr.ypos(a)
        vr.isReward=1;
    end
end

vr.rewardTimer = vr.rewardTimer + vr.dt; %time between rewards

if (vr.positionsuccess && vr.licksuccess == 1) && (vr.sametrial == 0 && vr.firsttimecheck == 0 &&  vr.trialNum(vr.t)>2) % if you don't want lick control then just set vr.numbertolick = 0 in initiation
    vr.numRewards = vr.numRewards + 1;
    vr.isReward = 1;
    vr.firsttimecheck = 1;
    vr.rewardTimer = 0;%reset at reward
    vr.sametrial = 1; %defines that is still the same trial
elseif (vr.position(2)>vr.topTarget)&& vr.firsttimecheck == 0 && vr.endtimecheck==0%teleport when animal reaches end and not get reward yet too
    vr.endtimecheck= 1;
    vr.isend =1;
    vr.rewardTimer = 0;
else
    vr.isReward = 0;
    vr.isend = 0;
end

if vr.isReward == 1 || vr.isend == 1
    vr.totaltrialcount = vr.totaltrialcount+1;
    vr.accuracy = vr.numRewards/vr.totaltrialcount*100;
end

% if vr.isend == 1 % reset the trial
%     vr.sametrial = 0;
% end

if vr.rewardTimer > vr.rewardTime && (vr.endtimecheck == 1) && vr.rewardTimer <= (vr.rewardTime+vr.rewarddarkTime) && (vr.position(2)>= vr.topTarget)
    vr.worlds{vr.currentWorld}.surface.visible(:) = false;
    vr.position(2) = vr.beginZone;
    vr.position(1) = 0;
    vr.position(4) = 0;
    vr.dp(1:4) = [0 0 0 0];
    vr.sametrial = 0;
    vr.trialNum(vr.t)=vr.trialNum(vr.t)+1;
elseif vr.rewardTimer > (vr.rewardTime+vr.rewarddarkTime) && (vr.endtimecheck == 1 || vr.firsttimecheck == 1)
    vr.firsttimecheck = 0;
    vr.endtimecheck = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end

if vr.worlds{vr.currentWorld}.surface.visible(:) == false
    vr.dp(1:4) = [0 0 0 0];
end
% key press
%200610 EH. Change lickThreshold increments
if double(vr.keyPressed == 52) %ascii code for "4"
    vr.lickThreshold = vr.lickThreshold+0.001;
    disp(vr.lickThreshold)
end
if double(vr.keyPressed == 53) %ascii code for "5"
    vr.lickThreshold = vr.lickThreshold-0.001;
    disp(vr.lickThreshold)
end
if double(vr.keyPressed) == 49  %ascii code for "1"
    vr.isReward = 1;
    reward(vr,vr.timeSolenoid);
    vr.numRewards = vr.numRewards + 1;
    vr.reward_multiple = 1;
end
if double(vr.keyPressed) == 50  %ascii code for "2"
    vr.isReward = 1;
    if vr.rewardZone>10
        vr.rewardZone=vr.rewardZone-10;
    end
    %     vr.isReward = 1;
    %     reward(vr,vr.timeSolenoid);
    %     vr.numRewards = vr.numRewards + 1;
    %     vr.reward_multiple = 2;
end

if double(vr.keyPressed) == 51  %ascii code for "3"
    vr.isReward = 1;
    if vr.rewardZone<160
        vr.rewardZone=vr.rewardZone+10;
    end
    %     vr.isReward = 1;
    %     reward(vr,vr.timeSolenoid);
    %     vr.numRewards = vr.numRewards + 1;
    %     vr.reward_multiple = 3;
end

if double(vr.keyPressed) == 39 || double(vr.keyPressed) == 27%ascii code for '`' or 'Esc'
    vr.worlds{vr.currentWorld}.surface.visible(:) = false;
end
% teleport world
% if double(vr.keyPressed) == 61  %ascii code for "+"
%     %         if double(vr.keyPressed) == 43  %ascii code for "+"
%     vr.isReward = 1;
%     reward(vr,vr.timeSolenoid);
%     vr.numRewards = vr.numRewards + 1;
%     vr.isReward = 0;
%     if vr.currentWorld == 1
%         vr.currentWorld = 2; % set the current world
%     else
%         vr.currentWorld = 1;
%     end
%     vr.position(2)= vr.beginZone;
% %     vr.position(2) = 0; % set the animal’s y position to 0
%     if vr.currentGain == 1
%         vr.currentGain = 2; % set the current world
%     else
%         vr.currentGain = 1;
%     end
% end

%change gamemode (or make two worlds the same but gamemode differ? adapt from the world changing code)
if double(vr.keyPressed) == 48  %ascii code for "0"
    %         if double(vr.keyPressed) == 43  %ascii code for "+"
    vr.isReward = 1;
    if vr.gamemode == 1
        vr.gamemode = 0; % set the current world
    else
        vr.gamemode = 1;
    end
    vr.position(2)= vr.beginZone;
    fprintf('Total Trial = %i, rewards = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.numRewards,vr.accuracy)
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.3f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
    fprintf('Ratio of licking in reward zone to outside reward zone = %2.2f\n',vr.lickinsidecount/vr.lickoutsidecount)
    disp(['Ran ' num2str(vr.iterations-1) ' iterations in ' num2str(vr.timeElapsed,4) ...
        ' s (' num2str(vr.timeElapsed*1000/(vr.iterations-1),3) ' ms/frame refresh time).']);
    
    vr.accuracy =0;
    vr.lickoutsidecount = 0;
    vr.lickinsidecount = 0;
    vr.totaltrialcount = 0;
    vr.numRewards = 0;
end

% change world
if double(vr.keyPressed) == 61  %ascii code for "="
    %         if double(vr.keyPressed) == 43  %ascii code for "+"
    vr.isReward = 1;
    if vr.currentWorld == 1
        vr.currentWorld = 2; % set the current world
    else
        vr.currentWorld = 1;
    end
    vr.position(2)= vr.beginZone;
    disp('===')
    fprintf('Total Trial = %i, rewards = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.numRewards,vr.accuracy)
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.3f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
    fprintf('Ratio of licking in reward zone to outside reward zone = %2.2f\n',vr.lickinsidecount/vr.lickoutsidecount)
    disp(['Ran ' num2str(vr.iterations-1) ' iterations in ' num2str(vr.timeElapsed,4) ...
        ' s (' num2str(vr.timeElapsed*1000/(vr.iterations-1),3) ' ms/frame refresh time).']);
    vr.currentGoal = vr.allGoals(vr.currentWorld);
    vr.accuracy =0;
    vr.lickoutsidecount = 0;
    vr.lickinsidecount = 0;
    vr.totaltrialcount = 0;
    vr.numRewards = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end

% change reward location
if double(vr.keyPressed) == 45 || double(vr.keyPressed) == 95  %ascii code for "-/_"
    vr.isChange=1;
end
if  vr.isChange==1 && vr.position(2)==vr.beginZone
    vr.trialNum(vr.t)=0;
    disp('===')
    fprintf('Total Trial = %i, rewards = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.numRewards,vr.accuracy)
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.3f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
    fprintf('Ratio of licking in reward zone to outside reward zone = %2.2f\n',vr.lickinsidecount/vr.lickoutsidecount)
    disp(['Ran ' num2str(vr.iterations-1) ' iterations in ' num2str(vr.timeElapsed,4) ...
        ' s (' num2str(vr.timeElapsed*1000/(vr.iterations-1),3) ' ms/frame refresh time).']);
    if sum(vr.changeRewLoc>=1)<vr.numMainRewZone
        a=[{40:80} {81:120} {121:160}];%defines the regions, here we have three
        indx=vr.i(sum(vr.changeRewLoc>=1)+1); %choose the region
        temp =a{1,indx}(1,randi(size(a{1,indx},2))); % defines the target within region
    else
        temp = randi([40,160]);
    end
    checkIndx=find(vr.changeRewLoc > 0); % make sure the new rewarded location is setted at least 20 cm apart from the previous one
    diff_rewZ=abs(vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))-temp);
    if diff_rewZ < 20
        if sum(vr.changeRewLoc>=1)<=3
            if vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))>temp %if the difference between the two regions is >20 and temp is smalles than the previous one and we are still in the first three rewaeded zones
                temp=randi([min(a{1,indx}),temp-(diff_rewZ+1)]);
            elseif vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))<temp
                temp=randi([temp+(diff_rewZ+1),max(a{1,indx})]);
            end
        elseif sum(vr.changeRewLoc>=1)>=3 %when location is in a random spot, (after three switches of rew zone)
            if vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))>temp
                temp=randi([40,temp-(diff_rewZ+1)]);
            elseif vr.changeRewLoc(1,checkIndx(1,sum(vr.changeRewLoc>=1)))<temp
                temp=randi([temp+(diff_rewZ+1),160]);
            end
        end
    end

    
    vr.changeRewLoc(vr.t)=temp;
    vr.currentGoal = temp; %set the new reward zone
    vr.accuracy =0; %set to zero the counters
    vr.lickoutsidecount = 0;
    vr.lickinsidecount = 0;
    vr.totaltrialcount = 0;
    vr.numRewards = 0;
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
    
    if vr.isReward==1
        vr.reward(vr.t)=vr.isReward;
    else
        vr.reward(vr.t)=0;
    end
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
VR.rewZone=vr.rewZone;
if exist('name_date_vr','var')
    save(['C:\Users\imaging_VR\Documents\MATLAB\tillTheEnd_variables\VR_data\phase1\' name_date_vr '.mat'],'VR')

end