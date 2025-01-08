function code = Track_180cm_eh_world_teleport_ROE_lick_control2
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
vr = initializeDAQ(vr);
try 
    load('C:\Users\imaging_VR\Documents\MATLAB\lickThreshold.mat','lickThreshold');
    vr.lickThreshold = lickThreshold;
catch
    vr.lickThreshold = 4.9;% might need to adjust, unfortunately the clampex recording is not reliable, use licksensor_toy
end
vr.gamemode = 1; % decide if the lick outside "reward zone" counts. "0" means automatic reward in zone
vr.random_initiation = 0; % randomly initiate reward for world 2. "0" uses set in next line. "1" random position.
vr.allGoals = [100 50]; %world 1 world 2
if vr.random_initiation
    vr.allGoals(2) = randi([50,150]);
end
vr.reward_multiple = 1;
vr.timeSolenoid = 140; %in milliseconds
vr.topTarget = 170;%from "linearTrackTwoDirections"
vr.bottomTarget = 10;%new term
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
vr.RotGainFactor= 1;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1; % default world at start
% vr.currentGoal = 100;
vr.rewardZone = 15; % length of reward zone after currentGoal EB:before was set as 10
vr.firsttimecheck = 0;% initialize
vr.endtimecheck =0;% reach the end without successful lick/pass target
vr.totaltrialcount =0; % initialize
vr.numRewards = 0;%actual earned rewards
vr.currentGain = 1;%gain world. 1=default, 2=gainFactor
vr.rewardTime = 5;
vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start
vr.rewardTimer = 0; %instancing the timer used for rewardtime and punishment time
vr.accuracy =0;
vr.lickoutsidecount = 0;
vr.lickinsidecount = 0;
vr.currentGoal = vr.allGoals(vr.currentWorld);

def_ROE;
vr.licksensor = arduino('COM4','Uno'); %calles the arduino for lick sensor % check device manager for COMx
vr.numbertolick = 1; % integer,n licks to be successful, set to 0 to make licksuccess always 1
vr.timetolick = 1; % secs,the first lick and nth lick has to be less than t sec apart, set to 0 to make licksuccess always 0
vr.lick_t = [];
vr.trialtoaccuracycheckpoint = 5; % run for 5 full trials (either success or get to the end)
% vr.framerateMin = 0.035; %seconds % make it wait until 0.03 if it is faster than that

vr.startTime = now;

% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
global islick
% add lick checking module
% logic: everytime it gives n licks, check the time difference between the nth
% lick and the n-4 th lick to see if it is within t seconds

switch vr.gamemode
    case 1 % has to lick in the reward zone
        if islick
            sound(1)
        end
        vr.licksuccess = 0;
        vr.positionsuccess = 1; % don't even check current position
        in_reward_zone = (vr.position(2)>=(vr.currentGoal-vr.rewardZone/2)) && (vr.position(2)<=(vr.currentGoal+vr.rewardZone/2));
        if ~in_reward_zone && islick == 1 % variable name in_reward_zone to be defined in the future
            islick = 0;
            vr.lickoutsidecount = vr.lickoutsidecount+1;
        end
        if islick % carried from movement function
            vr.lick_t = [vr.lick_t,vr.timeElapsed];
            vr.lickinsidecount = vr.lickinsidecount+1;
        end

        if numel(vr.lick_t)>=vr.numbertolick
            if vr.numbertolick ==0
                vr.licksuccess = 1;
                vr.lick_t = [];
            elseif vr.lick_t(end)- vr.lick_t(end-vr.numbertolick+1)<= vr.timetolick
                vr.licksuccess = 1;
                vr.lick_t = [];
            else
                idx = vr.lick_t >= vr.lick_t(end)-vr.timetolick;
                vr.lick_t = vr.lick_t(idx);
            end
        end
    case 0 % has to go pass the target point
        vr.licksuccess = 1; % don't even check lick if gamemode = 0
        vr.positionsuccess = (vr.position(2)>vr.currentGoal);
end

vr.rewardTimer = vr.rewardTimer + vr.dt;
if (vr.positionsuccess && vr.licksuccess == 1) && vr.firsttimecheck == 0  % if you don't want lick control then just set vr.numbertolick = 0 in initiation
    vr.numRewards = vr.numRewards + 1;
    vr.isReward = 1;
    vr.firsttimecheck= 1;
    vr.rewardTimer = 0;
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

% shrinking
if vr.totaltrialcount> vr.trialtoaccuracycheckpoint && vr.gamemode == 1 
    % check accuracy to adjust reward zone size
    if vr.accuracy<50 && vr.rewardZone<180 % don't make it too large
        vr.rewardZone = vr.rewardZone+5;
        disp('===')
        fprintf('Reward Zone = %i units \n',vr.rewardZone)
        fprintf('Total Trial = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.accuracy)
        vr.trialtoaccuracycheckpoint = vr.totaltrialcount+10;
    elseif vr.accuracy>80 && vr.rewardZone>10
        vr.rewardZone = vr.rewardZone-5;
        disp('===')
        fprintf('Reward Zone = %i units \n',vr.rewardZone)
        fprintf('Total Trial = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.accuracy)
        vr.trialtoaccuracycheckpoint = vr.totaltrialcount+10;
    end
end

if vr.rewardTimer > vr.rewardTime && (vr.endtimecheck == 1 || vr.firsttimecheck == 1) && vr.rewardTimer <= (vr.rewardTime+vr.rewarddarkTime)
    vr.worlds{vr.currentWorld}.surface.visible(:) = false;
    vr.position(2) = vr.beginZone;
    vr.position(1) = 0;
    vr.position(4) = 0;
    vr.dp(1:4) = [0 0 0 0];
elseif vr.rewardTimer > (vr.rewardTime+vr.rewarddarkTime) && (vr.endtimecheck == 1 || vr.firsttimecheck == 1)
    vr.firsttimecheck = 0;
    vr.endtimecheck = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end


% key press
if double(vr.keyPressed == 52) %ascii code for "4"
    vr.lickThreshold = vr.lickThreshold+0.01;
       disp(vr.lickThreshold)
end
if double(vr.keyPressed == 53) %ascii code for "5"
    vr.lickThreshold = vr.lickThreshold-0.01;
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
    reward(vr,vr.timeSolenoid);
    vr.numRewards = vr.numRewards + 1;
    vr.reward_multiple = 2;
end

if double(vr.keyPressed) == 51  %ascii code for "3"
    vr.isReward = 1;
    reward(vr,vr.timeSolenoid);
    vr.numRewards = vr.numRewards + 1;
    vr.reward_multiple = 3;
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
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.2f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
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
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.2f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
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
    vr.isReward = 1;
    vr.position(2)= vr.beginZone;
    disp('===')
    fprintf('Total Trial = %i, rewards = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.numRewards,vr.accuracy)
    fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.2f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
    fprintf('Ratio of licking in reward zone to outside reward zone = %2.2f\n',vr.lickinsidecount/vr.lickoutsidecount)
    disp(['Ran ' num2str(vr.iterations-1) ' iterations in ' num2str(vr.timeElapsed,4) ...
    ' s (' num2str(vr.timeElapsed*1000/(vr.iterations-1),3) ' ms/frame refresh time).']);
    temp = randi([50,150]);
    while abs(vr.currentGoal-temp)<10 % make sure it is not too close
        temp = randi([50,150]);
    end
    vr.currentGoal = temp;
    vr.accuracy =0;
    vr.lickoutsidecount = 0;
    vr.lickinsidecount = 0;
    vr.totaltrialcount = 0;
    vr.numRewards = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
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
    %     play_mario_coin()
end

% prevention of too fast update causing spiky waveforms in ROE
% if vr.iterations == 1
%    t1 = now;
% else
%     t2 = now;
%     deltat = (t2-t1)*3600*24;
%     while deltat < vr.framerateMin
%         t2 = now;
%         deltat = (t2-t1)*3600*24;
%     end
%     t1 = t2;
% end

% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)
disp('===')
fprintf('Total Trial = %i, rewards = %i, accuracy = %2.1f %% \n',vr.totaltrialcount,vr.numRewards,vr.accuracy)
fprintf('Reward Zone = %i units, goal location = %2.2f, lick threshold = %2.2f \n',vr.rewardZone,vr.currentGoal,vr.lickThreshold)
fprintf('Ratio of licking in reward zone to outside reward zone = %2.2f\n',vr.lickinsidecount/vr.lickoutsidecount)
lickThreshold = vr.lickThreshold;
save('C:\Users\imaging_VR\Documents\MATLAB\lickThreshold.mat','lickThreshold')