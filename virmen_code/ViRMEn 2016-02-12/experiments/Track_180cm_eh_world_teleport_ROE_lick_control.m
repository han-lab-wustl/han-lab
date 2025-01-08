function code = Track_180cm_eh_world_teleport_ROE_lick_control
% last updated 20200203
% - world 1: unidirectional run, reward at end, teleport to beginning

% - licking n times in t secs gives licksuccess
% - gamemode (0/1) decides whether lick outside the reward zone counts
% towards licksuccess
% - reward is given when licksuccess and positionsuccess(now is reaching the end of the track)
% - teleport is delayed: rewardTime states the amount of time after reward for animal to
% explore,rewarddarkTime states the amount of the time in dark before
% teleport

% - world 2: invisible and no reward


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
vr.reward_multiple = 1;
vr.timeSolenoid = 140; %in milliseconds
vr.topTarget = 170;%from "linearTrackTwoDirections"
vr.bottomTarget = 10;%new term
% %vr.endZone = 170;  %original
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
% vr.RotGainFactor= 2;%for changing rot gain in track 2
% vr.RotGainFactorEnd= .75;%for changing rot gain in EZ.75
vr.RotGainFactor= 1;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1; % default world at start
vr.currentGoal = 100;
vr.rewardZone = 100; % length of reward zone after currentGoal
vr.firsttimecheck = 0;

vr.numRewards = 0;%actual earned rewards
vr.water = 0;%keep track of total rewards for water tracking
vr.currentGain = 1;%gain world. 1=default, 2=gainFactor
vr.testRew=0;%track rew num for testing reward amount. see below
vr.rewardTime = 5; %amount of seconds that you stay after receiving a reward
vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start
vr.rewardTimer = 0; %instancing the timer used for rewardtime and punishment time

def_ROE;
vr.licksensor = arduino('COM4','Uno'); %calles the arduino for lick sensor % check device manager for COMx
vr.lickThreshold = 4.97;% might need to adjust
vr.numbertolick = 0; % integer,n licks to be successful, set to 0 to make licksuccess always 1
vr.timetolick = 1; % secs,the first lick and nth lick has to be less than t sec apart, set to 0 to make licksuccess always 0
vr.lick_t = [];
vr.licksuccess = 0; % variable used in conditioning reward acquirance
vr.gamemode = 0; % decide if the lick outside "reward zone" counts

vr.framerateMin = 0.03; %seconds % make it wait until 0.03 if it is faster than that

vr.worlds{2}.surface.visible(:) = false; % I think this is sufficient to make world 2 invisible at all time?

vr.startTime = now;

% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
global islick
if vr.collision % test if the animal is currently in collision
    % reduce the x and y components of displacement
    %vr.dp(1:2) = vr.dp(1:2) * vr.friction;
    %vr.dp(1) = vr.dp(1) * 0.95;
    vr.dp(1) = 0;
    %     vr.dp(4) = vr.dp(4)*0.2;
    %     vr.position(2) = 100;
    %     vr.dp(:) = 0;
end

%from "linearTrackTwoDirections"
%symbolYPosition = 2*(vr.position(2)-vr.trackMinY)/(vr.trackMaxY-vr.trackMinY) - 1;
%vr.plot(1).y = [-1 -1 1 1 -1]*vr.symbolSize + symbolYPosition;

% add lick checking module
% logic: everytime it gives n licks, check the time difference between the nth
% lick and the n-4 th lick to see if it is within t seconds

switch vr.gamemode
    case 1 % only count lick in reward zone
        in_reward_zone = (vr.position(2)>=vr.currentGoal-vr.rewardZone/2) && vr.position(2)<=(vr.currentGoal+vr.rewardZone/2);
        if ~in_reward_zone % variable name in_reward_zone to be defined in the future
            vr.islick = 0;
        end
end

if islick
    vr.lick_t = [vr.lick_t,vr.timeElapsed];
end
if numel(vr.lick_t)>=vr.numbertolick
    if vr.numbertolick == 0
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

vr.positionsuccess = (vr.position(2)>vr.currentGoal)&& (vr.position(2)<vr.currentGoal+vr.rewardZone); % for now we will just have getting to the end as a goal, we can make it to past reward zone too, or set to zero to not use position

if vr.currentWorld == 1
    vr.rewardTimer = vr.rewardTimer + vr.dt;    
    if (vr.positionsuccess && vr.licksuccess == 1) && vr.firsttimecheck == 0  % if you don't want lick control then just set vr.numbertolick = 0 in initiation
        %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
        vr.numRewards = vr.numRewards + 1;
        vr.isReward = 1;
        vr.firsttimecheck= 1;
        vr.rewardTimer = 0;
    elseif (vr.position(2)>vr.topTarget)&& vr.firsttimecheck == 0 %teleport when animal reaches end and not get reward yet too
         vr.firsttimecheck= 1;
        vr.rewardTimer = 0;
    else
        vr.isReward = 0;
    end
    
    if vr.rewardTimer > vr.rewardTime && vr.firsttimecheck == 1 && vr.rewardTimer <= (vr.rewardTime+vr.rewarddarkTime)
        vr.worlds{vr.currentWorld}.surface.visible(:) = false;
        vr.position(2) = vr.beginZone;
        vr.position(1) = 0;
        vr.position(4) = 0;
        vr.dp(1:4) = [0 0 0 0];
    elseif vr.rewardTimer > (vr.rewardTime+vr.rewarddarkTime) && vr.firsttimecheck == 1
        vr.firsttimecheck = 0;
        vr.rewardTimer=0;
        vr.worlds{vr.currentWorld}.surface.visible(:) = true;
    end
elseif vr.currentWorld == 2 % world 2 has no reward, can change to random reward too
    vr.isReward = 0;
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

%teleport world
if double(vr.keyPressed) == 61  %ascii code for "+"
    %         if double(vr.keyPressed) == 43  %ascii code for "+"
    vr.isReward = 1;
    reward(vr,vr.timeSolenoid);
    vr.numRewards = vr.numRewards + 1;
    vr.isReward = 0;
    if vr.currentWorld == 1
        vr.currentWorld = 2; % set the current world
    else
        vr.currentWorld = 1;
    end
    vr.position(2)= vr.beginZone;
%     vr.position(2) = 0; % set the animal’s y position to 0
    if vr.currentGain == 1
        vr.currentGain = 2; % set the current world
    else
        vr.currentGain = 1;
    end
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
%    disp(t1)
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
