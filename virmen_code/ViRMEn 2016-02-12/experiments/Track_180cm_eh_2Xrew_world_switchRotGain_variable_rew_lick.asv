function code = Track_180cm_eh_2Xrew_world_switchRotGain_variable_rew_lick
%send 1/10 view angle V out, -1 to 1 instead of -10 to 10
%attempt to decrease noise into lick channel, which shows big transients
%that very with view angle voltage. 191005 EH
%calls "function vr = initializeDAQ_low_view_V(vr)"
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
vr.reward_multiple2 = 1;
vr.timeSolenoid = 140; %in milliseconds
vr.topTarget = 170;%from "linearTrackTwoDirections"
vr.bottomTarget = 10;%new term
vr.rewardZone = 20; %indicates the size of the reward zone after visual cue
vr.binsize = 2.25; %cm, how big each bin is
vr.lickThreshold = 4.9;% calibrated threshold for what the arduino reads to be a lick
for arbx = 10:floor(180/vr.binsize)-5 %assigns an index for each bin. exclude 0 to binsize and end to avoid impossible reward
    vr.binsIndx(arbx-9) = arbx*vr.binsize;
end
vr.currentMode = 0; %switches between game modes ( auto reward =0 vs lick =1)

vr.punishmentTime = 10; %amount of seconds in the dark after missing a reward lick
vr.rewardTime = 5; %amount of seconds that you stay after receiving a reward
vr.rewarddarkTime = 2; % amount of seconds after vr.rewardtime that you are in the dark before reappearing at the start
% %vr.endZone = 170;  %original
vr.rewardTimer = vr.punishmentTime - vr.rewardTime +vr.rewarddarkTime; %instancing the timer used for rewardtime and punishment time
vr.beginZone = 8.5;
vr.end_gain = 1;
vr.track_gain = 1;
% vr.RotGainFactor= 2;%for changing rot gain in track 2
% vr.RotGainFactorEnd= .75;%for changing rot gain in EZ.75
vr.RotGainFactor= 1;%for changing rot gain in track 2
vr.RotGainFactorEnd= 1;%for changing rot gain in EZ.75
vr.currentWorld = 1;
vr.currentGoal = 50;
%vr.scaling = [eval(vr.exper.variables.scalingGoal) eval(vr.exper.variables.scalingStart)];
%vr.topTarget = (eval(vr.exper.variables.numCylinders)-1)*eval(vr.exper.variables.cylinderSpacing);
%vr.scalingDecay = eval(vr.exper.variables.scalingDecay);
%vr.currentGoal = 1;
vr.numRewards = 0;%actual earned rewards
vr.water = 0;%keep track of total rewards for water tracking
vr.currentGain = 1;%gain world. 1=default, 2=gainFactor
vr.testRew=0;%track rew num for testing reward amount. see below
%text box not working
% % Define a textbox and set its position, size and color
% vr.text(1).position = [-2.5 .5]; % upper-left corner of the screen
% vr.text(1).size = 0.03; % letter size as fraction of the screen
% vr.text(1).color = [1 1 0]; % yellow

% vr.text(2).string = '0';
% vr.text(2).position = [-.14 0];
% vr.text(2).size = .03;
% vr.text(2).color = [1 1 0];
vr.arduino = arduino(); %calles the arduino
vr.firsttimecheck = 2; % stat of end condition: 0 means still running, 1 means received a reward and being teleported back, 2 means missed reward and being punished
%vr.friction = 0.1; % define friction that will reduce velocity by 70% during collisions

%stores the reward vertices
indx = vr.worlds{vr.currentWorld}.objects.indices.RewardCylinders;
vertexFirstLast = vr.worlds{vr.currentWorld}.objects.vertices(indx,:);
vr.rewardcylindx = vertexFirstLast(1):vertexFirstLast(2);


%moves the cylinders to the first reward position
deltarewardcyl = vr.currentGoal - vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx(1));
vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) = vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) + deltarewardcyl;



 
vr.startTime = now;


% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
if vr.currentMode==0
    %original single dir track code
% if (vr.position(2)>vr.endZone)
%     %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
%     vr.isReward = 1;
%     reward(vr,vr.timeSolenoid);
%     %%Teleport added
%     vr.position(2) = vr.beginZone;
%     vr.dp(:) = 0;
if vr.collision % test if the animal is currently in collision
    % reduce the x and y components of displacement
    %vr.dp(1:2) = vr.dp(1:2) * vr.friction;
    %vr.dp(1) = vr.dp(1) * 0.95;
    vr.dp(1) = 0;
%     vr.dp(4) = vr.dp(4)*0.2;
%     vr.position(2) = 100;
%     vr.dp(:) = 0;
end  

currentlick = readVoltage(vr.arduino,'A0');
vr.rewardTimer = vr.rewardTimer + vr.dt;
vr.dp(2) = abs(vr.dp(1)) + abs(vr.dp(2)) + abs(vr.dp(3)) + abs(vr.dp(4)*12.5/pi);
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
if vr.rewardTimer > vr.rewardTime && vr.firsttimecheck == 1 && vr.rewardTimer <= (vr.rewardTime+vr.rewarddarkTime)

     deltarewardcyl = vr.currentGoal - vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx(1));
vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) = vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) + deltarewardcyl;
vr.worlds{vr.currentWorld}.surface.visible(:) = false;
 vr.position(2) = vr.beginZone;
            vr.position(1) = 0;
            vr.position(4) = 0;
            vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
elseif vr.rewardTimer > (vr.rewardTime+vr.rewarddarkTime) && vr.firsttimecheck == 1
     vr.firsttimecheck = 0;
     vr.worlds{vr.currentWorld}.surface.visible(:) = true;
     elseif vr.rewardTimer< (vr.punishmentTime+vr.rewardTime) && vr.firsttimecheck == 2
    deltarewardcyl = vr.currentGoal - vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx(1));
vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) = vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) + deltarewardcyl;
vr.worlds{vr.currentWorld}.surface.visible(:) = false;
 vr.position(2) = vr.beginZone;
            vr.position(1) = 0;
            vr.position(4) = 0;
            vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
elseif vr.rewardTimer > vr.punishmentTime+vr.rewardTime && vr.firsttimecheck ==2
    vr.firsttimecheck = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end
%text box not working
% % On every iteration, update the string to display the time elapsed
% vr.text(1).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];
% vr.text(1).string = num2str(vr.currentGoal);
% vr.text(2).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];


%from "linearTrackTwoDirections"
%symbolYPosition = 2*(vr.position(2)-vr.trackMinY)/(vr.trackMaxY-vr.trackMinY) - 1;
%vr.plot(1).y = [-1 -1 1 1 -1]*vr.symbolSize + symbolYPosition;

if (vr.position(2)>vr.currentGoal && vr.position(2)<(vr.currentGoal+vr.rewardZone)&& vr.rewardTimer > vr.rewardTime)
  vr.firsttimecheck = 1;
    
    switch vr.reward_multiple2
        case 3
            reward_triple(vr,vr.timeSolenoid);
        case 2
            reward_double(vr,vr.timeSolenoid);
        case 1
            reward(vr,vr.timeSolenoid);
    end
     identcurrentGoal = vr.currentGoal;
    vr.rewardTimer = 0;
%   for arbI = 1:10
%   vr.worlds{vr.currentWorld}.surface.colors(4,:) = 1-arbI/10;
%   pause(0.3)
%   end
   
    while (identcurrentGoal == vr.currentGoal) %prevents reward location from being the same
    vr.currentGoal = vr.binsIndx(unidrnd(length(vr.binsIndx)));
    end
   % for arbJ = 1:10
%      vr.worlds{vr.currentWorld}.surface.colors(4,:) = arbI/10;
%   pause(0.3)
% end
        %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
    vr.numRewards = vr.numRewards + 1; 
    vr.isReward = 1;
    
    else
        vr.isReward = 0;       
end

if vr.position(2) > vr.topTarget % test if the animal is at the end of the track (y > 200)
    %vr.dp(4) = vr.dp(4)*3.5; % set the animal’s y position to 0
    vr.dp(4) = vr.dp(4)*vr.end_gain; % set the animal’s y position to 0
    %vr.dp(:) = 0; % prevent any additional movement during teleportation
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactorEnd;
    end
end

if vr.position(2) < vr.bottomTarget % test if the animal is at the end of the track (y > 200)
       %vr.dp(4) = vr.dp(4)*3.5; % set the animal’s y position to 0
    vr.dp(4) = vr.dp(4)*vr.end_gain; % set the animal’s y position to 0
    %vr.dp(:) = 0; % prevent any additional movement during teleportation
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactorEnd;
    end
end
if vr.position(2) > vr.bottomTarget  && vr.position(2)<vr.topTarget
    vr.dp(4) = vr.dp(4)*vr.track_gain;
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactor;
    end
    
end
if vr.position(2) > vr.topTarget + vr.beginZone
    vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
end
%     if ~isnan(vr.keyPressed)
%         vr.keyPressed
%     end
    %teleport world
    if double(vr.keyPressed) == 49  %ascii code for "2"
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
    
    if double(vr.keyPressed) == 61  %ascii code for "+"
%         if double(vr.keyPressed) == 43  %ascii code for "+"
        vr.currentMode = 1;
        vr.rewardTimer = vr.rewardTime+0.1;
        vr.firsttimecheck = 1;
    end
elseif vr.currentMode == 1
        %original single dir track code
% if (vr.position(2)>vr.endZone)
%     %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
%     vr.isReward = 1;
%     reward(vr,vr.timeSolenoid);
%     %%Teleport added
%     vr.position(2) = vr.beginZone;
%     vr.dp(:) = 0;
if vr.collision % test if the animal is currently in collision
    % reduce the x and y components of displacement
    %vr.dp(1:2) = vr.dp(1:2) * vr.friction;
    %vr.dp(1) = vr.dp(1) * 0.95;
    vr.dp(1) = 0;
%     vr.dp(4) = vr.dp(4)*0.2;
%     vr.position(2) = 100;
%     vr.dp(:) = 0;
end  

currentlick = readVoltage(vr.arduino,'A0');
vr.rewardTimer = vr.rewardTimer + vr.dt;
vr.dp(2) = abs(vr.dp(1)) + abs(vr.dp(2)) + abs(vr.dp(3)) + abs(vr.dp(4)*12.5/pi);
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;

%teleport conditions
if vr.rewardTimer > vr.rewardTime && vr.firsttimecheck == 1 && vr.rewardTimer <= (vr.rewardTime+vr.rewarddarkTime)

     deltarewardcyl = vr.currentGoal - vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx(1));
vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) = vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) + deltarewardcyl;
vr.worlds{vr.currentWorld}.surface.visible(:) = false;
 vr.position(2) = vr.beginZone;
            vr.position(1) = 0;
            vr.position(4) = 0;
            vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
elseif vr.rewardTimer > (vr.rewardTime+vr.rewarddarkTime) && vr.firsttimecheck == 1
     vr.firsttimecheck = 0;
     vr.worlds{vr.currentWorld}.surface.visible(:) = true;
elseif vr.rewardTimer< (vr.punishmentTime+vr.rewardTime) && vr.firsttimecheck == 2
    deltarewardcyl = vr.currentGoal - vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx(1));
vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) = vr.worlds{vr.currentWorld}.surface.vertices(2,vr.rewardcylindx) + deltarewardcyl;
vr.worlds{vr.currentWorld}.surface.visible(:) = false;
 vr.position(2) = vr.beginZone;
            vr.position(1) = 0;
            vr.position(4) = 0;
            vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
elseif vr.rewardTimer > vr.punishmentTime+vr.rewardTime && vr.firsttimecheck ==2
    vr.firsttimecheck = 0;
    vr.worlds{vr.currentWorld}.surface.visible(:) = true;
end
%text box not working
% % On every iteration, update the string to display the time elapsed
% vr.text(1).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];
% vr.text(1).string = num2str(vr.currentGoal);
% vr.text(2).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];


%from "linearTrackTwoDirections"
%symbolYPosition = 2*(vr.position(2)-vr.trackMinY)/(vr.trackMaxY-vr.trackMinY) - 1;
%vr.plot(1).y = [-1 -1 1 1 -1]*vr.symbolSize + symbolYPosition;

if (vr.position(2)>vr.currentGoal && vr.position(2)<(vr.currentGoal+vr.rewardZone)&& vr.rewardTimer > vr.rewardTime && currentlick >vr.lickThreshold)
  vr.firsttimecheck = 1;
    
    switch vr.reward_multiple2
        case 3
            reward_triple(vr,vr.timeSolenoid);
        case 2
            reward_double(vr,vr.timeSolenoid);
        case 1
            reward(vr,vr.timeSolenoid);
    end
     identcurrentGoal = vr.currentGoal;
    vr.rewardTimer = 0;
%   for arbI = 1:10
%   vr.worlds{vr.currentWorld}.surface.colors(4,:) = 1-arbI/10;
%   pause(0.3)
%   end
   
    while (identcurrentGoal == vr.currentGoal) %prevents reward location from being the same
    vr.currentGoal = vr.binsIndx(unidrnd(length(vr.binsIndx)));
    end
   % for arbJ = 1:10
%      vr.worlds{vr.currentWorld}.surface.colors(4,:) = arbI/10;
%   pause(0.3)
% end
        %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
    vr.numRewards = vr.numRewards + 1; 
    vr.isReward = 1;
    
    else
        vr.isReward = 0;       
end

if vr.position(2) > vr.topTarget && vr.firsttimecheck == 0
    vr.firsttimecheck = 2;
    vr.rewardTimer = vr.rewardTime + 0.1;
end

if vr.position(2) > vr.topTarget + vr.beginZone
    vr.dp(2) = 0;
vr.dp(1) = 0;
vr.dp(3) = 0;
vr.dp(4) = 0;
end


if vr.position(2) > vr.topTarget % test if the animal is at the end of the track (y > 200)
    %vr.dp(4) = vr.dp(4)*3.5; % set the animal’s y position to 0
    vr.dp(4) = vr.dp(4)*vr.end_gain; % set the animal’s y position to 0
    %vr.dp(:) = 0; % prevent any additional movement during teleportation
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactorEnd;
    end
end

if vr.position(2) < vr.bottomTarget % test if the animal is at the end of the track (y > 200)
       %vr.dp(4) = vr.dp(4)*3.5; % set the animal’s y position to 0
    vr.dp(4) = vr.dp(4)*vr.end_gain; % set the animal’s y position to 0
    %vr.dp(:) = 0; % prevent any additional movement during teleportation
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactorEnd;
    end
end
if vr.position(2) > vr.bottomTarget  && vr.position(2)<vr.topTarget
    vr.dp(4) = vr.dp(4)*vr.track_gain;
    if vr.currentGain == 2  %if in world 2, modify rot gain in track
        vr.dp(4) = vr.dp(4)*vr.RotGainFactor;
    end
    
end
%     if ~isnan(vr.keyPressed)
%         vr.keyPressed
%     end
    %teleport world
    if double(vr.keyPressed) == 49  %ascii code for "2"
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
    
    if double(vr.keyPressed) == 61  %ascii code for "+"
%         if double(vr.keyPressed) == 43  %ascii code for "+"
        vr.currentMode = 0;
        vr.rewardTimer = vr.rewardTime+0.1;
        vr.firsttimecheck = 1;
    end
end
    

    
    
    
    
    


% if vr.isReward
%     vr.numRewards = vr.numRewards + 1;  
%     
% end






% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)

