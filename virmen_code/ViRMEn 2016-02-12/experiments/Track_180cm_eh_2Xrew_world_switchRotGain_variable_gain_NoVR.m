function code = Track_180cm_eh_2Xrew_world_switchRotGain_variable_gain_NoVR
%world 2 invisible, no rewards
%switches rotation gain in track in new world
%track gain can by modified during run
%"3" = 0.3 gain, "0"= 1. "8" = 0.8. "\" = 1.5. no rew at switch
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
% vr.reward_multiple = 2;

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
vr.currentWorld = 1;
vr.currentGoal = 0;
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

%vr.friction = 0.1; % define friction that will reduce velocity by 70% during collisions

vr.startTime = now;

% --8- RUNTIME code: executes on every iteration of the ViRMEn engine.
function vr = runtimeCodeFun(vr)
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

%     switch vr.reward_multiple
%         case 3
%             reward_triple(vr,vr.timeSolenoid);
%         case 2
%             reward_double(vr,vr.timeSolenoid);
%             
%     end




%text box not working
% % On every iteration, update the string to display the time elapsed
% vr.text(1).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];
% vr.text(1).string = num2str(vr.currentGoal);
% vr.text(2).string = ['TIME ' datestr(now-vr.startTime,'MM.SS')];


%from "linearTrackTwoDirections"
%symbolYPosition = 2*(vr.position(2)-vr.trackMinY)/(vr.trackMaxY-vr.trackMinY) - 1;
%vr.plot(1).y = [-1 -1 1 1 -1]*vr.symbolSize + symbolYPosition;

if (vr.currentGoal==1 && vr.position(2)>vr.topTarget) || (vr.currentGoal == 0 && vr.position(2)<vr.bottomTarget)
    if vr.currentWorld == 1; 
    vr.currentGoal = 1-vr.currentGoal;
        %vr.scaling(2) = vr.scaling(1)+(vr.scaling(2)-vr.scaling(1))*vr.scalingDecay;
    vr.numRewards = vr.numRewards + 1; 
    vr.isReward = 1;
    reward_double(vr,vr.timeSolenoid);
    end
%     switch vr.reward_multiple
%         case 3
%             reward_triple(vr,vr.timeSolenoid);
%         case 2
%             reward_double(vr,vr.timeSolenoid);
%     end
%     if vr.currentWorld == 2
%         reward_triple(vr,vr.timeSolenoid);
%     else
%         reward_double(vr,vr.timeSolenoid);
%     end
% reward(vr,vr.timeSolenoid);
    
    
%below for testing water reward amount
%keeps dispensing water rewards in top zone, 1Hz.
%turn off reward code above
% if (vr.position(2)>vr.topTarget)&&vr.testRew < 101;
% vr.isReward = 1;
%     reward(vr,vr.timeSolenoid);
%     vr.numRewards = vr.numRewards + 1;
%     vr.testRew = vr.testRew + 1;
%     pause(1);

         %%Teleport added
%      vr.position(2) = vr.beginZone;
%      vr.dp(:) = 0;
    
    %original again
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
%     if ~isnan(vr.keyPressed)
%         vr.keyPressed
%     end
    %teleport world
      
    
      
    if double(vr.keyPressed) == 49  %ascii code for "1"
%         vr.isReward = 1;
%         reward(vr,vr.timeSolenoid);
        vr.track_gain = 0.1;
    end   
    if double(vr.keyPressed) == 50  %ascii code for "2"
%         vr.isReward = 1;
        vr.track_gain = 0.2;
    end    
    
    if double(vr.keyPressed) == 51  %ascii code for "3"
%         vr.isReward = 1;
        vr.track_gain = 0.3;
    end    
    if double(vr.keyPressed) == 52  %ascii code for "4"
%         vr.isReward = 1;
        vr.track_gain = 0.4;
    end 
    if double(vr.keyPressed) == 53  %ascii code for "5"
%         vr.isReward = 1;
        vr.track_gain = 0.5;
    end  
    if double(vr.keyPressed) == 54  %ascii code for "6"
%         vr.isReward = 1;
        vr.track_gain = 0.6;
    end
    if double(vr.keyPressed) == 55  %ascii code for "7"
%         vr.isReward = 1;
%         reward(vr,vr.timeSolenoid);
        vr.track_gain = 0.7;
    end     
    if double(vr.keyPressed) == 56  %ascii code for "8"
%         vr.isReward = 1;
        vr.track_gain = 0.8;
    end  
    if double(vr.keyPressed) == 57  %ascii code for "9"
%         vr.isReward = 1;
        vr.track_gain = 0.9;
    end  
    if double(vr.keyPressed) == 48  %ascii code for "0"
%         vr.isReward = 1;
        vr.track_gain = 1;
    end  
    if double(vr.keyPressed) == 92  %ascii code for "\"
%         vr.isReward = 1;
        vr.track_gain = 1.5;
    end      
    if double(vr.keyPressed) == 61  %ascii code for "+"
%         if double(vr.keyPressed) == 43  %ascii code for "+"
            vr.isReward = 1;
            reward(vr,vr.timeSolenoid);
            vr.numRewards = vr.numRewards + 1;
        if vr.currentWorld == 1; 
            vr.currentWorld = 2; % set the current world
            vr.worlds{vr.currentWorld}.surface.visible(:) = false;%make world 2 invisible
        else
            vr.currentWorld = 1;
            vr.worlds{vr.currentWorld}.surface.visible(:) = true;%make world 1 visible
        end  
        %vr.position(2) = 0; % set the animal’s y position to 0
        if vr.currentGain == 1; 
            vr.currentGain = 2; % set the current world
        else
            vr.currentGain = 1;
        end  
    end
    
    
    
    
    
    


% if vr.isReward
%     vr.numRewards = vr.numRewards + 1;  
%     
% end





% --- TERMINATION code: executes after the ViRMEn engine stops.
function vr = terminationCodeFun(vr)
