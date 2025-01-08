function [velocity] = moveWithKeyboard_conLick_imageSync230130_GM(vr)
% Keyboard control movement function for ViRMEn
%   Left/Right: change view angle
%   CTRL + Left/Right: move left/right
%   Up/Down: move forward/backward
%   CTRL + Up/Down: move up/down

persistent keyboardControl
global roe timeROE %EB
movement = [0 0 0 0];

if ~isfield(keyboardControl,'forward')
    keyboardControl.forward = 0;
    keyboardControl.rotation = 0;
    keyboardControl.sideways = 0;
    keyboardControl.vertical = 0;
end

linearScale = 30;
rotationScale = 2;

movementType = vr.movementType;

switch vr.keyPressed
    case 262
        if vr.modifiers == 0
            keyboardControl.rotation = -rotationScale;
        elseif vr.modifiers == 2
            keyboardControl.sideways = linearScale;
        end
    case 263
        if vr.modifiers == 0
            keyboardControl.rotation = rotationScale;
        elseif vr.modifiers == 2
            keyboardControl.sideways = -linearScale;
        end
    case 264
        if vr.modifiers == 0
            keyboardControl.forward = -linearScale;
        elseif vr.modifiers == 2
            keyboardControl.vertical = -linearScale;
        end
    case 265
        if vr.modifiers == 0
            keyboardControl.forward = linearScale;
        elseif vr.modifiers == 2
            keyboardControl.vertical = linearScale;
        end
end
switch vr.keyReleased
    case {262, 263}
        keyboardControl.rotation = 0;
        keyboardControl.sideways = 0;
    case {264, 265}
        keyboardControl.forward = 0;
        keyboardControl.vertical = 0;
end
vr.ROE_outputScaling=0.004; %0.004% %only 2/3/2020 uses 0.001 raw ROE counts max near + -  800. # moved from  one of the initializingDAQ functions to make it comparable to all

vr.moveData = inputSingleScan(vr.moveSession);
vr.currentLick=vr.moveData(vr.conLicChannel);
vr.imageSync=vr.moveData(vr.imageSyncChannel); %2000920

global lickSensor %EB
lickSensor=vr.currentLick; %EB

global imageSync %200920
imageSync=vr.imageSync; %200920

roe = keyboardControl.forward;
timeROE = vr.dt;

% send output to recording.
outputSingleScan(vr.moveSession, [keyboardControl.forward*vr.ROE_outputScaling,vr.currentLick,(vr.position(2)*vr.yScaling+vr.yOffset)]);

velocity = [keyboardControl.forward*[sin(-vr.position(4)) cos(-vr.position(4))]+keyboardControl.sideways*[cos(vr.position(4)) sin(vr.position(4))] ...
    keyboardControl.vertical keyboardControl.rotation];