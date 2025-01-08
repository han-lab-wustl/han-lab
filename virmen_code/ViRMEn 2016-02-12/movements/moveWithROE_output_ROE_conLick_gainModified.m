% Created 20200115 by Jiaxin Cindy Tu (tu.j@wustl.edu)
%view angle output replaced by ROE count. %EH
%200611. xpos replace by lick sensor output, input into ai2.
% movement function for rotary encoder with arduino board
% [movement, movementType] = movementFunction(vr) % this is the format for
% any movment function, default movement type is 'velocity' if only one
% output
%modified to output raw ROE count instead of view angle
%200311. modified toreac contact lick sensor input to DAQ
function [movement,movementType] = moveWithROE_output_ROE_conLick_gainModified(vr)
global islick scalingFACTOR 

% global currentLick
movement = [0 0 0 0];

% make sure you reset the count at the initiation code

% Read data from arduino

if strcmp(vr.movementType, 'displacement')%'displacement' or 'velocity'
    if ~isfield(vr,'scaling')
        vr.scaling = -0.013; % tested for 1.8m track. ball cir = 21"=53.34cm. 3.37rot = 1.8m
    end
    [vr.count,vr.timeROE] = readCount(vr.encoder,'Reset',true); % read the count value from the last reset
    movement(2) = vr.scaling*vr.count*scalingFACTOR; % only y updated
elseif strcmp(vr.movementType,'velocity')
    if ~isfield(vr,'scaling')
        vr.scaling = -0.5; % this works about right p.s.the wheel is 20 inches in circumfirence
    end
    rpm = readSpeed(vr.encoder); % speed in revolutions per minute  %EB vr.  
    movement(2) = vr.scaling(1)*rpm; % only y updated
end
global roe timeROE %EB
if strcmp(vr.movementType, 'displacement')
    roe=vr.count;
    timeROE=vr.timeROE;
elseif strcmp(vr.movementType, 'velocity')
    roe=readSpeed(vr.encoder);
end
vr.ROE_outputScaling=0.01; %0.004% %only 2/3/2020 uses 0.001 raw ROE counts max near + -  800. # moved from  one of the initializingDAQ functions to make it comparable to all
movementType = vr.movementType;

if vr.iterations ==1
    vr.count = 0;
    movement(2) = 0;
end

%from orig move file
vr.moveData = inputSingleScan(vr.moveSession);
%x
%velocity(1) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*-sin(vr.position(4));
% velocity(1) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*-sin(vr.position(4));
% %y
% %velocity(2) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*cos(vr.position(4));
% velocity(2) = vr.moveData(vr.ballForwardChannel)*vr.scaling(2)*cos(vr.position(4));
% %velocity(4) = vr.moveData(vr.ballRotationChannel)*.1;%ptr(2)*vr.scaling(2)/500;
% %velocity(4) = vr.moveData(vr.ballRotationChannel);
% velocity(4) = ((-vr.moveData(vr.ballRotationChannel))*(vr.scaling(1)));
% %velocity(4) = vr.moveData(vr.ballRotationChannel)*.1;%problems turning
% % 1=angle, 2 =position x, 3=position Y
% outputSingleScan(vr.moveSession, [(mod(vr.position(4)+pi,2*pi) - pi)*vr.angleScaling,vr.position(1)*vr.xScaling+vr.xOffset,vr.position(2)*vr.yScaling+vr.yOffset]);
vr.currentLick=vr.moveData(vr.conLicChannel);

global lickSensor %EB
lickSensor=vr.currentLick; %EB


if lickSensor > 10 || lickSensor < -10 % crashes output if > 10 %was vr.currentLick
        vr.currentLick = 0;
end
if vr.count*vr.ROE_outputScaling > 10 || vr.count*vr.ROE_outputScaling < -10
    vr.ROE_outputScaling = 0;
end
% vr.lickData = inputSingleScan(vr.moveSession);
% vr.currentLick=vr.lickData(vr.conLicChannel);
% % currentLick=vr.currentLick;
%     if vr.currentLick > 10 % crashes output if > 10
%         vr.currentLick = 0;
%     end

% if isfield(vr,'conLic')
% %     vr.currentLick = readVoltage(vr.conLic,'A0');
%     vr.currentLick = double(vr.conLic);
%     vr.currentLick = vr.lickSession;
if vr.currentLick < vr.lickThreshold
        islick = 1;
        sound(1)
    else
        islick = 0;
end
% else
%     vr.currentLick = 0;
%     islick =0;
% end
% if vr.islick
%     sound(1) 
% end

% send output to recording.
if strcmp(vr.movementType, 'displacement')%'displacement' or 'velocity'
outputSingleScan(vr.moveSession, [vr.count*vr.ROE_outputScaling,vr.currentLick,vr.position(2)*vr.yScaling+vr.yOffset]);
elseif strcmp(vr.movementType,'velocity')
outputSingleScan(vr.moveSession, [rpm*vr.ROE_outputScaling,vr.currentLick,vr.position(2)*vr.yScaling+vr.yOffset]);
end
% outputSingleScan(vr.lickSession, [vr.currentLick]);

% function velocity = moveWithROE(vr)
% 
% velocity = [0 0 0 0];
% 
% % Read data from arduino
% rpm = readSpeed(vr.encoder); % speed in revolutions per minute
% 
% if ~isfield(vr,'scaling')
%     vr.scaling = -0.5; % this works about right p.s.the wheel is 20 inches in circumfirence
% end
% 
% velocity(2) = vr.scaling(1)*rpm; % only y updated
% 
