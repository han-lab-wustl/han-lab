% Created 20200115 by Jiaxin Cindy Tu (tu.j@wustl.edu)
%view angle output replaced by ROE count. %EH
%200611. xpos replace by lick sensor output, input into ai2.
%2000920. image synch TTL from scanbox, input into ai5. ai3 gives crosstalk
%with lick sensor in ai2. had to make imageSync global to save in exp.
% movement function for rotary encoder with arduino board
% [movement, movementType] = movementFunction(vr) % this is the format for
% any movment function, default movement type is 'velocity' if only one
% output
%modified to output raw ROE count instead of view angle
%200311. modified toreac contact lick sensor input to DAQ
function [movement,movementType] = moveWithROE_output_ROE_conLick_imageSync(vr)

movement = [0 0 0 0];

% make sure you reset the count at the initiation code

% Read data from arduino

if strcmp(vr.movementType, 'displacement')%'displacement' or 'velocity'
    if ~isfield(vr,'scaling')
        vr.scaling = -0.013; % tested for 1.8m track. ball cir = 21"=53.34cm. 3.37rot = 1.8m
    end
    [vr.count,vr.timeROE] = readCount(vr.encoder,'Reset',true); % read the count value from the last reset
    movement(2) = vr.scaling*vr.count; % only y updated
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
vr.ROE_outputScaling=0.004; %0.004% %only 2/3/2020 uses 0.001 raw ROE counts max near + -  800. # moved from  one of the initializingDAQ functions to make it comparable to all
movementType = vr.movementType;

if vr.iterations ==1
    vr.count = 0;
    movement(2) = 0;
end

vr.moveData = inputSingleScan(vr.moveSession);
vr.currentLick=vr.moveData(vr.conLicChannel);
vr.imageSync=vr.moveData(vr.imageSyncChannel); %2000920

global lickSensor %EB
lickSensor=vr.currentLick; %EB

global imageSync %200920
imageSync=vr.imageSync; %200920


if vr.currentLick >= 10 || vr.currentLick <= -10 % crashes output if > 10 %was vr.currentLick
        vr.currentLick = 0;
end
if vr.count*vr.ROE_outputScaling >= 10 || vr.count*vr.ROE_outputScaling <= -10 % crashes output if > 10 %was vr.currentLick
        vr.count = 0;
end

% if vr.currentLick < vr.lickThreshold 
%         sound(1)
% end

% send output to recording.
outputSingleScan(vr.moveSession, [vr.count*vr.ROE_outputScaling,vr.currentLick,(vr.position(2)*vr.yScaling+vr.yOffset)]);

