% Created 20200115 by Jiaxin Cindy Tu (tu.j@wustl.edu)
%view angle output replaced by ROE count. %EH
% movement function for rotary encoder with arduino board
% [movement, movementType] = movementFunction(vr) % this is the format for
% any movment function, default movement type is 'velocity' if only one
% output
%modified to output raw ROE count instead of view angle
%200311. modified toreac contact lick sensor input to DAQ
function [movement,movementType] = moveWithROE_output_ROE_contactLick(vr)
global islick
movement = [0 0 0 0];

% make sure you reset the count at the initiation code

% Read data from arduino

if strcmp(vr.movementType, 'displacement')
    if ~isfield(vr,'scaling')
        vr.scaling = -0.013; % tested for 1.8m track. ball cir = 21"=53.34cm. 3.37rot = 1.8m
    end
    vr.count = readCount(vr.encoder,'Reset',true); % read the count value from the last reset
    movement(2) = vr.scaling*vr.count; % only y updated
elseif strcmp(vr.movementType,'velocity')
    if ~isfield(vr,'scaling')
        vr.scaling = -0.5; % this works about right p.s.the wheel is 20 inches in circumfirence
    end
    rpm = readSpeed(vr.encoder); % speed in revolutions per minute    
    movement(2) = vr.scaling(1)*rpm; % only y updated
end

vr.ROE_outputScaling=0.004; %0.004% %only 2/3/2020 uses 0.001 raw ROE counts max near + -  800. # moved from  one of the initializingDAQ functions to make it comparable to all
movementType = vr.movementType;

if vr.iterations ==1
    vr.count = 0;
    movement(2) = 0;
end

if isfield(vr,'conLick')
%     vr.currentlick = readVoltage(vr.licksensor,'A0');
    vr.currentlick = vr.conLick;
    if vr.currentlick < vr.lickThreshold
        islick = 1;
    else
        islick = 0;
    end
else
    vr.currentlick = 0;
    islick =0;
end
% if vr.islick
%     sound(1) 
% end

% send output to recording.
outputSingleScan(vr.moveSession, [vr.count*vr.ROE_outputScaling,vr.currentlick,vr.position(2)*vr.yScaling+vr.yOffset]);


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
