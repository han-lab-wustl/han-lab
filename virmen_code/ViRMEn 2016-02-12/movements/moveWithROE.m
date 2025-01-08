% Created 20200115 by Jiaxin Cindy Tu (tu.j@wustl.edu)
% movement function for rotary encoder with arduino board
% [movement, movementType] = movementFunction(vr) % this is the format for
% any movment function, default movement type is 'velocity' if only one
% output
function [movement,movementType] = moveWithROE(vr)
movement = [0 0 0 0];

% make sure you reset the count at the initiation code

% Read data from arduino

if strcmp(vr.movementType, 'displacement')
    if ~isfield(vr,'scaling')
        vr.scaling = -0.008; % this works about right p.s.the wheel is 20 inches in circumfirence
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


movementType = vr.movementType;
% send output to recording
outputSingleScan(vr.moveSession, [(mod(vr.position(4)+pi,2*pi) - pi)*vr.angleScaling,vr.position(1)*vr.xScaling+vr.xOffset,vr.position(2)*vr.yScaling+vr.yOffset]);


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
