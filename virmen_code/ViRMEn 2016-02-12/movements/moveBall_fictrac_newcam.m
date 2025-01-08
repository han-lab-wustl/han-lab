function velocity = moveBall_fictrac_newcam(vr)
%190705. for fictrac ball tracking.
%voltage out through "LV_client_port_DAQ.vi" TCP read must be at 1024
%output lags a bit though. need to make faster if possible
%gain out from .vi at: 1800 for both for and yaw
%uses 8" diameter smooth styrofoam ball with spot pattern.


velocity = [0 0 0 0];
if ~isfield(vr,'scaling')
    %1st is yaw, 2nd is forward
    %vr.scaling = [30 30];%orig
    vr.scaling = [0.56 24];%fictrac scaling, new cam, for and yaw constant at 1800, up gain, set on VR2. 191111
%     vr.scaling = [0.165 11];%fictrac scaling. 190705. 
%      vr.scaling = [0.80 67];%old mouse gains. 180629. strange. 2.75 rotations forward(1.8m). 360deg is 7.5 yaw
%     vr.scaling = [0.16 15];%new sensor with arduino. tested with cylTrans, new track. may be diff with older versions
       %vr.scaling = [.8 90];%2.0 rotations forward(1.3m). 360deg is 7.5 yaw
    %vr.scaling = [.8 75];%2.3 rotations forward(1.5m). 360deg is 7.5 yaw

    
end
%ptr = get(0,'pointerlocation')-scr(3:4)/2;
%scaling is 7.5 rotations is 360 degrees and 2.75 forward is 1.8 m
vr.moveData = inputSingleScan(vr.moveSession);
%x
%velocity(1) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*-sin(vr.position(4));
velocity(1) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*-sin(vr.position(4));
%y
%velocity(2) = vr.moveData(vr.ballForwardChannel)*vr.scaling(1)*cos(vr.position(4));
velocity(2) = vr.moveData(vr.ballForwardChannel)*vr.scaling(2)*cos(vr.position(4));
%velocity(4) = vr.moveData(vr.ballRotationChannel)*.1;%ptr(2)*vr.scaling(2)/500;
%velocity(4) = vr.moveData(vr.ballRotationChannel);
velocity(4) = ((-vr.moveData(vr.ballRotationChannel))*(vr.scaling(1)));
%velocity(4) = vr.moveData(vr.ballRotationChannel)*.1;%problems turning
outputSingleScan(vr.moveSession, [(mod(vr.position(4)+pi,2*pi) - pi)*vr.angleScaling,vr.position(1)*vr.xScaling+vr.xOffset,vr.position(2)*vr.yScaling+vr.yOffset]);
%velocity(1:2) = [cos(vr.position(4)) -sin(vr.position(4)); sin(vr.position(4)) cos(vr.position(4))]*velocity(1:2)';