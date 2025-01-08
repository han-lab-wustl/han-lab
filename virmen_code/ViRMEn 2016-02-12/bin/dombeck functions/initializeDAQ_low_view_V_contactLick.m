function vr = initializeDAQ_low_view_V_contactLick(vr)
% 200310. add vr.contactLick as ai2 from contact lick sensor into ai2
%200120. add vr.ROE_outputScaling to output ROE count
%send 1/10 view angle V out, -1 to 1 instead of -10 to 10
%attempt to decrease noise into lick channel, which shows big transients
%that very with view angle voltage. 191005 EH
%Added JPB BAR, Nidaq data session1
vr.moveSession = daq.createSession('ni');
vr.waterSession = daq.createSession('ni');
% vr.lickSession = daq.createSession('ni');

%INPUTS
vr.ballForward = addAnalogInputChannel(vr.moveSession, 'Dev1', 'ai0', 'Voltage');
vr.ballRotation = addAnalogInputChannel(vr.moveSession, 'Dev1', 'ai1', 'Voltage');
vr.conLick = addAnalogInputChannel(vr.moveSession, 'Dev1', 'ai2', 'Voltage');%"vr.contactLick" gives BaseClass.gt error

%OUPUTS
vr.waterReward = addAnalogOutputChannel(vr.waterSession, 'Dev1','ao0','Voltage');
vr.ballYaw = addAnalogOutputChannel(vr.moveSession, 'Dev1','ao1','Voltage');
vr.ballX = addAnalogOutputChannel(vr.moveSession, 'Dev1','ao2','Voltage');
vr.ballY = addAnalogOutputChannel(vr.moveSession, 'Dev1','ao3','Voltage');

%Variables to be used for output
vr.ballForwardChannel = 1;
vr.ballRotationChannel = 2;
vr.conLickChannel = 3;


vr.xScaling = 18/str2double(vr.exper.variables.trackWidth);
vr.xOffset = 0; %Use all of dynamice range -10 V to 10V, offset to start at -9
vr.yScaling = 18/str2double(vr.exper.variables.trackLength);
vr.yOffset = -9;%Use all of dynamice range -10 V to 10V, offset to start at -9
% vr.yOffset = -10;%Use all of dynamice range -10 V to 10V, offset to start at -9
% vr.angleScaling = 9/pi;
vr.angleScaling = 0.9/pi;%-1 to 1 volt output, i think. 191025 EH
% vr.ROE_outputScaling=0.004; %raw ROE counts max near + -  800.

%vr.highVoltage = 5;
vr.highVoltage = 9.6;%grass stim sd9 that is triggered by 9.5V
vr.lowVoltage = 0;