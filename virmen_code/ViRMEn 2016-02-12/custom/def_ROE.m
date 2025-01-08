vr.arduino = arduino('COM3','Uno','Libraries','rotaryEncoder');
chA = 'D2'; % whichever line the black wire connects to
chB = 'D3'; % whichever line the white wire connects to
chZ = 'D4'; % whichever line the orange wire connects to
ppr = 1024; % reads off from the rotary encoder (pulse per revolution)
vr.encoder = rotaryEncoder(vr.arduino,chA,chB,ppr); %
vr.movementType = 'displacement'; %'displacement' or 'velocity'
resetCount(vr.encoder,0); % reset count to 0

