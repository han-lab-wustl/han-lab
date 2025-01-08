vr.arduino = arduino('COM3','Uno','Libraries',{'I2C','rotaryEncoder'});
chA = 'D2'; % whichever line the black wire connects to
chB = 'D3'; % whichever line the white wire connects to
chZ = 'D4'; % whichever line the orange wire connects to
vr.chSol2 = 'D9';
ppr = 1024; % reads off from the rotary encoder (pulse per revolution)
vr.encoder = rotaryEncoder(vr.arduino,chA,chB,ppr); %
vr.movementType = 'displacement'; %'displacement' or 'velocity'
resetCount(vr.encoder,0); % reset count to 0
configurePin(vr.arduino,vr.chSol2,'DigitalOutput') %configure pin to send ttl to solenoid
devaddress = scanI2CBus(vr.arduino,0);
vr.optodeviceObj = i2cdev(vr.arduino,devaddress{1});
write(vr.optodeviceObj,[1 0],'uint16')
