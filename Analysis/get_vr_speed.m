function vr_speed = get_vr_speed(position,Fs)

%% It returns speed in virtual world
% get_vr_speed(position,Fs)
% position - position on track
% Fs - Frame rate
%%
vr_speed(2:length(position)) = abs(diff(position));
vr_speed(1) = vr_speed(2);
vr_speed = vr_speed .*Fs;



