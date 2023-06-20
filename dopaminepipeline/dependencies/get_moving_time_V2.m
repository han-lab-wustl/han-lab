function moving_middle = get_moving_time_V2(velocity,thres,Fs)

%% It returns time pts when the animal is considered moving based on animal's change in y position
% position - position on track
% thres - Threshold speed in cm/s
% Fs - number of frames length minimum to be considered stopped.
%%
vr_speed = velocity;
vr_thresh = thres;
moving = find(vr_speed > vr_thresh);
stop = find(vr_speed <= vr_thresh);

stop_time_stretch =consecutive_stretch(stop);
clear stop moving
for i = 1:length(stop_time_stretch)
    stop_time_length(i) = length(stop_time_stretch{i});
end
delete_idx = stop_time_length<Fs;
stop_time_stretch(delete_idx) = [];
if length(stop_time_stretch) > 0
stop = cell2mat(stop_time_stretch');
moving_time = ones(1,length(vr_speed));
moving_time(stop) = 0;
else
    moving_time = ones(1,length(vr_speed));
end
moving = find(moving_time == 1);
moving_middle = moving;


