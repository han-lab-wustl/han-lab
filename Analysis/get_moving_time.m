function moving_middle = get_moving_time(position,thres,Fs)

%% It returns time pts when the animal is considered moving based on animal's change in y position
% position - position on track
% thres - Threshold speed in cm/s
% Fs - Frame rate
%%
vr_speed(2:length(position)) = abs(diff(position));
vr_speed(1) = vr_speed(2);
vr_thresh = thres/(Fs);
moving = find(vr_speed > vr_thresh );
stop = find(vr_speed <= vr_thresh);

stop_time_stretch =consecutive_stretch(stop);
clear stop moving
for i = 1:length(stop_time_stretch)
    stop_time_length(i) = length(stop_time_stretch{i});
end
delete_idx = stop_time_length<Fs;
stop_time_stretch(delete_idx) = [];
stop = cell2mat(stop_time_stretch);
moving_time = ones(1,length(vr_speed));
moving_time(stop) = 0;
moving = find(moving_time == 1);
pos_moving = position(moving);
moving_middle = moving(pos_moving > 3 & pos_moving < 177);


