function [moving, stop] = vr_stop_and_moving_time(position)

%  vr_speed(1) = abs(f2(1,1)) ;
vr_speed(2:length(position)) = abs(diff(position));
vr_speed(1) = vr_speed(2);
moving = find(vr_speed > 0.05); % cm/s?
stop = find(vr_speed <= 0.05);
% histogram(speed)
% thresh = 0.3; 
% moving_time = find(speed>thresh);
% stop_time = find(speed<thresh);
stop_time_stretch =consecutive_stretch(stop);
clear stop moving
for i = 1:length(stop_time_stretch)
    stop_time_length(i) = length(stop_time_stretch{i});
end 
delete_idx = stop_time_length<7;
stop_time_stretch(delete_idx) = [];
stop = cell2mat(stop_time_stretch);
moving_time = ones(1,length(vr_speed));
moving_time(stop) = 0;
moving = find(moving_time == 1);

% figure; 
% s1 = subplot(2,1,1);
% hold on; plot(position)
% scatter(stop,position(stop,1));
% s2 = subplot(2,1,2);
% hold on; 
% plot(vr_speed)
% scatter(stop,vr_speed(stop));
% linkaxes([s1 s2],'x')
