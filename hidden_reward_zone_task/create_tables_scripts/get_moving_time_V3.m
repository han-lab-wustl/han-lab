function [moving_middle, stop] = get_moving_time_V3(velocity,thres,Fs,ftol)

%% It returns time pts when the animal is considered moving based on animal's change in y position
% velocity - forward vel
% thres - Threshold speed in cm/s
% Fs - number of frames length minimum to be considered stopped.
% ftol - 10 frames 
% zd made edits to change direction of stop array to prevent crashing,
% 9/14/2023
%%
vr_speed = velocity;
vr_thresh = thres;
% vr_thresh= prctile(vr_speed,35);
moving = find(vr_speed > vr_thresh);
stop = find(vr_speed <= vr_thresh);

stop_time_stretch =consecutive_stretch(stop);

for i = 1:length(stop_time_stretch)
    stop_time_length(i) = length(stop_time_stretch{i});
end
delete_idx = stop_time_length<Fs;
stop_time_stretch(delete_idx) = [];

if length(stop_time_stretch) > 0
    for s = 1:length(stop_time_stretch)-1
%         scatter(stop_time_stretch{s},velocity(stop_time_stretch{s}),20,'y','filled')
        d = 1;
        if s+d<length(stop_time_stretch) % what is the point of this
            if ~isnan(stop_time_stretch{s+d})
                while abs(stop_time_stretch{s}(end)-stop_time_stretch{s+d}(1))<=ftol&&s+d<length(stop_time_stretch)
                    stop_time_stretch{s}=[stop_time_stretch{s} (stop_time_stretch{s}(end)+1:stop_time_stretch{s+d}(1)-1) stop_time_stretch{s+d}];
                    stop_time_stretch{s+d} = NaN;
                    d=d+1;
                   
                end
            end
        end
    end
%     stop_time_stretch(cellfun(@(x) isnan(x),stop_time_stretch,'UniformOutput',0)) = [];
    stop_time_stretch(cellfun(@(x) any(isnan(x)),stop_time_stretch)) = [];
    stop = cell2mat(stop_time_stretch);
    moving_time = ones(1,length(vr_speed));
    moving_time(stop) = 0;
else
    moving_time = ones(1,length(vr_speed));
end
moving = find(moving_time == 1);
moving_middle = moving;






