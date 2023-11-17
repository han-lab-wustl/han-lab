

function cell_activity = get_spatial_tuning_per_cell(Fc3,velocity,position, ...
    Fs,nBins,track_length)
%% Fc3 = dFF of 1 cell 
% position - position of animal on track 
% Fs - Frame rate of acquisition
% nBins - number of bins in which you want to divide the track into 
% track_length - Length of track

%%
% zahra hard coded to be consistent with the dopamine pipeline
thres = 5; % 5 cm/s is the velocity filter, only get 
% frames when the animal is moving faster than that
ftol = 10; % number of frames length minimum to be considered stopped
[time_moving,~] = get_moving_time_V3(velocity, thres, Fs, ftol);
bin_size = track_length/nBins;
pos_moving = position(time_moving);

for i = 1:nBins
    time_in_bin{i} = time_moving(pos_moving > (i-1)*bin_size & ...
        pos_moving <= i*bin_size);
end 

for bin = 1:nBins
   cell_activity(bin) = mean(Fc3(time_in_bin{bin})); 
end



