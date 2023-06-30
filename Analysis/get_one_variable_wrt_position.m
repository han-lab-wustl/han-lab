

function Y_smoothed = get_one_variable_wrt_position(Y,position,Fs,nBins,track_length)
%% Y = Variable that you want to linearize with position 
% position - position of animal on track 
% Fs - Frame rate of acquisition
% nBins - number of bins in which you want to divide the track into 
% track_length - Length of track

%%

time_moving = get_moving_time(position,5,Fs);
bin_size = track_length/nBins;
pos_moving = position(time_moving);

for i = 1:nBins
    time_in_bin{i} = time_moving(pos_moving > (i-1)*bin_size & pos_moving <= i*bin_size);
end 

for bin = 1:nBins
   Y_lin(bin) = mean(Y(time_in_bin{bin})); 
end

Y_smoothed = smoothdata(Y_lin,'gaussian',5);


