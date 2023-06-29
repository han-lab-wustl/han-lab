

function info = get_spatial_info_per_cell(Fc3,position,Fs,nBins,track_length)
%% Fc3 = dFF of 1 cell
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
    cell_activity(bin) = mean(Fc3(time_in_bin{bin}));
end

% cell_activity = smoothdata(cell_activity,'gaussian',5);

lambda_all = mean(Fc3(time_moving));
for i = 1:nBins
    time_fraction(i) = length(time_in_bin{i})/length(time_moving);
end

temp = (time_fraction .* cell_activity.* log2(cell_activity/lambda_all));
temp(temp == Inf) = 0;
temp(temp == -Inf) = 0;
temp(isnan(temp) == 1) = 0;
info = sum(temp/lambda_all);
if isnan(info)
    info = 0; 
end 
