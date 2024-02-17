function sparsity = get_spatial_sparsity_per_cell(Fc3,position,nBins,track_length,fv, thres, Fs, ftol)
%% Fc3 = dFF of 1 cell
% position - position of animal on track
% Fs - Frame rate of acquisition
% nBins - number of bins in which you want to divide the track into
% track_length - Length of track

%%

[time_moving,~] = get_moving_time_V3(fv, thres, Fs, ftol);
bin_size = track_length/nBins;
pos_moving = position(time_moving);

for i = 1:nBins
    time_in_bin{i} = time_moving(pos_moving > (i-1)*bin_size & pos_moving <= i*bin_size);
end

for bin = 1:nBins
    cell_activity(bin) = mean(Fc3(time_in_bin{bin}));
end

cell_activity = smoothdata(cell_activity,'gaussian',5);

for i = 1:nBins
time_fraction(i) = length(time_in_bin{i})/length(time_moving);
end 

temp1 = (time_fraction .* cell_activity);
temp1(temp1 == Inf) = 0;
temp1(temp1 == -Inf) = 0;
temp1(isnan(temp1) == 1) = 0;
sparse_nume = (sum(temp1))^2;

temp2 = (time_fraction .* (cell_activity.^2));
temp2(temp2 == Inf) = 0;
temp2(temp2 == -Inf) = 0;
temp2(isnan(temp2) == 1) = 0;
sparse_denom = sum(temp2);

sparsity = sparse_nume/sparse_denom;

if isnan(sparsity)
    sparsity = 0; 
end 