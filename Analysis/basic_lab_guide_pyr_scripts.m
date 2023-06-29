
%% Load the data
clearvars -except Pyrs
position = Pyrs(1).R1.day{1}.F1.ypos;
fwd = Pyrs(1).R1.day{1}.F1.fwd;
Fc3 = calc_Fc3(Pyrs(1).R1.day{1}.F1.dFF);
Fs = Pyrs(1).R1.day{1}.Fs;
nBins = 80; 
track_length = 180; 
position = linmap(position, [0,track_length]);

cell_activity = get_spatial_tuning_all_cells(Fc3,position,Fs,nBins,track_length);
[cellIdx, maxBin] = get_sorted_cells_idx(cell_activity);

figure; 
imagesc(cell_activity(cellIdx,:));
%% Velocity as a function of position

Y_smoothed = get_one_variable_wrt_position(fwd,position,Fs,nBins,track_length);
figure; 
plot(Y_smoothed)
%% Spatial Info
cell_info = get_spatial_info_all_cells(Fc3,position,Fs,nBins,track_length);
figure; 
histogram(cell_info,'Normalization','probability')
%% Spatial sparsity
cell_sparsity = get_spatial_sparsity_all_cells(Fc3,position,Fs,nBins,track_length);
figure; 
histogram(cell_sparsity,'Normalization','probability')

figure;
scatter(cell_sparsity, cell_info)
%% Place cells
[place_cells,total_place_cells,false_cells_num] = get_placecell_main(Fc3,position,nBins,track_length,Fs);

figure; 
imagesc(cell_activity(place_cells,:));

figure; 
histogram(cell_info(place_cells),'Normalization','probability')




