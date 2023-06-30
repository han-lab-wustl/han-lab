function shuffled_cell_info = get_shuffled_cell_info(Fc3,position,Fs,nBins,track_length)
%%
% position - moving time position
% allcellsactivity_fc3 - moving cell's activity - watch the dimensions
% actual place cells - putative place cells


%%


nshuffles = 1000;
all_cells_activity = Fc3';

for i = 1:size(all_cells_activity,1)
    bins2shuffle_forcell{i} = shuffling_bins(all_cells_activity(i,:));   
end 

shuffled_cell_info = zeros(nshuffles,size(Fc3,2));
for j = 1:nshuffles
     disp(['Shuffle number ', num2str(j)]) 
    clearvars shuffled_place_cells
    for i = 1:size(all_cells_activity,1)
        shuffledbins_forcell{i} = shuffle(bins2shuffle_forcell{i}); 
        shuffled_cells_activity(i,:) = all_cells_activity(i,cell2mat(shuffledbins_forcell{i}));
    end 
    
shuffled_cells_activity = shuffled_cells_activity'; % nTimes X nCells
cell_info_shuffled = get_spatial_info_all_cells(shuffled_cells_activity,position,Fs,nBins,track_length);

shuffled_cell_info(j,:) = cell_info_shuffled;

clear shuffled_cells_activity
end