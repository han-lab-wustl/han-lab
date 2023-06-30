function [place_cells_redetected,false_idx] = do_bootstrap_shuffle_2p(position, allcellsactivity_fc3,actual_place_cells,nbins,track_length )
%%
% position - moving time position
% allcellsactivity_fc3 - moving cell's activity - watch the dimensions
% actual place cells - putative place cells


%%


nshuffles = 1000;
all_cells_activity = allcellsactivity_fc3';

pos_moving = position;

for i = 1:size(all_cells_activity,1)
    
    bins2shuffle_forcell{i} = shuffling_bins(all_cells_activity(i,:));
    
end 

is_shuffled_place_cell = zeros(nshuffles,size(allcellsactivity_fc3,2));
for j = 1:nshuffles
     disp(['Shuffle number ', num2str(j)]) 
    clearvars shuffled_place_cells
    for i = 1:size(all_cells_activity,1)
        shuffledbins_forcell{i} = shuffle(bins2shuffle_forcell{i}); 
        shuffled_cells_activity(i,:) = all_cells_activity(i,cell2mat(shuffledbins_forcell{i}));
    end 
    
[~, shuffled_place_cells, ~ , ~, ~, ~,~] = get_putative_place_cells...
    (pos_moving, shuffled_cells_activity',nbins,track_length) ;
is_shuffled_place_cell(j,shuffled_place_cells) = 1;
end

place_cells_shuffles = is_shuffled_place_cell(:,actual_place_cells);
place_cells_redetected = sum(place_cells_shuffles,1);
false_idx = find(place_cells_redetected > 0.05*nshuffles);
