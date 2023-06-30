function [place_cells_redetected,place_cells_false_idx,all_cells_redetected,all_false_idx] = do_bootstrap_shuffle_2p_parfor(position, allcellsactivity_fc3,actual_place_cells,nbins,track_length )
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
parfor j = 1:nshuffles 
     disp(['Shuffle number ', num2str(j)]) 
    shuffled_place_cells = [];
    for i = 1:size(all_cells_activity,1)
        shuffledbins_forcell = shuffle(bins2shuffle_forcell{i}); 
        shuffled_cells_activity(i,:) = all_cells_activity(i,cell2mat(shuffledbins_forcell));
       
    end 
    
[~, shuffled_place_cells, ~ , ~, ~, ~,~] = get_putative_place_cells...
    (pos_moving, shuffled_cells_activity',nbins,track_length) ;
dline = zeros(1,size(all_cells_activity,1));
dline(shuffled_place_cells) = 1;
is_shuffled_place_cell(j,:) = dline;
shuffled_cells_activity = [];
end

all_cells_redetected = sum(is_shuffled_place_cell,1);
all_false_idx = find(all_cells_redetected>0.05*nshuffles);
place_cells_shuffles = is_shuffled_place_cell(:,actual_place_cells);
place_cells_redetected = sum(place_cells_shuffles,1); %total number of shuffles it passed as a place cells
place_cells_false_idx = find(place_cells_redetected > 0.05*nshuffles); %p value for place cells
