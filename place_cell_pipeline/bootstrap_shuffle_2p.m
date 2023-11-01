function [place_cells_redetected,false_idx] = bootstrap_shuffle_2p(time_moving,position, ...
    allcellsactivity_fc3,actual_place_cells)

nshuffles = 1000;
% [place_cell_num , actual_place_cells, field_area,peak_bin,pf_new] = place_cells(mouseCPP,position, allcellsactivity_fc3, mouse_num,vel_threshold ) ;
all_cells_activity = allcellsactivity_fc3;
% for i = 1:size(all_cells_activity,2)/bin_size
% bin{i} = all_cells_activity(:,(i-1)*bin_size + 1 : bin_size*i);
% end 

for i = 1:size(all_cells_activity,1)    
    bins2shuffle_forcell{i} = shuffling_bins(all_cells_activity(i,:));    
end 

is_shuffled_place_cell = zeros(nshuffles,size(fc3,1));

for j = 1:nshuffles
    disp(j)
    clearvars shuffled_place_cells
    for i = 1:size(all_cells_activity,1)
        shuffledbins_forcell{i} = shuffle(bins2shuffle_forcell{i}); 
        shuffled_cells_activity(i,:) = all_cells_activity(cell2mat(shuffledbins_forcell{i}),i);
    end 
    
%     figure; 
%     subplot(2,1,1)
%     imagesc(all_cells_activity)
%     subplot(2,1,2)
%     imagesc(shuffled_cells_activity)
    
    
% for i = 1:size(bin,2)
% shuffled_cells_activity(:,((i-1)*bin_size+1):bin_size*i) = shuffled_bins{i};  
% end
    % shuffled_cells_activity = shuffled_cells_activity';
    [~, shuffled_place_cells, ~, ...
        ~, ~, ~,shuffled_cell_activity_smoothed_norm{j}] = place_cells(time_moving,position, ...
        shuffled_cells_activity) ;
    is_shuffled_place_cell(j,shuffled_place_cells) = 1;
end

place_cells_shuffles = is_shuffled_place_cell(:,actual_place_cells);
place_cells_redetected = sum(place_cells_shuffles,1);
false_idx = find(place_cells_redetected > 0.05*nshuffles);


% not_place_cells = actual_place_cells(false_idx);

%  figure; imagesc(is_shuffled_place_cell)
%  figure; plot(place_cells_redetected)

